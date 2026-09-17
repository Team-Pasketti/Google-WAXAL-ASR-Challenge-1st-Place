#!/usr/bin/env python3
"""Beam-5 n-best over the WAXAL train audio, for noisy-student pseudo-labels.

The student is trained on Whisper's own output rather than WAXAL's
transcriptions. That is the point: WAXAL's annotators write `namoni` 3983 times
and `na moni` 1359, while the evaluation references split roughly 73% of the
time, so every model fitted to WAXAL labels inherits a convention that is
backwards for the test set. Labelling with Whisper sidesteps the annotators
entirely -- the student learns to transcribe, not to imitate a house style.

The learning signal comes from the augmentation applied to the *student's*
input, never here. The teacher reads clean audio; the student will read the
same audio speed-perturbed, frequency-masked and pitch-shifted, and must still
produce this transcript. That asymmetry is what makes it learnable: an earlier
attempt on this project pseudo-labelled 313k chunks with a same-architecture
teacher, no augmentation, and the student learned nothing because it could
already reproduce the labels.

Writes {id: {nbest, scores, seconds, language, speaker_id}} so the downstream
steps can rescore with Qwen, drop runaway decodes by duration, and hold out
speakers.

Sharding: run one process per GPU with --shard i --num-shards N and
CUDA_VISIBLE_DEVICES=i. Shards are disjoint by index and write separate files.
Every batch is flushed, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

SR = 16_000
MODEL = "Sunbird/asr-whisper-51-african-languages"
LANG_TOKEN = {"lin": "<|ln|>", "sna": "<|sn|>"}
LONG_SECONDS = 29.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / "kaggle-waxal-cleaned-16k-7lang",
                    help="local parquet tree; ignored when --repo is given")
    ap.add_argument("--repo", default="",
                    help="HF dataset to stream from instead, e.g. "
                         "enesssssw/waxal-cleaned-16k. Use this on a machine "
                         "that does not have the parquet tree locally.")
    ap.add_argument("--languages", nargs="+", default=["lin", "sna"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--nbest", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import Audio, load_dataset

    def shards_for(lang: str) -> list[str]:
        """Local parquet paths, or HF filenames downloaded on demand."""
        if not args.repo:
            return sorted(glob.glob(str(args.data_dir / f"{lang}_asr"
                                        / f"{args.split}-*.parquet")))
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        names = [f for f in api.list_repo_files(args.repo, repo_type="dataset")
                 if f"{lang}_asr/" in f and f"/{args.split}-" in f
                 and f.endswith(".parquet")]
        if not names:
            names = [f for f in api.list_repo_files(args.repo, repo_type="dataset")
                     if lang in f and args.split in f and f.endswith(".parquet")]
        print(f"  {lang}: {len(names)} shards on {args.repo}", flush=True)
        return [hf_hub_download(args.repo, n, repo_type="dataset")
                for n in sorted(names)]

    from transformers import (WhisperFeatureExtractor,
                              WhisperForConditionalGeneration,
                              WhisperTokenizerFast)
    # WhisperForConditionalGeneration.generate wraps beam search in long-form
    # segmentation that cannot return more than one sequence per clip; the plain
    # implementation gives an ordinary n-best. Long clips are handled separately
    # below rather than by that wrapper.
    from transformers.generation.utils import GenerationMixin
    plain = GenerationMixin.generate

    import transformers
    dtype_kw = ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
                else "torch_dtype")
    feat = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    tok = WhisperTokenizerFast.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, **{dtype_kw: torch.float16}).cuda().eval()
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    nots = tok.convert_tokens_to_ids("<|notimestamps|>")

    store: dict[str, dict] = {}
    if args.out.exists():
        store = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"resuming: {len(store)} clips already done", flush=True)

    t0 = time.time()
    for lang in args.languages:
        files = shards_for(lang)
        if not files:
            print(f"no shards for {lang}/{args.split}", flush=True)
            continue
        ds = load_dataset("parquet", data_files=files, split="train")
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        ref_col = next((c for c in ("transcription", "text", "sentence")
                        if c in ds.column_names), "transcription")
        if "id" not in ds.column_names:
            ds = ds.add_column("id", [f"{lang}_{args.split}{i}"
                                      for i in range(len(ds))])
        # disjoint slice for this GPU
        idx = [i for i in range(len(ds)) if i % args.num_shards == args.shard]
        print(f"{lang}/{args.split}: {len(ds)} clips, shard "
              f"{args.shard}/{args.num_shards} -> {len(idx)}", flush=True)

        prompt = [sot, tok.convert_tokens_to_ids(LANG_TOKEN[lang]),
                  transcribe, nots]
        pending = [i for i in idx if ds[i]["id"] not in store]
        for start in range(0, len(pending), args.batch):
            chunk = ds[pending[start:start + args.batch]]
            audio = [np.asarray(a["array"], dtype=np.float32)
                     for a in chunk["audio"]]
            secs = [len(a) / SR for a in audio]

            f = feat(audio, sampling_rate=SR,
                     return_tensors="pt").input_features.cuda().half()
            dec = torch.tensor([prompt] * f.shape[0], device=f.device)
            with torch.inference_mode():
                out = plain(model, input_features=f, decoder_input_ids=dec,
                            num_beams=args.num_beams,
                            num_return_sequences=args.nbest, do_sample=False,
                            max_new_tokens=200, output_scores=True,
                            return_dict_in_generate=True)
            texts = tok.batch_decode(out.sequences, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
            sc = out.sequences_scores.tolist()
            for j, uid in enumerate(chunk["id"]):
                lo = j * args.nbest
                store[uid] = {
                    "language": lang,
                    # speaker_id is what a speaker-disjoint holdout needs, and
                    # the reference is kept only for diagnostics -- the student
                    # is never trained on it, which is the whole point
                    "speaker_id": chunk.get("speaker_id", [None] * len(secs))[j],
                    "seconds": round(secs[j], 3),
                    "nbest": [t.strip() for t in texts[lo:lo + args.nbest]],
                    "scores": sc[lo:lo + args.nbest],
                    "waxal_reference": chunk.get(ref_col, [""] * len(secs))[j],
                }
            # flush every batch: this run is many hours and must never lose it
            args.out.write_text(json.dumps(store, ensure_ascii=False),
                                encoding="utf-8")
            if start % (args.batch * 50) == 0:
                n = len(store)
                rate = (n / max(time.time() - t0, 1e-9))
                todo = len(pending) - start
                print(f"  {lang} {n} done, {rate:.2f} clips/s, "
                      f"ETA {todo/max(rate,1e-9)/3600:.1f} h", flush=True)

    long_n = sum(1 for v in store.values() if v["seconds"] > LONG_SECONDS)
    runaway = sum(1 for v in store.values()
                  if len(v["nbest"][0]) / max(v["seconds"], 1e-9) > 20.0)
    print(f"\nwrote {args.out}: {len(store)} clips in "
          f"{(time.time()-t0)/3600:.2f} h")
    print(f"  over {LONG_SECONDS}s (truncated by the feature extractor): {long_n}")
    print(f"  runaway decodes to drop (>20 chars/sec): {runaway}")


if __name__ == "__main__":
    main()

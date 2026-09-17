#!/usr/bin/env python3
"""Generate Whisper n-best lists for the 892 test clips, for rescoring.

Two decode paths, chosen by duration, because the fast one is unsafe on long
audio:

  <= 30s  plain beam search via GenerationMixin, which returns a real n-best.
          WhisperForConditionalGeneration.generate wraps beam search in
          long-form segmentation that assumes one sequence per clip and
          crashes the moment num_return_sequences > 1.

  >  30s  Whisper's own generate, single hypothesis, no rescoring. 29 of the
          892 clips run past the 30-second window (longest 35.2s). The
          segmentation layer is exactly what handles those, and the bypass
          would silently transcribe only the first 30 seconds -- a mean 2.24s
          of speech dropped, on a metric that averages per utterance.

So the long clips are carried through unrescored rather than rescored on
truncated audio. They keep whatever the ordinary decode gives them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import (WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperTokenizerFast)

SR = 16_000
MODEL = "Sunbird/asr-whisper-51-african-languages"
LANG_TOKEN = {"lin": "<|ln|>", "sna": "<|sn|>"}
LONG_SECONDS = 29.0     # margin under Whisper's 30s window


def load_audio(path: Path) -> tuple[np.ndarray, float]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(1, dtype=np.float32)
    seconds = len(x) / sr
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32), seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path,
                    default=Path.home() / "waxal-newaudios/newaudios")
    ap.add_argument("--lid", type=Path,
                    default=Path.home() / "wasal_lm/lm/newaudios_announced3_predictions.csv")
    ap.add_argument("--lid-column", default="announced3_language")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--nbest", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--token", default="",
                    help="HF token for the gated SALT repo; "
                         "falls back to $HF_TOKEN")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers.generation.utils import GenerationMixin
    plain = GenerationMixin.generate

    rows = list(csv.DictReader(open(args.lid, encoding="utf-8-sig")))
    id_col = "ID" if "ID" in rows[0] else "id"
    lang_of = {r[id_col]: r[args.lid_column] for r in rows}
    ids = [r[id_col] for r in rows]

    # The base checkpoint is a gated repo, so it needs a token exactly as
    # model_whisper/run_inference_salt.py does. Accepted from --token or from
    # HF_TOKEN; passing None falls through to a cached `huggingface-cli login`,
    # so all three routes work and none of them is silent about failing.
    import os
    token = args.token or os.environ.get("HF_TOKEN") or None

    feat = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    tok = WhisperTokenizerFast.from_pretrained(MODEL, token=token)
    import transformers
    dtype_kw = ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
                else "torch_dtype")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, token=token, **{dtype_kw: torch.float16})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model = model.cuda().eval()
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    nots = tok.convert_tokens_to_ids("<|notimestamps|>")

    dur = {}
    for uid in ids:
        info = sf.info(str(args.audio_dir / f"{uid}.wav"))
        dur[uid] = info.frames / info.samplerate
    long_ids = [u for u in ids if dur[u] > LONG_SECONDS]
    print(f"{len(ids)} clips, {len(long_ids)} over {LONG_SECONDS}s "
          f"-> single-hypothesis path", flush=True)

    store: dict[str, dict] = {}
    t0 = time.time()
    for lang in LANG_TOKEN:
        prompt = [sot, tok.convert_tokens_to_ids(LANG_TOKEN[lang]),
                  transcribe, nots]
        group = [u for u in ids if lang_of.get(u) == lang and u not in set(long_ids)]
        for start in range(0, len(group), args.batch):
            chunk = group[start:start + args.batch]
            audio = [load_audio(args.audio_dir / f"{u}.wav")[0] for u in chunk]
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
            for j, uid in enumerate(chunk):
                lo = j * args.nbest
                store[uid] = {"language": lang,
                              "nbest": [t.strip() for t in texts[lo:lo + args.nbest]],
                              "scores": sc[lo:lo + args.nbest], "long": False}
            # flush every batch: this run is ~16 minutes of GPU and an
            # exception anywhere after it should never cost the whole thing
            args.out.write_text(json.dumps(store, ensure_ascii=False),
                                encoding="utf-8")
            if start % (args.batch * 20) == 0:
                n = len(store)
                print(f"  {lang} {n}/{len(ids)} "
                      f"{n/max(time.time()-t0,1e-9):.2f} clips/s", flush=True)

    # Long clips get both treatments and the safer result wins.
    #
    # Long-form decoding is the principled fix -- it reads the whole file
    # instead of the first 30 seconds -- but on this model it frequently runs
    # away: ID_QOYCEF returned 1303 characters of "Boutique. Boutique." for 35
    # seconds of audio where the truncated decode gave a clean 232. Losing the
    # 2.24s tail costs about 0.002; a repetition loop on one clip costs ~0.03
    # under per-utterance averaging, so truncation is the better failure.
    #
    # Rather than pick globally, take long-form when its output rate is
    # plausible and fall back to the truncated n-best when it is not.
    kept = fell_back = 0
    for uid in long_ids:
        lang = lang_of.get(uid)
        if lang not in LANG_TOKEN:
            continue
        audio, seconds = load_audio(args.audio_dir / f"{uid}.wav")
        prompt = [sot, tok.convert_tokens_to_ids(LANG_TOKEN[lang]),
                  transcribe, nots]

        trunc = feat([audio], sampling_rate=SR,
                     return_tensors="pt").input_features.cuda().half()
        dec = torch.tensor([prompt], device=trunc.device)
        with torch.inference_mode():
            out = plain(model, input_features=trunc, decoder_input_ids=dec,
                        num_beams=args.num_beams,
                        num_return_sequences=args.nbest, do_sample=False,
                        max_new_tokens=200, output_scores=True,
                        return_dict_in_generate=True)
        cands = [t.strip() for t in tok.batch_decode(
            out.sequences, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)]
        scores = out.sequences_scores.tolist()

        full = feat([audio], sampling_rate=SR, return_tensors="pt",
                    truncation=False,
                    padding="longest").input_features.cuda().half()
        with torch.inference_mode():
            # long-form predicts segment boundaries, so it requires timestamp
            # tokens; skip_special_tokens strips them back out of the text
            seq = model.generate(full, language=LANG_TOKEN[lang],
                                 task="transcribe", num_beams=args.num_beams,
                                 do_sample=False, return_timestamps=True)
        longtext = tok.batch_decode(seq, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0].strip()

        if longtext and len(longtext) / max(seconds, 1e-9) <= 20.0:
            store[uid] = {"language": lang, "nbest": [longtext],
                          "scores": [0.0], "long": True}
            kept += 1
        else:
            store[uid] = {"language": lang, "nbest": cands, "scores": scores,
                          "long": False}
            fell_back += 1
        args.out.write_text(json.dumps(store, ensure_ascii=False),
                            encoding="utf-8")
    print(f"long clips: {kept} kept long-form, {fell_back} fell back to "
          f"truncated (long-form looped)", flush=True)

    args.out.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    missing = [u for u in ids if u not in store]
    print(f"wrote {args.out}: {len(store)}/{len(ids)} clips "
          f"in {(time.time()-t0)/60:.1f} min; missing {len(missing)}")


if __name__ == "__main__":
    main()

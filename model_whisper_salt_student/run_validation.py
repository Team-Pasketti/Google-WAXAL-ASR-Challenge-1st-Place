#!/usr/bin/env python3
"""Score a student checkpoint on a held-out manifest.

Reports the metric two ways on purpose. The competition averages WER and CER
per utterance; pooling every character into one corpus ratio ranks checkpoints
differently, and on this project the two disagreed enough to matter -- two
runaway hypotheses out of 892 moved the leaderboard by 0.095 while the
corpus-level distance between the same two submissions was only 0.0219.

The honest split for this model is by speaker, not by clip. run_training.py
writes the held-out speaker ids and their clip ids to holdout.json in the
output directory; pass `--holdout` to restrict scoring to exactly those clips.
Clip-level holdout measures memorisation of voices the model already trained
on, which is the mistake WAXAL's own validation split makes -- 64 of its 65
Lingala speakers also appear in train.

usage:
    python model_whisper_salt_student/run_validation.py \
        --adapter model_whisper_salt_student/models/step00500 \
        --manifest ../input/valid_parakeet_final_cleaned2_ssd_2_langs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import score  # noqa: E402

SR = 16_000
BASE = "Sunbird/asr-whisper-51-african-languages"
LANG_TOKEN = {"lin": "<|ln|>", "sna": "<|sn|>"}


def infer_lang(path: str) -> str:
    p = path.replace("\\", "/").lower()
    return "sna" if "/sna" in p or "sna_" in Path(p).name else "lin"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, help="omit to score base SALT")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--holdout", type=Path,
                    help="holdout.json from training; restricts to its clips")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--feature-extractor", default="openai/whisper-large-v3")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import librosa
    from transformers import (WhisperFeatureExtractor,
                              WhisperForConditionalGeneration,
                              WhisperTokenizerFast)

    items = [json.loads(l) for l in
             args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.holdout and args.holdout.exists():
        keep = set(json.loads(args.holdout.read_text())["eval_clip_ids"])
        items = [it for it in items
                 if Path(it["audio_filepath"]).stem in keep]
        print(f"holdout: {len(items)} clips from {len(keep)} held-out ids")
    if args.limit:
        items = items[:args.limit]
    print(f"scoring {len(items)} utterances", flush=True)

    feat = WhisperFeatureExtractor.from_pretrained(args.feature_extractor)
    # Sunbird's tokenizer_config stores `extra_special_tokens` as a list, which
    # newer transformers expects to be a dict and crashes on. The tokens we
    # force below are standard Whisper ones, not additions, so discarding it
    # loses nothing.
    try:
        tok = WhisperTokenizerFast.from_pretrained(args.base,
                                                   extra_special_tokens={})
    except Exception:
        tok = WhisperTokenizerFast.from_pretrained(args.base)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.float16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
        model = model.merge_and_unload()
        print(f"merged adapter {args.adapter}")
    model = model.cuda().eval()

    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    notimestamps = tok.convert_tokens_to_ids("<|notimestamps|>")

    refs, hyps = [], []
    # batch within a language: forced_decoder_ids is per-batch, not per-sample
    by_lang: dict[str, list] = {}
    for it in items:
        by_lang.setdefault(it.get("language") or infer_lang(it["audio_filepath"]),
                           []).append(it)

    for lang, group in by_lang.items():
        forced = [(1, tok.convert_tokens_to_ids(LANG_TOKEN[lang])),
                  (2, transcribe), (3, notimestamps)]
        for i in range(0, len(group), args.batch):
            chunk = group[i:i + args.batch]
            audios = [librosa.load(x["audio_filepath"], sr=SR, mono=True)[0]
                      for x in chunk]
            fx = feat(audios, sampling_rate=SR,
                      return_tensors="pt").input_features.cuda().half()
            with torch.inference_mode():
                ids = model.generate(fx, forced_decoder_ids=forced,
                                     num_beams=args.num_beams, do_sample=False,
                                     max_new_tokens=200)
            for it, t in zip(chunk, tok.batch_decode(ids,
                                                     skip_special_tokens=True)):
                refs.append(it["text"])
                hyps.append(t.strip())
        print(f"  {lang}: {len(group)} done", flush=True)

    macro = score(refs, hyps)
    corpus = score(refs, hyps, macro=False)
    print(f"\nmacro  WER {macro['wer']:.4f}  CER {macro['cer']:.4f}  "
          f"score {macro['score']:.6f}   <- the competition metric")
    print(f"corpus WER {corpus['wer']:.4f}  CER {corpus['cer']:.4f}  "
          f"score {corpus['score']:.6f}")

    if args.out:
        args.out.write_text(json.dumps(
            {"adapter": str(args.adapter), "n": len(refs),
             "macro": macro, "corpus": corpus}, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

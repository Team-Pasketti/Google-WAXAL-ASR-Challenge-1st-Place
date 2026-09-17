#!/usr/bin/env python3
"""Materialise the WAXAL training audio in the layout Model 8 reads.

`proc_data/t01_download_and_preapre_data_for_training.py` prepares the same
source dataset for the other seven models, but it writes loose WAV files plus a
`markdown.jsonl` index, and it drops every column except id, audio, text and
duration. Model 8 needs two things that layout cannot give it:

  * `speaker_id`, because the holdout is speaker-disjoint. Holding out clips
    would measure memorisation of voices the model already trained on, which is
    the mistake WAXAL's own validation split makes -- 64 of its 65 Lingala
    speakers also appear in train.
  * random access to decoded audio during training, since every epoch applies a
    different random speed, pitch and mask to the same clip. Re-reading and
    re-decoding 23k WAVs per epoch is the slow way to do that.

So this writes parquet shards instead, one directory per language, which is the
layout `pseudo_label_teacher.py` and `run_training.py` glob for:

    <out>/lin_asr/train-00000-of-00001.parquet
    <out>/sna_asr/train-00000-of-00001.parquet

Same source dataset as t01 -- `enesssssw/waxal-cleaned-16k`, the 16 kHz cleaned
WAXAL upload -- so no new data enters the solution here; this is a re-shaping of
what the other models already use.

usage:
    python model_whisper_salt_student/prepare_data.py --out ../input/waxal_parquet
    # then pass that path as --data-dir to pseudo_label_teacher.py and run_training.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = "enesssssw/waxal-cleaned-16k"
LANGS = ["lin_asr", "sna_asr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="destination for <lang>_asr/<split>-*.parquet")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--languages", nargs="+", default=LANGS)
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="train is what Model 8 uses; add validation/test only "
                         "if you want them locally")
    ap.add_argument("--shard-size", type=int, default=2000,
                    help="rows per parquet shard; keeps single files under a "
                         "few GB so a failed write costs one shard")
    ap.add_argument("--token", default="",
                    help="HF token, if the source repo is private to you")
    args = ap.parse_args()

    from datasets import load_dataset

    args.out.mkdir(parents=True, exist_ok=True)
    for lang in args.languages:
        print(f"=== {lang}", flush=True)
        ds_all = load_dataset(args.repo, lang, token=args.token or None)
        lang_dir = args.out / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        for split in args.splits:
            if split not in ds_all:
                print(f"  no split {split!r}, skipping")
                continue
            ds = ds_all[split]
            cols = ds.column_names
            print(f"  {split}: {len(ds)} rows, columns {cols}")
            if "speaker_id" not in cols:
                # Not fatal: run_training.py falls back to None and the holdout
                # degenerates to clip-level. Say so loudly rather than let a
                # silently weaker validation split look like a real one.
                print("  WARNING: no speaker_id column -- the speaker-disjoint "
                      "holdout will fall back to clip-level, which measures "
                      "memorisation rather than generalisation")

            n = len(ds)
            total = max((n + args.shard_size - 1) // args.shard_size, 1)
            for i in range(total):
                shard = ds.shard(num_shards=total, index=i, contiguous=True)
                path = lang_dir / f"{split}-{i:05d}-of-{total:05d}.parquet"
                if path.exists():
                    print(f"    {path.name} exists, skipping")
                    continue
                shard.to_parquet(str(path))
                print(f"    wrote {path.name} ({len(shard)} rows)", flush=True)

    print(f"\ndone. pass --data-dir {args.out} to pseudo_label_teacher.py "
          "and run_training.py")


if __name__ == "__main__":
    main()

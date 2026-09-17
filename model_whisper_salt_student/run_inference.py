#!/usr/bin/env python3
"""Model 8 inference: SALT + noisy-student LoRA -> submission CSV.

Three stages, chained here so the member can be regenerated with one command.
They are separate scripts because the middle one needs a 14B LLM resident on
the GPU and the other two need Whisper, and running them in one process means
holding both.

  1. nbest_decode.py     beam 5, 5-best, language token forced per clip from
                         the LID routing file, LoRA adapter merged into SALT.
  2. qwen_rescore score  one forward pass per hypothesis through
                         AfriqueQwen-14B, saved to json.
  3. qwen_rescore apply  shallow fusion of the acoustic and LM scores, picks
                         one hypothesis per clip, writes ID,Target.

Unlike Model 7, this one forces the *correct* language token per clip
(lin -> <|ln|>, sna -> <|sn|>) rather than a deliberately mismatched one. The
adapter was trained with those tokens in the prompt, so the mismatch trick that
helps the zero-shot checkpoint does not transfer.

Output goes straight to submission/submission_whisper_salt_student.csv in the
ID,Target schema that proc_data/r03_ensemble_subms.py consumes -- no jsonl
conversion step needed for this member.

usage:
    python model_whisper_salt_student/run_inference.py \
        --audio-dir input/newaudios \
        --adapter model_whisper_salt_student/models/step00500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# lin=0.01 is deliberately almost zero. Rescoring Lingala measured -0.002 at
# w=0.3 and -0.0044 at w=0.05 on the leaderboard: its acoustic score is already
# the better judge and the LM only adds noise. Shona at 0.3 is worth about
# +0.002. The word bonus offsets both log-prob terms growing more negative with
# length, which biases the pick toward short hypotheses -- and under
# per-utterance averaging a deletion on a short clip is expensive.
WEIGHTS = "lin=per_token:0.01:0.0,sna=per_token:0.3:0.04"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    # RAW audio, deliberately -- not input/newaudios_cleaned_mono/ that the
    # other seven members read. This member was decoded from the unprocessed
    # competition audio on a separate machine, and that is the hypothesis set
    # inside the winning ensemble. Pointing this at the MSST-cleaned audio
    # produces a different member and will not reproduce the 0.7696 submission.
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "input/newaudios")
    ap.add_argument("--adapter", type=Path,
                    default=HERE / "models/step00500",
                    help="LoRA directory, or an HF repo id")
    ap.add_argument("--lid", type=Path,
                    default=ROOT / "postprocess/artifacts/language_predictions.csv")
    ap.add_argument("--qwen", default="McGill-NLP/AfriqueQwen-14B")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--nbest", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--work", type=Path, default=ROOT / "submission")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "submission/submission_whisper_salt_student.csv")
    ap.add_argument("--token", default="",
                    help="HF token for the gated SALT repo; "
                         "falls back to $HF_TOKEN")
    ap.add_argument("--skip-nbest", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    nbest = args.work / "nbest_salt_student.json"
    lmscore = args.work / "lmscore_salt_student.json"
    py = sys.executable

    if not args.skip_nbest:
        run([py, HERE / "nbest_decode.py",
             "--audio-dir", args.audio_dir, "--lid", args.lid,
             "--num-beams", args.num_beams, "--nbest", args.nbest,
             "--batch", args.batch, "--adapter", args.adapter,
             "--token", args.token, "--out", nbest])

    if not args.skip_score:
        run([py, HERE / "qwen_rescore.py", "score",
             "--nbest", nbest, "--model", args.qwen,
             "--load-4bit", "--batch", 16, "--out", lmscore])

    # --lincaps capitalises the first letter and closes with a stop, worth
    # about +0.004 on its own. It is applied here rather than in postprocessing
    # because the ensemble votes at character level: a member that disagrees
    # with the others on the first character of every clip loses those votes.
    run([py, HERE / "qwen_rescore.py", "apply",
         "--nbest", nbest, "--lm-scores", lmscore,
         "--lid", args.lid, "--weights", WEIGHTS, "--lincaps",
         "--out", args.out])

    print(f"\nwrote {args.out}", flush=True)
    print("this file is entry 8 of proc_data/r03_ensemble_subms.py "
          "(weight 10.0)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the exact winning postprocessed submission in one command.

The pipeline is deliberately proposal-and-verification based. Correlated local
checkpoints propose high-recall repairs; architecturally diverse ASR systems
independently decide which proposals survive. No reference transcript, manual
row edit, or generative text model is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE_MEMBER_NAMES = [
    "st500_rep.csv",
    "st500_raw.csv",
    "v2s200_rep.csv",
    "v2s300_rep.csv",
    "v2s00400_rep.csv",
    "v2s00500_rep.csv",
    "submission_v6_stage2.csv",
    "submission_whisper_b5_repaired_lincaps.csv",
    "A1_asrafrica_baseline.csv",
    "submission_whisper_raw.csv",
]
DIVERSE_MEMBER_NAMES = [
    "st500_raw.csv",
    "submission_test2_mms-1b-all_.csv",
    "submission_test2_parakeet_ne.csv",
    "submission_test2_wav2vec-300.csv",
    "submission_test2_whisper_51_.csv",
    "submission_test2_whisper_ft_.csv",
]
WINNING_SHA256 = "9a665cd4b81a0443300b021eec373088a46b2ea9a746dfb826bb1923d434ad31"


def run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def validate_submission(path: Path, expected_ids: list[str]) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if not rows or set(rows[0]) != {"ID", "Target"}:
        raise RuntimeError(f"{path}: expected exactly ID,Target columns")
    ids = [r["ID"] for r in rows]
    if ids != expected_ids:
        raise RuntimeError("Final ID order differs from ensemble input")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Final submission contains duplicate IDs")
    if any(not r["Target"].strip() for r in rows):
        raise RuntimeError("Final submission contains an empty Target")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ensemble", type=Path,
        default=HERE / "artifacts/raw_weighted_char_ensemble.csv",
        help="weighted character-level ensemble CSV",
    )
    ap.add_argument(
        "--out", type=Path,
        default=HERE.parent / "submission/submission_final_postprocessed.csv",
    )
    ap.add_argument(
        "--audit", type=Path,
        default=HERE.parent / "submission/submission_final_postprocessed.audit.csv",
    )
    ap.add_argument(
        "--candidate-members", type=Path, nargs="+",
        help="override the bundled high-recall candidate-generation members",
    )
    ap.add_argument(
        "--diverse-members", type=Path, nargs="+",
        help="override the bundled architecturally diverse verifier members",
    )
    ap.add_argument(
        "--lid", type=Path,
        default=HERE / "artifacts/language_predictions.csv",
    )
    ap.add_argument("--pool", type=Path, default=HERE / "corpus/pool")
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument(
        "--verify-reference", action="store_true",
        help="require byte identity with the bundled competition artifact",
    )
    args = ap.parse_args()

    ensemble = args.ensemble.resolve()
    if not ensemble.exists():
        raise FileNotFoundError(ensemble)
    candidate_dir = HERE / "artifacts/candidate_members"
    diverse_dir = HERE / "artifacts/diverse_members"
    candidate_members = (
        [p.resolve() for p in args.candidate_members]
        if args.candidate_members else [candidate_dir / n for n in CANDIDATE_MEMBER_NAMES]
    )
    diverse_members = (
        [p.resolve() for p in args.diverse_members]
        if args.diverse_members else [diverse_dir / n for n in DIVERSE_MEMBER_NAMES]
    )
    lid = args.lid.resolve()
    pool = args.pool.resolve()
    required = candidate_members + diverse_members + [
        lid, pool / "lin.waxal.txt", pool / "sna.waxal.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing postprocessing artifacts:\n" + "\n".join(missing))

    source_rows = list(csv.DictReader(ensemble.open(encoding="utf-8-sig", newline="")))
    if not source_rows or set(source_rows[0]) != {"ID", "Target"}:
        raise RuntimeError("Ensemble input must contain exactly ID,Target")
    expected_ids = [r["ID"] for r in source_rows]

    own_temp = args.work_dir is None
    temp_context = tempfile.TemporaryDirectory(prefix="waxal_postprocess_") if own_temp else None
    work = Path(temp_context.name) if temp_context else args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    pp = work / "01_confirmed_postprocess.csv"
    rules = work / "02_residual_rules.csv"
    ghost = work / "03_ghost_candidates.csv"
    majority = work / "04_nonword_candidates.csv"
    final_tmp = work / "05_diverse_verified.csv"
    audit_tmp = work / "05_diverse_verified.audit.csv"
    py = sys.executable

    run([py, str(HERE / "apply_best_postprocessing.py"),
         "--submission", str(ensemble), "--out", str(pp),
         "--lid", str(lid), "--pool", str(pool)])
    run([py, str(HERE / "residual_confirmed_rules.py"),
         "--submission", str(pp), "--stage", "baverb", "--out", str(rules),
         "--lid", str(lid), "--pool", str(pool)])
    run([py, str(HERE / "ghost_tokens.py"),
         "--submission", str(rules), "--members",
         *map(str, candidate_members), "--same-clip", "--min-votes", "5",
         "--pool", str(pool), "--lid", str(lid), "--out", str(ghost)])
    run([py, str(HERE / "majority_fix.py"),
         "--submission", str(ghost), "--members", *map(str, candidate_members),
         "--margin", "2", "--min-votes", "4", "--only-nonword",
         "--show", "0", "--pool", str(pool), "--lid", str(lid),
         "--out", str(majority)])
    run([py, str(HERE / "diverse_verify_chain.py"),
         "--base", str(rules), "--ghost", str(ghost), "--majority", str(majority),
         "--members", *map(str, diverse_members), "--out", str(final_tmp),
         "--audit", str(audit_tmp)])

    validate_submission(final_tmp, expected_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(final_tmp, args.out)
    shutil.copyfile(audit_tmp, args.audit)
    digest = sha256(args.out)
    print(f"Final submission: {args.out}")
    print(f"SHA-256: {digest}")

    if args.verify_reference:
        reference = HERE / "reference/winning_submission.csv"
        reference_digest = sha256(reference)
        if digest != WINNING_SHA256 or reference_digest != WINNING_SHA256:
            raise RuntimeError(
                "Reference verification failed: "
                f"output={digest}, bundled={reference_digest}, expected={WINNING_SHA256}"
            )
        print("Verified byte-identical to the bundled winning submission.")

    if temp_context:
        temp_context.cleanup()


if __name__ == "__main__":
    main()

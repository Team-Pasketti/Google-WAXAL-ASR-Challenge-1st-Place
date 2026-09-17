#!/usr/bin/env python3
"""Turn pseudo-label n-best lists into the final training targets.

Applies exactly the pipeline that scored best on the leaderboard, so the
student is trained toward the system we would otherwise submit:

  pick     AfriqueQwen rescoring at lin=0.01, sna=0.3. Lingala is left almost
           untouched on purpose -- rescoring it measured -0.002 at w=0.3 and
           -0.0044 at w=0.05 on the leaderboard, so its acoustic score is
           already the better judge.
  drop     clips whose chosen hypothesis runs past 20 characters/second. Those
           are Whisper's repetition loops (one 1.0s clip produced 492
           characters), and training on them would teach the student to loop.
  split    namoni -> na moni and the ba- class. WAXAL writes `namoni` 3983
           times against `na moni` 1359, but the evaluation references split
           about 73% of the time; baking the correction into the labels stops
           the student inheriting the corpus's backwards convention.
  format   capitalise, close with a stop.

Also drops clips over 29s, where the feature extractor silently truncates the
audio: the label would describe speech the student never hears.

Writes {id: {"label", "language", "speaker_id", "seconds"}} -- speaker_id kept
so training can hold out speakers, which is the only honest way to measure the
generalisation this whole exercise is aimed at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "postprocess" / "remote"))

MAX_CHARS_PER_SEC = 20.0
MAX_SECONDS = 29.0


def immediate_repeat(text: str, max_k: int = 8) -> float:
    """Longest run of a phrase repeated back-to-back, as a share of the clip.

    The chars/sec test only catches repetition that outruns the clip length, so
    it found 4 loops in 26.5k. It misses `Kiti moko ya mbanzi kiti moko ya
    mbanzi kiti moko ya mbanzi` on a 20-second clip, which fits the duration
    budget and is still worthless as a training target.

    Counting duplicate n-grams instead does not work: a decoder loop scores 0.50
    and a genuine Lingala colour list -- "eza na langi ya bozinga ... eza na
    langi ya chokola", a person naming crayons -- scores 0.45. The two are
    indistinguishable that way, and dropping at 0.15 would discard exactly the
    long descriptive clips worth keeping.

    What separates them is adjacency. A loop emits the same k words again
    *immediately*; a speaker reusing a frame puts different fillers between the
    repeats. On the same examples this scores 0.67 and 1.00 for the loops and
    0.00 for both real clips.
    """
    w = text.lower().split()
    worst = 0.0
    for k in range(1, min(max_k, len(w) // 2) + 1):
        i = 0
        while i + 2 * k <= len(w):
            if w[i:i + k] == w[i + k:i + 2 * k]:
                reps = 2
                while (i + (reps + 1) * k <= len(w)
                       and w[i:i + k] == w[i + reps * k:i + (reps + 1) * k]):
                    reps += 1
                worst = max(worst, reps * k / len(w))
                i += reps * k
            else:
                i += 1
    return worst


def cer_nospace(a: str, b: str) -> float:
    """Whitespace-blind, so WAXAL's spacing conventions cannot inflate it.

    Used only to *reject* labels, never to supply them: WAXAL writes `namoni`
    where the evaluation references mostly write `na moni`, so its text must not
    reach the student -- but it is still a sound check on whether Whisper heard
    the same words."""
    from metrics import _edit
    x, y = "".join(a.split()), "".join(b.split())
    return _edit(x, y) / max(len(x), 1)


def finish(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    for i, ch in enumerate(t):
        if ch.isalpha():
            t = t[:i] + ch.upper() + t[i + 1:]
            break
        if ch.isalnum():
            break
    if t[-1] not in ".?!":
        t += "."
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", type=Path, nargs="+", required=True,
                    help="pseudo_label_waxal.py outputs (one per shard)")
    ap.add_argument("--lm-scores", type=Path, nargs="*", default=[],
                    help="qwen_rescore.py score outputs; without them the "
                         "acoustic 1-best is used")
    ap.add_argument("--pool", type=Path,
                    default=Path.home() / "wasal_lm/lm/corpus/pool")
    ap.add_argument("--w-lin", type=float, default=0.01)
    ap.add_argument("--w-sna", type=float, default=0.3)
    ap.add_argument("--word-bonus-sna", type=float, default=0.04)
    ap.add_argument("--no-splits", action="store_true")
    ap.add_argument("--max-ref-cer", type=float, default=0.30,
                    help="drop labels this far from the WAXAL reference "
                         "(whitespace-blind). Catches loops and gross errors; "
                         "the reference is used to reject, never to supply text")
    ap.add_argument("--max-repeat", type=float, default=0.25,
                    help="drop labels where a phrase repeats back-to-back over "
                         "this share of the clip. Loops score 0.67-1.00, real "
                         "repetitive speech scores 0.00")
    ap.add_argument("--min-acoustic", type=float, default=-1e9,
                    help="drop labels below this Whisper sequence score")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from respace_probe import waxal_tables, apply_split
    from form_probe import apply_form

    data: dict[str, dict] = {}
    for p in args.nbest:
        data.update(json.loads(p.read_text(encoding="utf-8")))
    lm: dict[str, list] = {}
    for p in args.lm_scores:
        lm.update(json.loads(p.read_text(encoding="utf-8")))
    print(f"{len(data)} clips, {len(lm)} with LM scores", flush=True)

    ba = {}
    if not args.no_splits:
        tl, _ = waxal_tables(args.pool, "lin", 3, 0.95)
        ba = {j: s for j, s in tl.items()
              if s.split()[0].lower() in {"ba", "bazo", "baza"}}

    out: dict[str, dict] = {}
    dropped_loop = dropped_long = dropped_empty = 0
    dropped_cer = dropped_rep = dropped_ac = 0
    for uid, rec in data.items():
        secs = rec.get("seconds", 0.0)
        if secs > MAX_SECONDS:
            dropped_long += 1
            continue
        lang = rec["language"]
        cands = rec["nbest"]
        w = args.w_lin if lang == "lin" else args.w_sna
        wb = 0.0 if lang == "lin" else args.word_bonus_sna
        best, best_s = cands[0], None
        if uid in lm and len(cands) > 1:
            for i, hyp in enumerate(cands):
                total, ntok = lm[uid][i]
                s = (rec["scores"][i] + w * (total / max(ntok, 1))
                     + wb * len(hyp.split()))
                if best_s is None or s > best_s:
                    best, best_s = hyp, s

        if not best.strip():
            dropped_empty += 1
            continue
        if len(best) / max(secs, 1e-9) > MAX_CHARS_PER_SEC:
            dropped_loop += 1
            continue
        if immediate_repeat(best) > args.max_repeat:
            dropped_rep += 1
            continue
        ref = rec.get("waxal_reference") or ""
        if ref and cer_nospace(ref, best) > args.max_ref_cer:
            dropped_cer += 1
            continue
        if rec["scores"][0] < args.min_acoustic:
            dropped_ac += 1
            continue

        if not args.no_splits and lang == "lin":
            best, _ = apply_form(best, "na", "moni", "split")
            best, _ = apply_split(best, ba)
        out[uid] = {"label": finish(best), "language": lang,
                    "speaker_id": rec.get("speaker_id"),
                    "seconds": secs}

    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    spk = {v["speaker_id"] for v in out.values() if v["speaker_id"]}
    print(f"wrote {args.out}: {len(out)} labels")
    print(f"  dropped: {dropped_loop} chars/sec runaway, {dropped_rep} repetition, "
          f"{dropped_cer} far from WAXAL (>{args.max_ref_cer}), "
          f"{dropped_ac} low acoustic, {dropped_long} over {MAX_SECONDS}s, "
          f"{dropped_empty} empty")
    print(f"  {len(spk)} distinct speakers")
    for lang in ("lin", "sna"):
        n = sum(1 for v in out.values() if v["language"] == lang)
        print(f"  {lang}: {n} clips")
    for uid in list(out)[:2]:
        print(f"   {uid}: {out[uid]['label'][:95]}")


if __name__ == "__main__":
    main()

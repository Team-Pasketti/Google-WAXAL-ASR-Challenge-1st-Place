#!/usr/bin/env python3
"""Find tokens the ensemble produced that no model ever did.

An ASR model can only emit subword sequences it learned, so a token appearing in
some model's output is very likely a real word -- possibly one WAXAL happens to
lack, like `kurwira` ("to fight for") or the party name ZANU. A character-level
median has no such constraint: it walks a Levenshtein path between hypotheses
and can land on a string that is one edit from two real words and is neither.

So neither test alone is safe. WAXAL absence over-fires on rare-but-real words;
member absence over-fires on tokens our postprocessing created (joining
`ba lakisi` into `balakisi` yields a token no raw member emitted). Their
conjunction is tight: absent from WAXAL *and* emitted by no model is a string
with no evidence of existing anywhere.

Those are the median's ghosts. Each costs a whole token in WER while costing
only an edit or two in CER, which is exactly the asymmetry to attack when a
rival matches your WER on much worse CER.

Repair is to the nearest attested string -- searched over WAXAL plus every
model's vocabulary, so a rare real word is a valid target.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent


def toks(text: str):
    for w in text.split():
        c = w.strip(".,!?;:\"'()").lower()
        if c and c.isalpha():
            yield c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--members", type=Path, nargs="+", required=True)
    ap.add_argument("--pool", type=Path, default=HERE / "corpus/pool")
    ap.add_argument("--lid", type=Path,
                    default=HERE.parent / "newaudios_announced3_predictions.csv")
    ap.add_argument("--lid-column", default="announced3_language")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-dist", type=int, default=1)
    # Corpus frequency picks the globally commoner word, which is not the same
    # question: `nyema` -> `nyama` and `furi` -> `uri` win on frequency without
    # any evidence about this clip. Restricting the repair to a token some model
    # actually emitted *for this clip* replaces a prior with a witness -- the
    # models heard the audio, the corpus did not.
    ap.add_argument("--same-clip", action="store_true",
                    help="repair only to a token a member produced in that clip")
    # One model's word is one model's guess. Requiring several to agree turns
    # the repair from a witness into a consensus, at the cost of leaving the
    # thinly-supported ghosts alone -- which is the right trade when a wrong
    # repair costs exactly what the ghost already costs.
    ap.add_argument("--min-votes", type=int, default=1)
    # Dropping the WAXAL gate admits tokens that are real words but that no
    # model produced for this clip. The models heard the audio and the corpus
    # did not, so their silence is evidence against the word here -- but a
    # median can also legitimately reconstruct a word no single member got, so
    # this set is only safe under heavy agreement. Substitutions between two
    # frequent short words (`bazo`/`bato`) are the risky core of it.
    ap.add_argument("--no-waxal-gate", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows_l = list(csv.DictReader(open(args.lid, encoding="utf-8-sig")))
    idc = "ID" if "ID" in rows_l[0] else "id"
    lid = {r[idc]: r[args.lid_column] for r in rows_l}

    wax = {l: Counter(re.findall(
        r"[a-z]+", (args.pool / f"{l}.waxal.txt").read_text(encoding="utf-8").lower()))
        for l in ("lin", "sna")}

    # every token any model emitted, per language
    member_vocab = {l: Counter() for l in ("lin", "sna")}
    for p in args.members:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            l = lid.get(r["ID"])
            if l in member_vocab:
                member_vocab[l].update(toks(r["Target"]))

    # search space for repairs: attested anywhere
    attested = {l: Counter(wax[l]) for l in wax}
    for l in attested:
        for w, n in member_vocab[l].items():
            attested[l][w] += n
    bylen = {l: {} for l in attested}
    for l, c in attested.items():
        for w in c:
            bylen[l].setdefault(len(w), []).append(w)

    # What each member said, clip by clip. Counted as distinct *systems*, not
    # occurrences: a word used twice in one transcript is one system's opinion
    # stated twice, and letting it count double would make a 10-system vote
    # reach 19.
    per_clip: dict[str, Counter] = {}
    for p in args.members:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            per_clip.setdefault(r["ID"], Counter()).update(set(toks(r["Target"])))

    rows = list(csv.DictReader(open(args.submission, encoding="utf-8-sig")))
    n_tok = n_ghost = 0
    ghosts, repairs = Counter(), {}
    for r in rows:
        l = lid.get(r["ID"])
        if l not in wax:
            continue
        for w in toks(r["Target"]):
            if len(w) < args.min_len:
                continue
            n_tok += 1
            if args.no_waxal_gate:
                # only the models' silence on this clip counts
                if per_clip.get(r["ID"], Counter()).get(w, 0):
                    continue
            elif wax[l].get(w, 0) or member_vocab[l].get(w, 0):
                continue
            n_ghost += 1
            key = (l, w, r["ID"]) if args.same_clip else (l, w)
            ghosts[key] += 1
            if key in repairs:
                continue
            best = None
            if args.same_clip:
                # witnesses: tokens a model produced for THIS clip
                for cand, votes in per_clip.get(r["ID"], Counter()).items():
                    if cand == w or abs(len(cand) - len(w)) > args.max_dist:
                        continue
                    if Levenshtein.distance(w, cand,
                                            score_cutoff=args.max_dist) <= args.max_dist:
                        if best is None or votes > best[1]:
                            best = (cand, votes)
            else:
                for L in range(len(w) - args.max_dist, len(w) + args.max_dist + 1):
                    for cand in bylen[l].get(L, ()):
                        if cand == w:
                            continue
                        if Levenshtein.distance(w, cand,
                                                score_cutoff=args.max_dist) <= args.max_dist:
                            sc = attested[l][cand]
                            if best is None or sc > best[1]:
                                best = (cand, sc)
            if best and best[1] < args.min_votes:
                best = None
            repairs[key] = best

    print(f"{args.submission.name}")
    print(f"  {len(args.members)} member systems, "
          f"{sum(len(v) for v in member_vocab.values())} distinct model tokens")
    print(f"  {n_tok} tokens (len>={args.min_len}) | "
          f"{n_ghost} produced by NO model and absent from WAXAL "
          f"({100*n_ghost/max(n_tok,1):.2f}%)")
    fixable = {k: v for k, v in repairs.items() if v}
    n_fix = sum(ghosts[k] for k in fixable)
    print(f"  {len(fixable)} of {len(ghosts)} distinct ghosts have an attested "
          f"neighbour within {args.max_dist} edit -> {n_fix} tokens repairable\n")
    lab = "votes" if args.same_clip else "seen"
    print(f"  {'lang':5}{'ghost':24}{'->':3}{'repair':24}{lab:>6}{'n':>4}")
    for key, n in ghosts.most_common():
        b = repairs[key]
        if b:
            print(f"  {key[0]:5}{key[1]:24}-> {b[0]:24}{b[1]:>6}{n:>4}")
    print("\n  unrepairable ghosts (no witness):")
    for key, n in ghosts.most_common():
        if not repairs[key]:
            print(f"    {key[0]} {key[1]} (x{n})")

    if args.out:
        out_rows = []
        changed = 0
        for r in rows:
            l = lid.get(r["ID"])
            t = r["Target"]
            if l in wax:
                new = []
                for word in t.split():
                    core = word.strip(".,!?;:\"'()")
                    trail = word[len(word.rstrip(".,!?;:\"'()")):]
                    b = repairs.get((l, core.lower(), r["ID"]) if args.same_clip
                                    else (l, core.lower()))
                    if b and core.isalpha() and len(core) >= args.min_len:
                        rep = b[0]
                        new.append((rep.capitalize() if core[:1].isupper()
                                    else rep) + trail)
                    else:
                        new.append(word)
                nt = " ".join(new)
                if nt != t:
                    changed += 1
                t = nt
            out_rows.append((r["ID"], t))
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["ID", "Target"])
            w_.writerows(out_rows)
        print(f"\nwrote {args.out.name}: {changed} rows changed")


if __name__ == "__main__":
    main()

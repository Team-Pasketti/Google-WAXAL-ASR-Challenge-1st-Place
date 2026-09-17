#!/usr/bin/env python3
"""Switch ensemble tokens that a clear majority of members spells differently.

The ghost-repair submission gained +0.000495, but the audit showed 26 of its 29
changes were not ghosts at all -- real members had produced those tokens. What
the change actually did was replace a spelling few members supported with one
more of them supported (`nyema` 2 -> `nyama` 4, `bapegi` 1 -> `bapeki` 3). The
two that moved *against* the member count, `mapusha` 3 -> `mapushi` 2 and
`osuki` 3 -> nothing, are the ones most likely to have cost.

So the real rule is a vote margin, and it applies to every token rather than the
handful that happened to be out of vocabulary.

Two conditions keep this from re-running the ensemble badly:

  margin      a variant replaces the current token only if strictly more members
              spell it that way, by at least --margin. Ties leave the median
              alone, since the median saw the acoustics and the tie-break is
              its job.
  distance    only variants within one edit. Anything further is a different
              word, and choosing between different words is what the ensemble
              already did.

Members must carry the same postprocessing as the submission. Without that,
every split our spacing rules make looks like a token no member produced --
`nazomona` -> `nazo mona` makes `nazo` and `mona` appear unsupported, and the
rule would offer to undo the confirmed +0.00227 split.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

TRAIL = ".,!?;:\"'()"


def toks(text: str):
    for w in text.split():
        c = w.strip(TRAIL).lower()
        if c and c.isalpha():
            yield c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--members", type=Path, nargs="+", required=True)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--margin", type=int, default=2,
                    help="required excess of member votes over the current token")
    ap.add_argument("--min-votes", type=int, default=3,
                    help="absolute votes the replacement needs")
    # Overriding the median wholesale measured -0.000563 on 182 tokens, of
    # which 158 were cases where the median had chosen a perfectly real word
    # that a minority of members supported. It is usually right to do that: the
    # median sees character alignment across every member at once, which a vote
    # count does not. Where it is demonstrably broken is on strings that are not
    # words at all -- that slice is what the +0.000495 ghost repair actually hit.
    ap.add_argument("--only-nonword", action="store_true",
                    help="only replace tokens absent from the WAXAL vocabulary")
    ap.add_argument("--pool", type=Path,
                    default=Path(__file__).resolve().parent / "corpus/pool")
    ap.add_argument("--lid", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "newaudios_announced3_predictions.csv")
    ap.add_argument("--lid-column", default="announced3_language")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    wax, lid = {}, {}
    if args.only_nonword:
        import re
        from collections import Counter as _C
        wax = {l: _C(re.findall(r"[a-z]+", (args.pool / f"{l}.waxal.txt")
                                .read_text(encoding="utf-8").lower()))
               for l in ("lin", "sna")}
        rl = list(csv.DictReader(open(args.lid, encoding="utf-8-sig")))
        idc = "ID" if "ID" in rl[0] else "id"
        lid = {r[idc]: r[args.lid_column] for r in rl}

    per: dict[str, Counter] = {}
    for p in args.members:
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            per.setdefault(r["ID"], Counter()).update(set(toks(r["Target"])))

    rows = list(csv.DictReader(open(args.submission, encoding="utf-8-sig")))
    changes, out_rows = [], []
    for r in rows:
        votes = per.get(r["ID"], Counter())
        new_words = []
        for w in r["Target"].split():
            core = w.strip(TRAIL)
            trail = w[len(w.rstrip(TRAIL)):]
            lc = core.lower()
            if len(lc) < args.min_len or not lc.isalpha():
                new_words.append(w)
                continue
            if args.only_nonword:
                l = lid.get(r["ID"])
                if l not in wax or wax[l].get(lc, 0):
                    new_words.append(w)
                    continue
            mine = votes.get(lc, 0)
            best = None
            for cand, v in votes.items():
                if cand == lc or abs(len(cand) - len(lc)) > 1:
                    continue
                if v < args.min_votes or v - mine < args.margin:
                    continue
                if Levenshtein.distance(lc, cand, score_cutoff=1) <= 1:
                    if best is None or v > best[1]:
                        best = (cand, v)
            if best:
                rep = best[0]
                changes.append((r["ID"], lc, mine, rep, best[1]))
                new_words.append((rep.capitalize() if core[:1].isupper()
                                  else rep) + trail)
            else:
                new_words.append(w)
        out_rows.append((r["ID"], " ".join(new_words)))

    print(f"{args.submission.name} vs {len(args.members)} members | "
          f"margin>={args.margin}, votes>={args.min_votes}")
    print(f"  {len(changes)} tokens in "
          f"{len({c[0] for c in changes})} clips\n")
    print(f"  {'clip':13}{'current':22}{'v':>3}{'  ->  '}{'majority':22}{'v':>3}")
    for uid, old, vo, new, vn in changes[:args.show]:
        print(f"  {uid:13}{old:22}{vo:>3}  ->  {new:22}{vn:>3}")
    if len(changes) > args.show:
        print(f"  ... {len(changes) - args.show} more")

    if args.out:
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID", "Target"])
            w.writerows(out_rows)
        src = {r["ID"]: r["Target"] for r in rows}
        d = dict(out_rows)
        print(f"\nwrote {args.out.name}: "
              f"{sum(1 for k, v in d.items() if v != src[k])} rows changed, "
              f"{sum(1 for v in d.values() if not v.strip())} empty")


if __name__ == "__main__":
    main()

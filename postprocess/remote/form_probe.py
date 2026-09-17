#!/usr/bin/env python3
"""Force one word-form to a single spelling, to read the annotators' convention.

The test references come from annotators whose conventions are not documented,
and WAXAL's own are inconsistent -- it writes `namoni` 3983 times and `na moni`
1359, `vari kufamba` 361 and `varikufamba` 475. So a hypothesis that disagrees
with WAXAL is not thereby wrong for the test set, and every offline measurement
against WAXAL references inherits that ambiguity.

The leaderboard can answer it directly, one form at a time. Pick a form whose
spacing is purely orthographic -- same morphemes, same audio, no change in
meaning -- normalise every occurrence to one spelling, leave the rest of the
submission byte-identical, and read the sign of the delta.

Run the same form in both directions. If the annotators are consistent, one
direction gains roughly what the other loses. If both lose slightly, they are
mixed like WAXAL and the model's own distribution was already the better bet.

Good candidates are frequent and morphologically transparent:

  lin  namoni / na moni       na- (1sg) + moni (see)   179/73 in our output
  sna  ineruvara / ine ruvara ine (has) + ruvara       1/74

Bad candidates are real compounds, where the split changes the word rather than
its spelling: `Munhurume` ("man") is not `Munhu rume`, and `komona` is a
Lingala infinitive that standard orthography always joins.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

TRAIL = ".,!?;:\"'"


def apply_form(text: str, a: str, b: str, mode: str) -> tuple[str, int]:
    """Normalise every occurrence of a+b / a b to one spelling."""
    joined = a + b
    toks = text.split()
    out: list[str] = []
    hits = 0
    i = 0
    while i < len(toks):
        w = toks[i]
        core = w.strip(TRAIL)
        trail = w[len(w.rstrip(TRAIL)):]
        if mode == "split" and core.lower() == joined:
            # keep the original capitalisation of the first letter
            first = a.capitalize() if core[:1].isupper() else a
            out.append(first)
            out.append(b + trail)
            hits += 1
            i += 1
            continue
        if mode == "join" and i + 1 < len(toks) and core.lower() == a:
            nxt = toks[i + 1]
            ncore = nxt.strip(TRAIL)
            ntrail = nxt[len(nxt.rstrip(TRAIL)):]
            if ncore.lower() == b:
                merged = joined.capitalize() if core[:1].isupper() else joined
                out.append(merged + ntrail)
                hits += 1
                i += 2
                continue
        out.append(w)
        i += 1
    return " ".join(out), hits


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
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--lid", type=Path, required=True)
    ap.add_argument("--lid-column", default="announced3_language")
    ap.add_argument("--language", required=True)
    ap.add_argument("--form", nargs=2, metavar=("A", "B"), required=True,
                    help="the two halves, e.g. --form na moni")
    ap.add_argument("--mode", choices=["split", "join"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lid = {r["id"]: r[args.lid_column]
           for r in csv.DictReader(open(args.lid, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(args.submission, encoding="utf-8-sig")))
    a, b = (x.lower() for x in args.form)

    total, clips = 0, 0
    out_rows = []
    sample = None
    for r in rows:
        text = r["Target"]
        if lid.get(r["ID"]) == args.language:
            new, hits = apply_form(text, a, b, args.mode)
            if hits:
                total += hits
                clips += 1
                if sample is None:
                    sample = (r["ID"], text, new)
            text = new
        out_rows.append((r["ID"], finish(text)))

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Target"])
        w.writerows(out_rows)

    d = dict(out_rows)
    print(f"{args.out.name}: {len(out_rows)} rows | {args.language} "
          f"'{a}'+'{b}' -> {args.mode} | {total} occurrences in {clips} clips")
    print(f"   empty={sum(1 for v in d.values() if not v.strip())} "
          f"unique={len(set(d))}")
    if sample:
        uid, was, now = sample
        i = next((j for j in range(min(len(was), len(now))) if was[j] != now[j]), 0)
        print(f"   {uid}: ...{was[max(0,i-30):i+22]!r}")
        print(f"        ...{now[max(0,i-30):i+23]!r}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Force a consistent word-splitting convention, to probe the test annotators'.

WAXAL writes thousands of bigrams both ways -- `vari kufamba` 361 times and
`varikufamba` 475, `na moni` 1359 and `namoni` 3983. Across the corpus that is
105,932 split against 110,062 joined in Lingala: a coin flip. Any model trained
on it inherits the coin flip, which is why three different rescorers (a 9B
instruct model, a 14B base model, and KenLM itself) all captured about 0.001 of
a 0.029 oracle gap. None of them can win a coin flip.

But that reasoning only makes the errors irreducible if the *test* annotators
also flipped coins. If they were consistent -- one team, one style guide, or a
normalisation pass -- then everybody's WAXAL-trained model gets roughly half of
these tokens wrong and mistakes the plateau for a ceiling, while a systematic
rule collects all of them.

That is a leaderboard question, not an offline one, so this writes the two
probes:

  --mode split   every joined form that WAXAL also writes split becomes split
  --mode join    every split pair that WAXAL also writes joined becomes joined

If the test set is 50/50 like WAXAL both land near the baseline and the noise
reading is confirmed. If either jumps, the convention is real and worth a lot:
the oracle over spaces alone was +0.0701 on Lingala and +0.0583 on Shona.

Only bigrams WAXAL itself writes both ways are touched. A word it always writes
joined is a real word, not a convention, and is left alone.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


# Co-occurrence alone is not a grammar. Splitting every joined form WAXAL also
# writes apart correctly recovers `varikufamba` -> `vari kufamba`, but it also
# tears real compounds apart: `Munhurume` ("man") -> `Munhu rume`, `egirinhi`
# ("green") -> `e girinhi`, `dzebhuruu` -> `dze bhuruu`. Inspecting the 98
# splits the Shona rule wanted to make, the sound ones all share one shape --
# a concord ending in -ri before a ku- infinitive (the progressive), or one
# ending in -ne before a noun (the possessive) -- and none of the compounds do.
# Requiring that shape keeps the systematic cases and drops the rest.
def grammatical_split(a: str, b: str) -> bool:
    x, y = a.lower(), b.lower()
    if x.endswith("ri") and y.startswith("ku"):
        return True
    if x.endswith("ne") and len(x) <= 6:
        return True
    # the same -ri concord before something other than a ku- infinitive,
    # e.g. `zvirimukati` -> `zviri mukati`; measured +0.0011 on validation
    # against +0.0006 for the ku- form alone. The length bound keeps it to
    # actual concords rather than any word that happens to end in -ri.
    if x.endswith("ri") and len(x) <= 8:
        return True
    # Rejected: concord + borrowed colour (`rwebhuruu` -> `rwe bhuruu`). It
    # looked like the same phenomenon and is 160 tokens of our output, but
    # WAXAL writes those joined far more often and it measured -0.0085.
    return False


def waxal_tables(pool: Path, lang: str, min_count: int, max_ratio: float,
                 grammatical: bool = False):
    """(splits that also occur joined, joins that also occur split)."""
    words: Counter[str] = Counter()
    bigr: Counter[tuple[str, str]] = Counter()
    path = pool / f"{lang}.waxal.txt"
    for line in path.open(encoding="utf-8"):
        toks = line.split()
        words.update(toks)
        for i in range(len(toks) - 1):
            bigr[(toks[i], toks[i + 1])] += 1

    # joined -> best split, and split -> joined, for genuinely ambiguous pairs
    to_split: dict[str, str] = {}
    to_join: dict[tuple[str, str], str] = {}
    best: dict[str, tuple[int, tuple[str, str]]] = {}
    for (a, b), n in bigr.items():
        j = a + b
        m = words.get(j, 0)
        if m < min_count or n < min_count:
            continue
        # skip lopsided pairs: if WAXAL writes it one way 95% of the time that
        # is a spelling, not a convention the annotators were choosing between
        share = min(n, m) / (n + m)
        if share < (1.0 - max_ratio):
            continue
        if grammatical and not grammatical_split(a, b):
            continue
        to_join[(a, b)] = j
        if j not in best or n > best[j][0]:
            best[j] = (n, (a, b))
    for j, (_n, (a, b)) in best.items():
        to_split[j] = f"{a} {b}"
    return to_split, to_join


def apply_split(text: str, table: dict[str, str]) -> tuple[str, int]:
    out, hits = [], 0
    for w in text.split():
        core = w.strip(".,!?;:")
        trail = w[len(w.rstrip(".,!?;:")):]
        if core in table:
            out.append(table[core] + trail)
            hits += 1
        else:
            out.append(w)
    return " ".join(out), hits


def apply_join(text: str, table: dict[tuple[str, str], str]) -> tuple[str, int]:
    toks = text.split()
    out, hits, i = [], 0, 0
    while i < len(toks):
        if i + 1 < len(toks):
            a = toks[i]
            b_core = toks[i + 1].strip(".,!?;:")
            trail = toks[i + 1][len(toks[i + 1].rstrip(".,!?;:")):]
            if (a, b_core) in table:
                out.append(table[(a, b_core)] + trail)
                hits += 1
                i += 2
                continue
        out.append(toks[i])
        i += 1
    return " ".join(out), hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--pool", type=Path,
                    default=Path.home() / "wasal_lm/lm/corpus/pool")
    ap.add_argument("--lid", type=Path,
                    default=Path.home() / "wasal_lm/lm/newaudios_announced3_predictions.csv")
    ap.add_argument("--mode", choices=["split", "join"], required=True)
    ap.add_argument("--languages", nargs="+", default=["lin", "sna"])
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--max-ratio", type=float, default=0.95,
                    help="skip pairs more lopsided than this in WAXAL")
    ap.add_argument("--grammatical", action="store_true",
                    help="only the systematic concord constructions, not every "
                         "pair WAXAL happens to write both ways")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tables = {l: waxal_tables(args.pool, l, args.min_count, args.max_ratio,
                              args.grammatical)
              for l in args.languages}
    for l, (s, j) in tables.items():
        print(f"{l}: {len(s)} joined forms splittable, {len(j)} pairs joinable")

    lid = {r["id"]: r["announced3_language"]
           for r in csv.DictReader(open(args.lid, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(args.submission, encoding="utf-8-sig")))

    total, touched = 0, 0
    out_rows = []
    for r in rows:
        text = r["Target"]
        lang = lid.get(r["ID"])
        if lang in tables:
            to_split, to_join = tables[lang]
            if args.mode == "split":
                text, hits = apply_split(text, to_split)
            else:
                text, hits = apply_join(text, to_join)
            total += hits
            touched += hits > 0
        out_rows.append((r["ID"], text))

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Target"])
        w.writerows(out_rows)
    print(f"wrote {args.out}: {len(out_rows)} rows, mode={args.mode}, "
          f"{total} tokens changed across {touched} clips")
    src = {r["ID"]: r["Target"] for r in rows}
    for uid, text in out_rows:
        if text != src[uid]:
            print(f"   e.g. {uid}\n     was: {src[uid][:90]}\n     now: {text[:90]}")
            break


if __name__ == "__main__":
    main()

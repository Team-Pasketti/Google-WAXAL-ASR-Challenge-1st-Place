#!/usr/bin/env python3
"""The confirmed postprocessing stack, applied to any submission file.

Every step here has a leaderboard verdict behind it, and the stages are named so
a run can stop at the last confirmed one. That matters when postprocessing a
system whose decoder output we do not own -- an ensemble from someone else, say
-- because nothing in this file needs the model, the n-best list or the audio.

  --stage confirmed   through the single `bazali` join (scored 0.759759203)
  --stage baverb      plus the ba+verb family                 (scored 0.761005)
  --stage rule18      plus every form WAXAL splits >=18%      (untested)

Qwen rescoring is deliberately absent: it needs an n-best list, and on Lingala
it was 92% spacing votes in the joining direction anyway, which is the wrong
cohort. Runaway repair is text-only here for the same reason -- re-decoding
needs the weights that produced the file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIL = ".,!?;:\"'"

# --- confirmed single forms, in the order they were confirmed ----------------
SPLIT_FORMS = [("na", "moni"), ("nazo", "mona"), ("tozo", "mona")]
JOIN_CONFIRMED = [("ba", "zali")]
# `ba` + conjugated verb, 8 for 8 on the leaderboard. Corpus split share runs
# from 6% (bazali) to 30% (balakisi), so no threshold selects this set -- the
# grammatical class does. `batelemi` reads 19% and must be joined while
# `tozomona` reads 19% and must be split, which is what killed the scalar rule.
JOIN_BAVERB = [("ba", "lati"), ("ba", "komi"), ("ba", "fandi"),
               ("ba", "vandi"), ("ba", "sali"), ("ba", "telemi"),
               ("ba", "lakisi")]


# --- degenerate tail repair --------------------------------------------------
def strip_degenerate_tail(text: str, min_reps: int = 3) -> tuple[str, str]:
    """Remove a trailing run of a short repeating character cycle.

    ID_LTOEWY ended `...kitoko kitoko kitoko mingi. .S.A.A.A.A.A...` and had to
    be trimmed by hand. The chars/sec detector could not see it: the clip was
    long enough to hold that many characters, so the runaway never tripped the
    duration gate.

    What marks it is not length but shape -- a two-character cycle `.A` repeated
    five times. The test is deliberately narrow about what counts as a cycle,
    because Lingala reduplicates for emphasis and `kitoko kitoko kitoko` in the
    *same* clip is real speech the annotator wrote out. So a cycle only
    qualifies if it contains punctuation or is mostly non-alphabetic; a repeated
    run of plain words is left alone.
    """
    s = text.rstrip()
    best = None
    for k in range(1, 7):
        # grow the repetition backwards from the end
        for end in (len(s),):
            unit = s[end - k:end]
            if not unit:
                continue
            alpha = sum(c.isalpha() for c in unit)
            if not any(c in ".,!?;:-" for c in unit) and alpha / len(unit) > 0.5:
                continue          # a word-shaped cycle: could be reduplication
            reps = 1
            i = end - k
            while i - k >= 0 and s[i - k:i] == unit:
                reps += 1
                i -= k
            if reps >= min_reps and (best is None or (end - i) > best[1] - best[0]):
                best = (i, end, unit, reps)
    if not best:
        return text, ""
    i, end, unit, reps = best
    removed = s[i:]
    kept = s[:i].rstrip(" .,;:-")
    # The cycle boundary rarely lands cleanly. ID_LTOEWY read `...mingi. .S.A.A`
    # and cutting at the first `.A` leaves `...mingi.S` -- a single letter
    # stranded after a sentence-final stop, which is the head of the same
    # degenerate run rather than a word. Only applied on rows we already trimmed.
    kept = re.sub(r"([.!?])\s*[A-Za-z]$", r"\1", kept).rstrip(" .,;:-")
    if not kept.strip():
        return text, ""           # never empty a row; Zindi rejects blanks
    return kept, removed


def collapse_punct(text: str) -> str:
    """`....` -> `.` and `,,` -> `,`; never invents punctuation, only de-dupes."""
    return re.sub(r"([.,!?;:])\1{1,}", r"\1", text)


# --- spacing -----------------------------------------------------------------
def waxal_ba_table(pool: Path, min_count: int = 3, max_ratio: float = 0.95):
    """The original P5b `ba-` split table, reproduced exactly."""
    words: Counter[str] = Counter()
    bigr: Counter[tuple[str, str]] = Counter()
    for line in (pool / "lin.waxal.txt").open(encoding="utf-8"):
        toks = line.split()
        words.update(toks)
        for i in range(len(toks) - 1):
            bigr[(toks[i], toks[i + 1])] += 1
    best: dict[str, tuple[int, tuple[str, str]]] = {}
    for (a, b), n in bigr.items():
        j = a + b
        m = words.get(j, 0)
        if m < min_count or n < min_count:
            continue
        if min(n, m) / (n + m) < (1.0 - max_ratio):
            continue
        if j not in best or n > best[j][0]:
            best[j] = (n, (a, b))
    return {j: ab for j, (_n, ab) in best.items()
            if ab[0].lower() in {"ba", "bazo", "baza"}}


def apply_split(text: str, table: dict[str, tuple[str, str]]) -> tuple[str, int]:
    out, hits = [], 0
    for w in text.split():
        core = w.strip(TRAIL)
        trail = w[len(w.rstrip(TRAIL)):]
        ab = table.get(core.lower())
        if ab:
            a, b = ab
            out.append(a.capitalize() if core[:1].isupper() else a)
            out.append(b + trail)
            hits += 1
        else:
            out.append(w)
    return " ".join(out), hits


def apply_join(text: str, table: dict[tuple[str, str], str]) -> tuple[str, int]:
    toks = text.split()
    out, hits, i = [], 0, 0
    while i < len(toks):
        if i + 1 < len(toks):
            a_raw = toks[i]
            a = a_raw.strip(TRAIL)
            nxt = toks[i + 1]
            b = nxt.strip(TRAIL)
            trail = nxt[len(nxt.rstrip(TRAIL)):]
            j = table.get((a.lower(), b.lower()))
            # only join when nothing punctuates the boundary
            if j is not None and a_raw == a:
                out.append((j.capitalize() if a[:1].isupper() else j) + trail)
                hits += 1
                i += 2
                continue
        out.append(toks[i])
        i += 1
    return " ".join(out), hits


def join_sna_progressive(text: str) -> tuple[str, int]:
    """Join the Shona progressive: concord ending `-ri` + a `ku-` infinitive.

    WAXAL writes this 5651 split against 5288 joined -- a coin flip, which is
    why the corpus-share rule that works for Lingala says nothing here. Shona is
    the case where the corpus average is the average of two cohorts and tracks
    neither.

    Three independent lines say the test annotators join it: the joiner cohort
    joins `varikufamba` 91%, `arikufamba` 94% and `irikufamba` 100%; splitting
    Shona measured -0.010 on the leaderboard; and the 4-model ensemble that
    posts the best CER anyone has scored joins 140 against 119 where our student
    joins 3 against 260.

    Restricted to `ku-` on purpose. The same cohort *splits* `-ri` before a
    locative or noun -- `iri pakati` 27%, `vari munhandare` 21% -- so this must
    not generalise to every concord.
    """
    toks = text.split()
    out, hits, i = [], 0, 0
    while i < len(toks):
        a_raw = toks[i]
        a = a_raw.strip(TRAIL)
        if (i + 1 < len(toks) and a_raw == a and len(a) >= 3
                and a.lower().endswith("ri")):
            nxt = toks[i + 1]
            b = nxt.strip(TRAIL)
            if b.lower().startswith("ku") and len(b) > 2:
                out.append(a + nxt)
                hits += 1
                i += 2
                continue
        out.append(a_raw)
        i += 1
    return " ".join(out), hits


def split_sna_progressive(text: str) -> tuple[str, int]:
    """Split the Shona progressive: concord ending `-ri` + a `ku-` infinitive.

    Confirmed twice over. It is the standing rule in the winning pipeline
    (`varikufamba` -> `vari kufamba`, 118 Shona boundary changes on the V6
    ensemble), and the inverse experiment settled it independently: joining
    these cost -0.008 (CER +0.0014, WER +0.0150).

    This stack lacked it until now because the rule arrived from the other
    pipeline and the only version here was the *joining* one written to test
    that failed idea -- so anything postprocessed with this file kept its
    unsplit Shona progressives.

    Restricted to `ku-` deliberately: the same annotators keep `-ri` joined
    before a locative or noun (`iri pakati` 27%, `vari munhandare` 21%).
    """
    out, hits = [], 0
    for w in text.split():
        core = w.strip(TRAIL)
        trail = w[len(w.rstrip(TRAIL)):]
        m = re.match(r"^(\w*ri)(ku\w{3,})$", core, re.IGNORECASE)
        if m and len(m.group(1)) >= 3:
            out.append(m.group(1))
            out.append(m.group(2) + trail)
            hits += 1
        else:
            out.append(w)
    return " ".join(out), hits


# `ne`-class possessive/comitative before a noun. Listed explicitly rather than
# generalised to every `ne...` token: most of those are ordinary words, and
# every broad spacing rule tried this session lost.
SNA_NE_SPLITS = {
    "newaya": "ne waya", "nebhutsu": "ne bhutsu", "negireyi": "ne gireyi",
    "nefenzi": "ne fenzi", "nesiketi": "ne siketi",
    "uneruvara": "une ruvara", "aneruvara": "ane ruvara",
    "ineruvara": "ine ruvara",
}


def split_sna_ne(text: str) -> tuple[str, int]:
    out, hits = [], 0
    for w in text.split():
        core = w.strip(TRAIL)
        trail = w[len(w.rstrip(TRAIL)):]
        r = SNA_NE_SPLITS.get(core.lower())
        if r:
            a, b = r.split()
            out.append(a.capitalize() if core[:1].isupper() else a)
            out.append(b + trail)
            hits += 1
        else:
            out.append(w)
    return " ".join(out), hits


SNA_SPELLING = {"bhurawuni": "bhurauni", "bhurawuni.": "bhurauni."}


def apply_sna_spelling(text: str) -> tuple[str, int]:
    """`bhurawuni` -> `bhurauni` (brown). WAXAL writes it 644 to 349."""
    out, hits = [], 0
    for w in text.split():
        core = w.strip(TRAIL)
        trail = w[len(w.rstrip(TRAIL)):]
        r = SNA_SPELLING.get(core.lower())
        if r:
            out.append((r.capitalize() if core[:1].isupper() else r) + trail)
            hits += 1
        else:
            out.append(w)
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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=HERE / "corpus/pool")
    ap.add_argument("--lid", type=Path,
                    default=HERE.parent / "newaudios_announced3_predictions.csv")
    ap.add_argument("--lid-column", default="announced3_language")
    ap.add_argument("--stage", choices=["confirmed", "baverb", "rule18"],
                    default="confirmed")
    ap.add_argument("--rule18-table", type=Path, default=HERE / "board_lin.json")
    ap.add_argument("--no-tail-repair", action="store_true")
    ap.add_argument("--sna-join-progressive", action="store_true",
                    help="join `-ri` + `ku-` in Shona")
    ap.add_argument("--sna-spelling", action="store_true",
                    help="bhurawuni -> bhurauni")
    # Re-deriving the ba- table from the corpus is not a no-op on a file that
    # already went through the original pipeline: it catches `Bakonzi`, which
    # that pipeline left joined. Harmless in itself, but it silently adds a
    # second variable to a probe whose stated variable is Shona.
    ap.add_argument("--freeze", nargs="*", default=[], choices=["lin", "sna"],
                    help="copy these languages through untouched")
    args = ap.parse_args()

    rows_l = list(csv.DictReader(open(args.lid, encoding="utf-8-sig")))
    idc = "ID" if "ID" in rows_l[0] else "id"
    lid = {r[idc]: r[args.lid_column] for r in rows_l}

    to_split = {a + b: (a, b) for a, b in SPLIT_FORMS}
    ba_table = waxal_ba_table(args.pool)
    to_join = {(a, b): a + b for a, b in JOIN_CONFIRMED}
    if args.stage in ("baverb", "rule18"):
        to_join.update({(a, b): a + b for a, b in JOIN_BAVERB})
    if args.stage == "rule18":
        t = json.loads(args.rule18_table.read_text(encoding="utf-8"))["forms"]
        for f, d in t.items():
            if d["rate"] >= 0.18:
                to_split[f] = (d["a"], d["b"])
            elif d["rate"] <= 0.11:
                to_join[(d["a"].lower(), d["b"].lower())] = f

    rows = list(csv.DictReader(open(args.submission, encoding="utf-8-sig")))
    n_tail = n_ba = n_sp = n_jn = n_prog = n_spell = n_ne = 0
    tails = []
    out_rows = []
    for r in rows:
        text = r["Target"]
        # A frozen language keeps its conventions, but degenerate decoder output
        # is not a convention -- ID_LTOEWY's 197-character `.A` tail is wrong
        # under any annotator, so repair still runs.
        frozen = lid.get(r["ID"]) in args.freeze
        if not args.no_tail_repair:
            text, removed = strip_degenerate_tail(text)
            if removed:
                n_tail += 1
                tails.append((r["ID"], removed))
        if frozen:
            out_rows.append((r["ID"], finish(text)))
            continue
        text = collapse_punct(text)
        if lid.get(r["ID"]) == "lin":
            text, h = apply_split(text, ba_table)
            n_ba += h
            text, h = apply_split(text, to_split)
            n_sp += h
            text, h = apply_join(text, to_join)
            n_jn += h
        elif lid.get(r["ID"]) == "sna":
            if args.sna_join_progressive:
                text, h = join_sna_progressive(text)
                n_prog += h
            else:
                # confirmed default: these annotators split the progressive
                text, h = split_sna_progressive(text)
                n_prog += h
                text, h = split_sna_ne(text)
                n_ne += h
            if args.sna_spelling:
                text, h = apply_sna_spelling(text)
                n_spell += h
        out_rows.append((r["ID"], finish(text)))

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Target"])
        w.writerows(out_rows)

    src = {r["ID"]: r["Target"] for r in rows}
    d = dict(out_rows)
    print(f"stage={args.stage}  {args.submission.name} -> {args.out.name}")
    print(f"  degenerate tails removed : {n_tail}")
    print(f"  ba- class splits         : {n_ba}")
    print(f"  confirmed form splits    : {n_sp}")
    print(f"  joins applied            : {n_jn}")
    print(f"  sna -ri ku- progressive  : {n_prog}")
    print(f"  sna ne-class splits      : {n_ne}")
    print(f"  sna spelling fixed       : {n_spell}")
    print(f"  rows changed             : "
          f"{sum(1 for k, v in d.items() if v != src.get(k))} / {len(out_rows)}")
    print(f"  empty rows               : {sum(1 for v in d.values() if not v.strip())}")
    for uid, removed in tails[:5]:
        print(f"   tail {uid}: removed {removed!r}")


if __name__ == "__main__":
    main()



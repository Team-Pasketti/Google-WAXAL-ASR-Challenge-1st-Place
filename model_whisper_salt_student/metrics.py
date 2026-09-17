"""WER/CER and the competition score, averaged per utterance.

The challenge averages per utterance, not per corpus, and the difference is not
cosmetic. Corpus-level averaging weights an utterance by its length, so a
garbage transcript of a three-word clip is nearly free; the macro average gives
every clip a vote, and a clip whose reference is one word long can absorb a WER
of 62 on its own.

Measured on this project: two runaway hypotheses out of 892 moved the
leaderboard by 0.095, while the corpus-level distance between the same two
submissions was only 0.0219 -- and with a common reference the triangle
inequality caps any score gap at that distance, so corpus-level averaging
cannot produce what was observed. The per-utterance distance was 0.1260, which
can. Ranking decoder settings on the corpus figure therefore optimises
something the leaderboard does not score.
"""

from __future__ import annotations


def _edit(a, b) -> int:
    """Levenshtein distance over any two sequences."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    return _edit(r, hyp.split()) / max(len(r), 1)


def cer(ref: str, hyp: str) -> float:
    return _edit(ref, hyp) / max(len(ref), 1)


def score(refs: list[str], hyps: list[str], *, macro: bool = True) -> dict:
    """`1 - (WER + CER) / 2`, the competition metric."""
    if macro:
        w = sum(wer(r, h) for r, h in zip(refs, hyps)) / max(len(refs), 1)
        c = sum(cer(r, h) for r, h in zip(refs, hyps)) / max(len(refs), 1)
    else:
        wn = sum(len(r.split()) for r in refs)
        cn = sum(len(r) for r in refs)
        w = sum(_edit(r.split(), h.split()) for r, h in zip(refs, hyps)) / max(wn, 1)
        c = sum(_edit(r, h) for r, h in zip(refs, hyps)) / max(cn, 1)
    return {"wer": w, "cer": c, "score": 1.0 - (w + c) / 2.0}

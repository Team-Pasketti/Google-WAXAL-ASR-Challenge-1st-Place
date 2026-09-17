#!/usr/bin/env python3
"""Rescore Whisper n-best lists with an African-language LLM.

Whisper decides acoustically, one clip at a time, and its beam can only rank
what it heard. A model that has read far more Lingala and Shona than Whisper
ever transcribed can tell which of ten candidates is an actual sentence -- so
the LLM never proposes text, it only re-ranks what the beam already produced.
That also bounds the whole idea: the oracle over the n-best list is the
ceiling, and nothing here can exceed it.

Two stages, deliberately split:

  score   GPU. One forward pass per hypothesis, saved to json.
  tune    CPU. Grid-search the interpolation weights against macro WER/CER.

Tuning is where the guessing happens and it wants many cheap iterations, so it
must not drag a 9B model along with it.

The combination is the standard shallow-fusion form,

    combined = acoustic + lm_weight * lm_logprob_per_token + word_bonus * words

with the LM term length-normalised because Whisper's sequences_scores already
is. The word bonus exists because both log-prob terms grow more negative with
length, which biases the pick toward short hypotheses -- and under per-utterance
averaging a deletion on a short clip is expensive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import cer, score, wer  # noqa: E402

LANG_NAME = {"lin": "Lingala", "sna": "Shona"}


# ---------------------------------------------------------------- score

def run_score(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = json.loads(args.nbest.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": "cuda", "trust_remote_code": True}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    # An in-context prefix of real sentences in the same language, when given,
    # replaces the bare "Shona: " label. Three different language models -- a
    # WAXAL 5-gram, a 9B instruct model and a 14B base model -- all picked the
    # best of five candidates about 29% of the time against a 20% baseline,
    # which says the limit is the question rather than the model. Conditioning
    # on the domain asks a different question: not "is this likely African
    # text" but "does this continue text that looks like this".
    shots: dict[str, str] = {}
    if args.shots:
        import random
        rng = random.Random(0)
        for path in args.shots:
            lang = path.name.split(".")[0]
            lines = [t for t in (x.strip() for x in path.open(encoding="utf-8"))
                     if 40 < len(t) < 200]
            if not lines:
                continue
            pick = rng.sample(lines, min(args.n_shots, len(lines)))
            shots[lang] = "\n".join(pick) + "\n"
        print(f"in-context shots: "
              f"{ {k: len(v) for k, v in shots.items()} } chars")

    jobs = []
    for uid, rec in data.items():
        lang = rec["language"]
        prefix = shots.get(lang) or f"{LANG_NAME.get(lang, lang)}: "
        for i, hyp in enumerate(rec["nbest"]):
            jobs.append((uid, i, prefix, hyp))
    print(f"{len(data)} clips, {len(jobs)} hypotheses", flush=True)

    out: dict[str, list] = {uid: [None] * len(r["nbest"]) for uid, r in data.items()}
    t0 = time.time()
    for start in range(0, len(jobs), args.batch):
        chunk = jobs[start:start + args.batch]
        full = [p + (h if h else " ") for _, _, p, h in chunk]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length).to(model.device)
        with torch.inference_mode():
            logits = model(**enc).logits.float()
        logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
        target = enc.input_ids[:, 1:]
        picked = logprobs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        for j, (uid, idx, prefix, _hyp) in enumerate(chunk):
            # score only the hypothesis, never the language prefix: the prefix
            # is identical across a clip's candidates and would just add a
            # constant, but its length varies by language and would not
            n_prefix = len(tok(prefix, add_special_tokens=True).input_ids)
            mask = enc.attention_mask[j, 1:].bool().clone()
            mask[:max(n_prefix - 1, 0)] = False
            vals = picked[j][mask]
            out[uid][idx] = [float(vals.sum()), int(mask.sum())]
        if start % (args.batch * 20) == 0:
            n = start + len(chunk)
            print(f"  {n}/{len(jobs)} {n/max(time.time()-t0,1e-9):.1f} hyp/s",
                  flush=True)

    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out} in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------- tune

def run_tune(args: argparse.Namespace) -> None:
    data = json.loads(args.nbest.read_text(encoding="utf-8"))
    lm = json.loads(args.lm_scores.read_text(encoding="utf-8"))
    langs = sorted({r["language"] for r in data.values()})

    # How the LM log-prob is normalised matters more than the weight on it.
    # Dividing by token count -- the obvious choice, since Whisper's own score
    # is length-normalised -- drops the LM to chance on Lingala (21% vs a 20%
    # random baseline). The raw sum keeps the length information that actually
    # correlates with correctness and reaches 29%, and per-word normalisation
    # reaches 31% on Shona, above the acoustic score's own 30%.
    NORMS = {
        "sum": lambda total, ntok, words: total,
        "per_token": lambda total, ntok, words: total / max(ntok, 1),
        "per_word": lambda total, ntok, words: total / max(words, 1),
    }

    # Precomputeeach (clip, candidate) error rates once. The grid only ever picks
    # an index per clip, so scoring is then an average over lookups -- without
    # this, every one of the ~150 weight combinations re-runs Levenshtein over
    # the whole set and the search takes minutes instead of a second.
    cost: dict[str, list[tuple[float, float]]] = {}
    feats: dict[str, list[tuple[float, int, int]]] = {}
    for uid, rec in data.items():
        ref = rec["reference"]
        cost[uid] = [(wer(ref, h), cer(ref, h)) for h in rec["nbest"]]
        feats[uid] = [(lm[uid][i][0], lm[uid][i][1], len(h.split()))
                      if lm.get(uid) and lm[uid][i] else (0.0, 1, len(h.split()))
                      for i, h in enumerate(rec["nbest"])]

    def evaluate(w_lm: float, w_word: float, norm, subset: list[str]) -> float:
        sw = sc = 0.0
        for uid in subset:
            ac = data[uid]["scores"]
            best_i, best_s = 0, None
            for i, (total, ntok, words) in enumerate(feats[uid]):
                s = ac[i] + w_lm * norm(total, ntok, words) + w_word * words
                if best_s is None or s > best_s:
                    best_i, best_s = i, s
            w, c = cost[uid][best_i]
            sw += w
            sc += c
        n = max(len(subset), 1)
        return 1 - (sw / n + sc / n) / 2

    # the sum-normalised LM term is ~2 orders of magnitude larger than the
    # length-normalised acoustic score, so its useful weights are far smaller
    grid_lm = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 1.2]
    grid_w = [0.0, 0.005, 0.01, 0.02, 0.04]
    print(f"grid: {len(grid_lm)}x{len(grid_w)}x{len(NORMS)} per language\n")

    for lang in langs + ["ALL"]:
        subset = [u for u, r in data.items()
                  if lang == "ALL" or r["language"] == lang]
        if not subset:
            continue
        n = len(subset)
        one = 1 - (sum(cost[u][0][0] for u in subset) / n
                   + sum(cost[u][0][1] for u in subset) / n) / 2
        picks = [min(range(len(cost[u])), key=lambda i: cost[u][i][0])
                 for u in subset]
        orc = 1 - (sum(cost[u][i][0] for u, i in zip(subset, picks)) / n
                   + sum(cost[u][i][1] for u, i in zip(subset, picks)) / n) / 2
        best = max(((evaluate(a, b, fn, subset), a, b, nm)
                    for a in grid_lm for b in grid_w
                    for nm, fn in NORMS.items()), key=lambda t: t[0])
        print(f"=== {lang} (n={len(subset)}) ===")
        print(f"  1-best   {one:.4f}")
        print(f"  rescored {best[0]:.4f}   (norm={best[3]}, lm_weight={best[1]}, "
              f"word_bonus={best[2]})")
        print(f"  oracle   {orc:.4f}")
        print(f"  gain over 1-best {best[0]-one:+.4f}   "
              f"captured {100*(best[0]-one)/max(orc-one,1e-9):.0f}% of the oracle gap")
        # a weight tuned on 150 clips can fit noise; showing the runner-up
        # normalisation makes an unstable optimum visible rather than implied
        for nm, fn in NORMS.items():
            top = max(evaluate(a, b, fn, subset) for a in grid_lm for b in grid_w)
            print(f"     best with norm={nm:9s}: {top:.4f}")
        print()


# ---------------------------------------------------------------- apply

NORM_FNS = {
    "sum": lambda total, ntok, words: total,
    "per_token": lambda total, ntok, words: total / max(ntok, 1),
    "per_word": lambda total, ntok, words: total / max(words, 1),
}


def run_apply(args: argparse.Namespace) -> None:
    import csv

    data = json.loads(args.nbest.read_text(encoding="utf-8"))
    lm = json.loads(args.lm_scores.read_text(encoding="utf-8"))

    spec = {}
    for part in args.weights.split(","):
        lang, cfg = part.split("=")
        norm, w_lm, w_word = cfg.split(":")
        spec[lang.strip()] = (NORM_FNS[norm], float(w_lm), float(w_word))
    print(f"weights: { {k: (v[1], v[2]) for k, v in spec.items()} }")

    picked: dict[str, str] = {}
    changed = 0
    for uid, rec in data.items():
        cands = rec["nbest"]
        # clips over 30 seconds came through the long-form path with a single
        # hypothesis, so there is nothing to rescore and nothing to change
        if len(cands) == 1 or uid not in lm:
            picked[uid] = cands[0]
            continue
        norm, w_lm, w_word = spec.get(rec["language"], (NORM_FNS["sum"], 0.0, 0.0))
        best_i, best_s = 0, None
        for i, hyp in enumerate(cands):
            total, ntok = lm[uid][i]
            s = (rec["scores"][i] + w_lm * norm(total, ntok, len(hyp.split()))
                 + w_word * len(hyp.split()))
            if best_s is None or s > best_s:
                best_i, best_s = i, s
        picked[uid] = cands[best_i]
        changed += best_i != 0

    rows = list(csv.DictReader(open(args.lid, encoding="utf-8-sig")))
    id_col = "ID" if "ID" in rows[0] else "id"
    lang_col = "announced3_language"
    out = {}
    for r in rows:
        uid = r[id_col]
        text = picked.get(uid, "").strip()
        if args.lincaps and r.get(lang_col) == "lin" and text:
            for i, ch in enumerate(text):
                if ch.isalpha():
                    text = text[:i] + ch.upper() + text[i + 1:]
                    break
                if ch.isalnum():
                    break
            if text[-1] not in ".?!":
                text += "."
        out[uid] = text

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "Target"])
        for r in rows:
            w.writerow([r[id_col], out.get(r[id_col], "")])
    empty = sum(1 for v in out.values() if not v)
    print(f"wrote {args.out}: {len(rows)} rows, {changed} clips moved off "
          f"the 1-best, {empty} empty")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--nbest", type=Path, required=True)
    s.add_argument("--model", default="Sunbird/Sunflower-Qwen3.5-9B-bnb-4bit")
    s.add_argument("--load-4bit", action="store_true")
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--max-length", type=int, default=768)
    s.add_argument("--shots", type=Path, nargs="+",
                   help="corpus files, named <lang>.*, whose sentences become "
                        "an in-context prefix in place of the language label")
    s.add_argument("--n-shots", type=int, default=8)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=run_score)

    t = sub.add_parser("tune")
    t.add_argument("--nbest", type=Path, required=True)
    t.add_argument("--lm-scores", type=Path, required=True)
    t.set_defaults(func=run_tune)

    a = sub.add_parser("apply")
    a.add_argument("--nbest", type=Path, required=True)
    a.add_argument("--lm-scores", type=Path, required=True)
    a.add_argument("--weights", required=True,
                   help="per language, e.g. "
                        "'lin=per_word:0.0:0.0,sna=per_token:0.001:0.005' "
                        "as norm:lm_weight:word_bonus")
    a.add_argument("--lid", type=Path,
                   default=Path.home() / "wasal_lm/lm/newaudios_announced3_predictions.csv")
    a.add_argument("--lincaps", action="store_true",
                   help="capitalise the first letter and add a final stop on "
                        "Lingala, the rule the leaderboard submissions carry")
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=run_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# Orthography-aware consensus postprocessing

The character median is deliberately permissive: it can recover a transcription
that no individual recognizer produced, which is one reason it gives excellent
CER. The same freedom occasionally creates an output that is locally plausible
at character level but not a valid orthographic unit. This directory implements
a constrained second view of the ensemble: preserve the median by default,
apply a small set of language-structured boundary rules, and change a lexical
decision only when independent acoustic systems support the alternative.

This is not transcript rewriting. The pipeline never sees a reference label and
never asks a language model to generate text. Every lexical replacement must be
proposed by an ASR hypothesis for the same clip and survive fixed vote rules.

## One-command exact reproduction

From the repository root:

```bash
python postprocess/run_postprocess.py --verify-reference
```

Default input:

```text
postprocess/artifacts/raw_weighted_char_ensemble.csv
```

Default outputs:

```text
submission/submission_final_postprocessed.csv
submission/submission_final_postprocessed.audit.csv
```

The bundled competition artifacts reproduce the winning CSV byte-for-byte:

```text
SHA-256  9a665cd4b81a0443300b021eec373088a46b2ea9a746dfb826bb1923d434ad31
```

To postprocess a newly generated ensemble:

```bash
python postprocess/run_postprocess.py \
  --ensemble submission/submission_final_8.csv \
  --out submission/submission_final_postprocessed.csv
```

For new data, also provide a language-routing CSV and the candidate/verifier
member CSVs using `--lid`, `--candidate-members`, and `--diverse-members`.
Every CSV must have the same `ID,Target` schema and ID set.

## Method

### 1. Language-structured orthography

`apply_best_postprocessing.py` applies deterministic rules learned from WAXAL
orthographic statistics and grammatical structure:

- Lingala function-word boundaries (`namoni -> na moni`, `naye -> na ye`, etc.).
- Selective `ba/bazo/baza` treatment: noun-like corpus forms may split, while a
  fixed family of conjugated verbs is rejoined (`bazali`, `batie`, `balati`,
  `bakomi`, `bafandi`, `bavandi`, `basali`, `batelemi`, `balakisi`).
- Shona auxiliary/progressive boundaries (`varikufamba -> vari kufamba`) using
  a frequent table plus a conservative rare `ri + ku...` pattern.
- Initial capitalization and sentence-final punctuation.

`residual_confirmed_rules.py` catches the few progressive forms outside the
first pass's narrower prefix list. On the winning input it changed six rows.

### 2. High-recall repair proposals

Two proposal mechanisms inspect postprocessed member hypotheses:

- `ghost_tokens.py` finds an OOV ensemble token unsupported by the proposal
  members and searches for a one-edit, same-clip alternative with at least five
  votes.
- `majority_fix.py` proposes a one-edit replacement for an OOV token with at
  least four local votes and a margin of two.

The proposal pool intentionally contains several checkpoints from related
training runs. Their agreement has high recall for suspicious tokens, but it is
correlated and is therefore not accepted as final evidence.

### 3. Independent diverse verification

`diverse_verify_chain.py` checks every proposal against six architecturally
diverse systems. It keeps:

- a strict ghost repair only when diverse systems favor the replacement;
- a relaxed repair only with at least three diverse votes and a vote margin of
  at least two.

On the winning run, 21/24 strict proposals and 20/118 relaxed proposals
survived: 41 token repairs across 39 rows. The other 101 proposals were
rejected automatically.

This proposal/verifier split is the central idea. Correlated checkpoints are
useful detectors, while diversity is used as a precision control. The median
remains untouched whenever the independent evidence is insufficient.

## Included artifacts

```text
artifacts/raw_weighted_char_ensemble.csv  exact ensemble entering postprocess
artifacts/language_predictions.csv       Lingala/Shona routing
artifacts/candidate_members/             high-recall proposal hypotheses
artifacts/diverse_members/               independent verification hypotheses
corpus/pool/{lin,sna}.waxal.txt           orthographic frequency source
reference/winning_submission.csv         byte-level reproduction target
```

The prediction artifacts are included because postprocessing is a consensus
operation: the final CSV cannot be reconstructed from the median alone.

## Output guarantees

`run_postprocess.py` fails if:

- required artifacts are missing;
- the ensemble does not contain exactly `ID,Target`;
- output IDs or order differ from the ensemble;
- IDs are duplicated;
- any final target is empty;
- `--verify-reference` does not match the winning SHA-256.

Intermediate files are written in a temporary directory. Pass `--work-dir` to
retain them for inspection.

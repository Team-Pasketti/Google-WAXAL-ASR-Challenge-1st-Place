# Model 8 — SALT noisy-student (LoRA)

Entry 8 of `proc_data/r03_ensemble_subms.py`, weight 10.0.

Base: `Sunbird/asr-whisper-51-african-languages` — the same checkpoint Model 7
uses zero-shot. This member is that model adapted with a LoRA trained on its
own pseudo-labels, so the two are architecturally identical and differ only in
what they have been made robust to. Contributed independently of the other
seven models and trained on separate hardware.

**Adapter**: `models/step00500` (61 MB, PEFT). Also on the Hub — see
"Getting the adapter" below.

## What it is

The base is already trained on Google WAXAL including `lin` and `sna`, so there
is little new to teach it and a lot to forget. A full fine-tune of a Sunbird
Whisper was tried on this project and collapsed: WER 0.3302 → 0.4172 in 3k
steps. So the student is not taught new words. It is taught **robustness to
unheard voices**, which is the measured failure mode — the evaluation set is
entirely new speakers, and WAXAL holds only 90 distinct Lingala speakers and
160 Shona. No quantity of extra WAXAL audio adds speakers; augmentation
manufactures them.

Two decisions carry the result.

**The labels come from Whisper, never from WAXAL's transcribers.** WAXAL writes
`namoni` 3983 times against `na moni` 1359, while the evaluation references
split roughly 73% of the time. A student fitted to WAXAL text inherits a house
style that is backwards for the test set, and then has to be corrected
afterwards by a rule list covering only the forms someone thought to enumerate.
Relabelling with the teacher removes the annotators from the loop, and the
confirmed splits are baked into the targets before training.

**The teacher reads clean audio; the student reads augmented audio and must
still produce the teacher's transcript.** That asymmetry is the entire
gradient. An earlier attempt on this project pseudo-labelled 313k chunks with a
same-architecture teacher and no augmentation, and the student learned nothing —
it could already reproduce the labels, so most of each batch carried no signal.

Augmentation is weighted toward speaker identity, because that is what the
model must generalise over:

| | p | range |
|---|---|---|
| speed perturb (resample, no pitch correction — moves formants) | 0.7 | 0.9 / 1.0 / 1.1 |
| pitch shift (resample then pad/trim: pitch moves, duration does not) | 0.3 | 0.95–1.05 |
| additive noise | 0.2 | SNR 15–30 dB |
| reverb | 0.15 | decay 0.2–0.6 |
| SpecAugment | always | 2 freq masks ≤20, 2 time masks ≤40 |

Noise and reverb are deliberately small: they simulate a channel shift nobody
has evidence for.

## Environments

The adapter was exported by **peft 0.14.0**, which is what
`models/step00500/README.md` records and what `adapter_config.json` corroborates
(it carries `lora_bias`, `eva_config` and `exclude_modules`, all introduced in
0.14.0, and no `corda_config`, added in 0.15.0).

One caveat for anyone reproducing this. On the contributing machine these stages
never shared an environment: the Whisper stages (1 and 4, and inference) ran on
`transformers 4.46.3`, and the AfriqueQwen rescoring (stage 2) on
`transformers 5.14.1`, in two separate envs. That is why `run_inference.py`
shells out per stage instead of importing — it was written to straddle the two.
The repository pins `transformers==4.57.6`, which is between the two versions
and untested for these scripts. If stage 2 misbehaves, running it in its own
environment is the intended fallback, not a workaround.

## Training

Four stages. They are separate scripts because stage 2 needs a 14B LLM resident
on the GPU while 1 and 4 need Whisper.

```bash
# 0. materialise the WAXAL training audio as parquet shards.
#    t01 writes loose WAVs and drops speaker_id, which the
#    speaker-disjoint holdout needs, so this stage is separate.
python prepare_data.py --out ../input/waxal_parquet

# 1. teacher: beam-5 5-best over WAXAL train audio
python pseudo_label_teacher.py --data-dir ../input/waxal_parquet \
    --languages lin sna --split train --num-beams 5 --nbest 5 --batch 2 \
    --out nbest_waxal.json

# 2. LLM scoring of every hypothesis
python qwen_rescore.py score --nbest nbest_waxal.json \
    --model McGill-NLP/AfriqueQwen-14B --load-4bit --batch 16 \
    --out lmscore_waxal.json

# 3. pick, filter, apply confirmed splits -> training targets
python finalize_targets.py --nbest nbest_waxal.json \
    --lm-scores lmscore_waxal.json \
    --w-lin 0.01 --w-sna 0.3 --word-bonus-sna 0.04 \
    --out pseudo_targets.json

# 4. LoRA student
python run_training.py --labels pseudo_targets.json \
    --data-dir ../input/waxal_parquet --split train --languages lin sna \
    --holdout-speakers 15 --lora-r 16 --lora-alpha 32 \
    --target-modules q_proj k_proj v_proj out_proj \
    --lr 1e-5 --batch-size 8 --accum 4 --max-steps 2000 --save-every 500 \
    --out student_v1
```

Stage 3 settings and why: `--w-lin 0.01` is deliberately almost zero — rescoring
Lingala measured −0.002 at 0.3 and −0.0044 at 0.05 on the leaderboard, so its
acoustic score is already the better judge. Shona at 0.3 is worth about +0.002.
Hypotheses over 20 characters/second are dropped as repetition loops (one 1.0 s
clip produced 492 characters), and clips over 29 s are dropped because the
feature extractor truncates them silently, so the label would describe speech
the student never hears.

Stage 4 numbers from the actual run: 23,080 pseudo-labels, 228 speakers, 15 held
out → 22,608 train / 472 eval clips. 15,728,640 trainable of 1,559,219,200
(1.01%), as 128 encoder and 256 decoder LoRA adapters — 384 attention
projections in total, which is Whisper large-v3's 32 encoder self-attention
blocks plus 32 decoder blocks × (self + cross), times four projections each.

Note that `run_training.py` prints 256 and 512 for those two counts. It walks
`named_modules()`, and PEFT exposes both `lora_A` (a ModuleDict) and
`lora_A.default` (the Linear inside it), so the printed figure is exactly
double. The assertion it guards — encoder adapters must not be zero — is
unaffected.

### The checkpoint that shipped is step 500 of 2000

The run was scheduled for 2000 steps and the useful checkpoint is the first
one. Step 1000 was worse; the 2000-step checkpoint worse still. 500 steps ×
32 samples = 16,000 samples, about 0.7 of one epoch.

That is the labelled data running out, not the method saturating: teacher and
student saw the same 23,080 clips, a 1:1 ratio, where the noisy-student
literature runs closer to 230:1. A separate run with a complete 500-step
schedule (full anneal rather than truncation) was trained and evaluated but not
used.

## Inference

```bash
python run_inference.py --audio-dir ../input/newaudios --adapter models/step00500
```

Writes `submission/submission_whisper_salt_student.csv` in the `ID,Target`
schema `r03_ensemble_subms.py` consumes — no jsonl conversion needed for this
member. Internally: beam 5 keeping 5-best → AfriqueQwen-14B rescoring → shallow
fusion → capitalise and close with a stop.

**This member reads the raw audio.** `--audio-dir` defaults to
`input/newaudios/`, *not* the MSST-cleaned `input/newaudios_cleaned_mono/` that
the other seven members consume. That is deliberate, not an oversight: Model 8
was developed on a separate machine without the separation front-end, so the
hypotheses inside the winning ensemble came from unprocessed audio. Pointing it
at the cleaned audio yields a different member and will not reproduce the
0.769614 submission. It also means this member is the only one whose errors are
not shaped by the separation model — a small extra source of the decorrelation
the character ensemble is built to exploit.

Beam width was measured, not assumed: beam 1 → 5 was worth **+0.007**, larger
than any single postprocessing rule found on this project, and 10-best measured
**−0.022**, so 5 is the peak. The LLM never proposes text — it only re-ranks
what the beam produced, which makes the n-best oracle a hard ceiling on the
stage; it captures about 13% of that oracle.

**Difference from Model 7:** this member forces the *correct* language token per
clip (`lin → <|ln|>`, `sna → <|sn|>`). The deliberate mismatch that helps the
zero-shot checkpoint (Runyankole, see §4.3) does not transfer, because the
adapter was trained with the correct tokens in the prompt.

## Getting the adapter

Loading is two lines on top of the base model:

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

model = WhisperForConditionalGeneration.from_pretrained(
    "Sunbird/asr-whisper-51-african-languages", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "models/step00500")
model = model.merge_and_unload()      # fold LoRA in; inference needs no peft
```

`merge_and_unload()` matters for throughput — merged weights run at base-model
speed, unmerged adapters add a matmul per attention projection.

The adapter directory is a standard PEFT export (`adapter_config.json` +
`adapter_model.safetensors`) and `--adapter` also accepts an HF repo id.

## Validation

```bash
python run_validation.py --adapter models/step00500 \
    --manifest ../input/valid_parakeet_final_cleaned2_ssd_2_langs.jsonl
```

Prints WER/CER both per-utterance (the competition metric) and corpus-pooled.
Pass `--holdout student_v1/holdout.json` to restrict scoring to the 15 held-out
speakers. Holdout is by **speaker**, not by clip: clip-level holdout measures
memorisation of voices already trained on, the mistake WAXAL's own validation
split makes, where 64 of its 65 Lingala speakers also appear in train.

## One trap worth recording

`use_reentrant=False` in `gradient_checkpointing_enable` is a correctness
requirement, not a preference. Whisper's encoder is fed conv features rather
than embeddings, so nothing at its input requires grad; the reentrant
checkpoint implementation then prunes the whole encoder from the backward graph
and **every encoder LoRA adapter receives no gradient**. Measured on this model:
reentrant gives 0 encoder parameters with a gradient, non-reentrant gives 128.
The decoder trains normally either way, so the loss curve looks healthy while
the only part that can learn speaker invariance sits frozen.
`enable_input_require_grads()` does not fix it — it hooks embeddings.

`run_training.py` asserts on this: it counts encoder adapters after
`get_peft_model` and exits if the count is zero, because PEFT matches by name
suffix and a wrong `--target-modules` list fails silently. The earlier
decoder-only LoRA on this project scored 0.7459 against base Whisper's 0.7455 —
no acoustic gain — and narrowed the candidate pool from 4.70 to 4.63 distinct
of 5, which cost the rescoring stage downstream.

#!/usr/bin/env python3
"""Noisy-student fine-tune of Whisper-51 on its own pseudo-labels.

The point is not to teach the model new words -- it produced these labels
itself. It is to make it robust to voices it has not heard. The evaluation set
uses entirely different speakers, and WAXAL contains only 90 distinct Lingala
speakers and 160 Shona ones, so no amount of extra WAXAL audio adds speakers.
Augmentation manufactures them instead.

Two design points, both load-bearing:

1. The teacher read CLEAN audio; the student reads AUGMENTED audio and must
   still produce the teacher's transcript. That asymmetry is the entire
   learning signal. An earlier attempt on this project pseudo-labelled 313k
   chunks with a same-architecture teacher and no augmentation, and the student
   learned nothing -- it could already reproduce the labels, so most of each
   batch carried no gradient.

2. The labels come from Whisper, never from WAXAL's transcribers. WAXAL writes
   `namoni` 3983 times against `na moni` 1359, while the evaluation references
   split roughly 73% of the time; a student trained on WAXAL text inherits a
   convention that is backwards for the test set. Training on Whisper's output
   (with the measured splits already applied) sidesteps the annotators.

Augmentation is weighted toward speaker identity -- speed, VTLP-style spectral
warp, pitch -- because that is the measured failure mode. Noise and reverb are
available but small: they simulate a channel shift nobody has evidence for.

Holds out speakers rather than clips, since clip-level holdout would measure
memorisation of the same voices, which is exactly the mistake WAXAL validation
makes (64 of its 65 Lingala speakers also appear in train).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

SR = 16_000
MODEL = "Sunbird/asr-whisper-51-african-languages"
LANG_TOKEN = {"lin": "<|ln|>", "sna": "<|sn|>"}


# ------------------------------------------------------------------ augment

def speed_perturb(x: np.ndarray, rate: float) -> np.ndarray:
    """Resample without correcting pitch: shifts formants, i.e. the speaker."""
    if abs(rate - 1.0) < 1e-3:
        return x
    n = int(round(len(x) / rate))
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def spec_augment(feats: torch.Tensor, n_freq: int, freq_w: int,
                 n_time: int, time_w: int) -> torch.Tensor:
    """Mask bands of the log-mel. Applied after feature extraction."""
    out = feats.clone()
    n_mel, n_frames = out.shape[-2], out.shape[-1]
    for _ in range(n_freq):
        w = random.randint(0, freq_w)
        if w:
            f0 = random.randint(0, max(n_mel - w, 0))
            out[..., f0:f0 + w, :] = 0.0
    for _ in range(n_time):
        w = random.randint(0, time_w)
        if w:
            t0 = random.randint(0, max(n_frames - w, 0))
            out[..., :, t0:t0 + w] = 0.0
    return out


def add_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-6
    noise = np.random.randn(len(x)).astype(np.float32)
    noise *= rms / (10 ** (snr_db / 20.0)) / (float(np.sqrt(np.mean(noise ** 2))) or 1e-6)
    return (x + noise).astype(np.float32)


def reverb(x: np.ndarray, decay: float, taps: int) -> np.ndarray:
    ir = np.zeros(taps, dtype=np.float32)
    ir[0] = 1.0
    for i in range(1, taps, max(taps // 8, 1)):
        ir[i] = decay ** (i / max(taps, 1)) * random.uniform(0.1, 0.4)
    y = np.convolve(x, ir)[:len(x)]
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * (float(np.max(np.abs(x))) or 1.0)).astype(np.float32)


def augment_wave(x: np.ndarray, a: argparse.Namespace) -> np.ndarray:
    if random.random() < a.p_speed:
        x = speed_perturb(x, random.choice(a.speeds))
    if random.random() < a.p_pitch:
        # resample then pad/trim: pitch and formants move, duration does not
        r = random.uniform(*a.pitch_range)
        y = speed_perturb(x, r)
        x = (np.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y[:len(x)])
    if a.p_noise and random.random() < a.p_noise:
        x = add_noise(x, random.uniform(*a.snr_range))
    if a.p_reverb and random.random() < a.p_reverb:
        x = reverb(x, random.uniform(0.2, 0.6), int(0.05 * SR))
    return x


# ------------------------------------------------------------------ data

class PseudoSet(torch.utils.data.Dataset):
    def __init__(self, rows, audio_index, feat, tok, args):
        self.rows, self.audio_index = rows, audio_index
        self.feat, self.tok, self.args = feat, tok, args

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        uid, rec = self.rows[i]
        wave = self.audio_index(uid)
        wave = augment_wave(np.asarray(wave, dtype=np.float32), self.args)
        f = self.feat(wave, sampling_rate=SR,
                      return_tensors="pt").input_features[0]
        if self.args.spec_augment:
            f = spec_augment(f, self.args.freq_masks, self.args.freq_width,
                             self.args.time_masks, self.args.time_width)
        prompt = self.tok.convert_tokens_to_ids(
            ["<|startoftranscript|>", LANG_TOKEN[rec["language"]],
             "<|transcribe|>", "<|notimestamps|>"])
        body = self.tok(rec["label"], add_special_tokens=False).input_ids
        ids = prompt + body + [self.tok.eos_token_id]
        return {"input_features": f, "labels": torch.tensor(ids[:448])}


def collate(batch, pad_id):
    feats = torch.stack([b["input_features"] for b in batch])
    n = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b["labels"])] = b["labels"]
    return {"input_features": feats, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True,
                    help="finalize_pseudo_labels.py output")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / "kaggle-waxal-cleaned-16k-7lang")
    ap.add_argument("--split", default="train")
    ap.add_argument("--languages", nargs="+", default=["lin", "sna"])
    ap.add_argument("--holdout-speakers", type=int, default=15,
                    help="speakers reserved for eval, never seen in training")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    # The encoder is the point. Speaker identity is acoustic -- formants, pitch,
    # speaking rate -- so a decoder-only adapter cannot learn invariance to it,
    # it can only relearn the text distribution. The earlier LoRA on this
    # project froze the encoder deliberately and scored 0.7459 against base
    # Whisper's 0.7455: no acoustic gain, and it narrowed the candidate pool
    # (4.70 -> 4.63 distinct of 5), which cost the rescoring stage downstream.
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "out_proj"],
                    help="matched by name suffix, so these hit encoder "
                         "self-attention as well as the decoder")
    ap.add_argument("--encoder-only", action="store_true",
                    help="adapt the encoder alone and leave the decoder frozen")
    ap.add_argument("--full-finetune", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=500)
    # augmentation
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.9, 1.0, 1.1])
    ap.add_argument("--p-speed", type=float, default=0.7)
    ap.add_argument("--p-pitch", type=float, default=0.3)
    ap.add_argument("--pitch-range", type=float, nargs=2, default=[0.95, 1.05])
    ap.add_argument("--p-noise", type=float, default=0.2)
    ap.add_argument("--snr-range", type=float, nargs=2, default=[15.0, 30.0])
    ap.add_argument("--p-reverb", type=float, default=0.15)
    ap.add_argument("--spec-augment", action="store_true", default=True)
    ap.add_argument("--freq-masks", type=int, default=2)
    ap.add_argument("--freq-width", type=int, default=20)
    ap.add_argument("--time-masks", type=int, default=2)
    ap.add_argument("--time-width", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import glob
    from datasets import Audio, load_dataset
    from transformers import (WhisperFeatureExtractor,
                              WhisperForConditionalGeneration,
                              WhisperTokenizerFast)

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    print(f"{len(labels)} pseudo-labels", flush=True)

    # speaker-disjoint holdout: clip-level holdout would only measure
    # memorisation of voices the model already trained on
    speakers = sorted({v["speaker_id"] for v in labels.values() if v["speaker_id"]})
    random.Random(0).shuffle(speakers)
    held = set(speakers[:args.holdout_speakers])
    train_rows = [(k, v) for k, v in labels.items() if v["speaker_id"] not in held]
    eval_rows = [(k, v) for k, v in labels.items() if v["speaker_id"] in held]
    print(f"{len(speakers)} speakers -> {len(held)} held out "
          f"({len(train_rows)} train / {len(eval_rows)} eval clips)", flush=True)

    # audio stays on disk; index maps id -> (shard dataset, row)
    print("indexing audio...", flush=True)
    store = {}
    for lang in args.languages:
        files = sorted(glob.glob(str(args.data_dir / f"{lang}_asr"
                                     / f"{args.split}-*.parquet")))
        ds = load_dataset("parquet", data_files=files, split="train")
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))
        for i, uid in enumerate(ds["id"]):
            if uid in labels:
                store[uid] = (ds, i)
    print(f"indexed {len(store)} clips", flush=True)

    def audio_of(uid):
        ds, i = store[uid]
        return ds[i]["audio"]["array"]

    feat = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    tok = WhisperTokenizerFast.from_pretrained(MODEL)
    import transformers
    dtype_kw = ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
                else "torch_dtype")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, **{dtype_kw: torch.bfloat16}).cuda()
    model.config.use_cache = False
    # use_reentrant=False is required, not a preference. Whisper's encoder is
    # fed conv features rather than embeddings, so nothing at its input
    # requires grad; the reentrant checkpoint implementation then prunes the
    # whole encoder from the backward graph and every encoder LoRA adapter
    # receives no gradient at all. Measured on this model: reentrant gives 0
    # encoder parameters with a gradient, non-reentrant gives 128. The decoder
    # trains normally either way, so the loss curve looks healthy while the
    # encoder -- the only part that can learn speaker invariance -- is frozen.
    # enable_input_require_grads() does not fix it; it hooks embeddings.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model
        if args.encoder_only:
            targets = [n for n, _ in model.named_modules()
                       if n.startswith("model.encoder")
                       and n.rsplit(".", 1)[-1] in set(args.target_modules)]
        else:
            targets = list(args.target_modules)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", target_modules=targets))
        model.print_trainable_parameters()
        # Confirm the encoder is actually being adapted rather than assumed to
        # be: peft matches by name suffix, so a wrong module list fails silently
        # and would leave this run identical to the decoder-only LoRA that
        # already failed.
        enc = sum(1 for n, _ in model.named_modules()
                  if "lora_A" in n and "encoder" in n and "encoder_attn" not in n)
        dec = sum(1 for n, _ in model.named_modules()
                  if "lora_A" in n and "decoder" in n)
        print(f"LoRA adapters: {enc} in the encoder, {dec} in the decoder",
              flush=True)
        if enc == 0:
            raise SystemExit("no encoder adapters -- check --target-modules")

    train_rows = [(k, v) for k, v in train_rows if k in store]
    ds_train = PseudoSet(train_rows, audio_of, feat, tok, args)
    loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2,
        collate_fn=lambda b: collate(b, tok.pad_token_id), drop_last=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.max_steps, pct_start=0.1)

    args.out.mkdir(parents=True, exist_ok=True)
    step, done, running = 0, 0, 0.0
    model.train()
    while step < args.max_steps:
        for batch in loader:
            out = model(input_features=batch["input_features"].cuda().to(torch.bfloat16),
                        labels=batch["labels"].cuda())
            (out.loss / args.accum).backward()
            running += out.loss.item()
            done += 1
            if done % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{args.max_steps} "
                      f"loss {running/max(done,1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
                running, done = 0.0, 0
            if step % args.save_every == 0 or step == args.max_steps:
                d = args.out / f"step{step:05d}"
                model.save_pretrained(str(d))
                print(f"saved {d}", flush=True)
            if step >= args.max_steps:
                break
    json.dump({"held_out_speakers": sorted(held),
               "eval_clip_ids": [k for k, _ in eval_rows]},
              open(args.out / "holdout.json", "w"), indent=2)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()

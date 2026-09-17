import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")

if __name__ == '__main__':
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_cache/'

import evaluate
import numpy as np
import soundfile as sf
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.granite_speech import (
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechProcessor,
)

ASR_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #
def load_manifest(path, prompt, max_duration=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio = item["audio_filepath"]
            if not os.path.isfile(audio):
                raise FileNotFoundError(audio)
            if max_duration and item.get("duration", 0) > max_duration:
                continue
            text = item["text"].strip()
            if not text:
                continue
            rows.append({"audio": audio, "text": text, "prompt": prompt})
    return Dataset.from_list(rows)


class GraniteCollator:
    """Промпт паддится слева, таргет — справа; лосс только по таргету."""

    def __init__(self, processor, inference_mode=False):
        self.processor = processor
        self.inference_mode = inference_mode
        self.sr = processor.audio_processor.sampling_rate

    def _read_audio(self, path):
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != self.sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        return wav

    def __call__(self, examples):
        prompts = [ex["prompt"] for ex in examples]
        audios = [self._read_audio(ex["audio"]) for ex in examples]

        processed = self.processor(
            prompts, audios, return_tensors="pt", padding=True, padding_side="left"
        )
        input_ids = processed.input_ids
        attention_mask = processed.attention_mask
        labels = None

        if not self.inference_mode:
            eos = self.processor.tokenizer.eos_token
            targets = self.processor.tokenizer(
                [ex["text"] + eos for ex in examples],
                return_tensors="pt",
                padding=True,
                padding_side="right",
            )
            input_ids = torch.cat([input_ids, targets.input_ids], dim=1)
            attention_mask = torch.cat([attention_mask, targets.attention_mask], dim=1)
            labels = targets.input_ids.clone()
            labels[~targets.attention_mask.bool()] = -100
            # часть промпта в лосс не входит
            labels = torch.cat(
                [torch.full_like(processed.input_ids, -100), labels], dim=1
            )

        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "input_features": processed.input_features,
                "input_features_mask": processed.input_features_mask,
            }
        )


# --------------------------------------------------------------------------- #
# Что размораживаем
# --------------------------------------------------------------------------- #
def should_train(name, train_last_n_layers, total_lm_layers):
    # проектор, LoRA и lm_head тренируем всегда
    if "projector" in name or "lora" in name or "lm_head" in name:
        return True
    if train_last_n_layers <= 0:
        return False
    prefix = "language_model.model.layers."
    if not name.startswith(prefix):
        return False
    idx_str = name[len(prefix):].split(".", 1)[0]
    if not idx_str.isdigit():
        return False
    return int(idx_str) >= max(total_lm_layers - train_last_n_layers, 0)


# --------------------------------------------------------------------------- #
# Валидация через generate()
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate_wer_cer(model, processor, dataset, batch_size=8, num_workers=4):
    from torch.utils.data import DataLoader

    collator = GraniteCollator(processor, inference_mode=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers
    )
    was_training = model.training
    model.eval()

    preds = []
    for batch in loader:
        batch = batch.to(model.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                **batch, max_new_tokens=1024, do_sample=False, num_beams=1
            )
        out = out[:, batch.input_ids.shape[1]:].cpu()
        preds += processor.tokenizer.batch_decode(out, skip_special_tokens=True)

    refs = [t.strip().lower() for t in dataset["text"]]
    preds = [p.strip().lower() for p in preds]

    wer = evaluate.load("wer").compute(references=refs, predictions=preds)
    cer = evaluate.load("cer").compute(references=refs, predictions=preds)
    if was_training:
        model.train()
    return wer, cer


class ValidCheckpointCallback(TrainerCallback):
    """Аналог вашего колбэка: считает WER/CER и переименовывает чекпоинт."""

    def __init__(self, processor, dataset, batch_size=8):
        self.processor = processor
        self.dataset = dataset
        self.batch_size = batch_size

    def on_save(self, args, state, control, model=None, **kwargs):
        wer, cer = evaluate_wer_cer(
            model, self.processor, self.dataset, batch_size=self.batch_size
        )
        print(f"[step {state.global_step}] WER={wer:.4f} CER={cer:.4f}")
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt.is_dir():
            try:
                ckpt.rename(
                    ckpt.parent / f"{ckpt.name}_wer_{wer:.4f}_cer_{cer:.4f}"
                )
            except Exception as e:  # noqa: BLE001
                print("Rename error:", e)
        return control


def main():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    base_model_name = "ibm-granite/granite-speech-4.1-2b"

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=root_path + "input/train.jsonl")
    ap.add_argument("--valid", default=root_path + "input/valid.jsonl")
    ap.add_argument("--model-name", default=base_model_name)
    ap.add_argument("--output-dir", default=root_path + "model_ibm_granite/models/granite-speech")
    ap.add_argument("--train-last-n-layers", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--max-duration", type=float, default=60.0)
    ap.add_argument("--eval-subset", type=int, default=200, help="valid num")
    args = ap.parse_args()

    processor = GraniteSpeechProcessor.from_pretrained(base_model_name)
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16
    )

    train_ds = load_manifest(args.train, ASR_PROMPT, args.max_duration)
    valid_ds = load_manifest(args.valid, ASR_PROMPT, args.max_duration)
    eval_ds = valid_ds.select(range(min(args.eval_subset, len(valid_ds))))
    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Eval subset: {len(eval_ds)}")

    # промпт нужно построить через chat template
    tok = processor.tokenizer
    def add_chat(ex):
        ex["prompt"] = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return ex
    train_ds = train_ds.map(add_chat)
    eval_ds = eval_ds.map(add_chat)

    total_lm_layers = len(model.language_model.model.layers)
    for name, p in model.named_parameters():
        if args.model_name == base_model_name:
            p.requires_grad = should_train(name, args.train_last_n_layers, total_lm_layers)
        else:
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%), "
          f"LM layers: {total_lm_layers}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=15,
        eval_strategy="no",
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        report_to="none",
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=GraniteCollator(processor),
        processing_class=processor,
        callbacks=[ValidCheckpointCallback(processor, eval_ds, args.batch_size)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    wer1, cer1 = evaluate_wer_cer(model, processor, eval_ds, args.batch_size)
    print(f"Final: WER={wer1:.4f}, CER={cer1:.4f})")


if __name__ == "__main__":
    main()
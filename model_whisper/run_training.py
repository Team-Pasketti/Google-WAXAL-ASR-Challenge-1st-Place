import os
os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_cache/'

import json
from pathlib import Path
import soundfile as sf
import evaluate
from typing import Dict, List, Union
import pandas as pd
import torch
import numpy as np

from datasets import Dataset, Features, Value
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from dataclasses import dataclass
from run_validation import valid_checkpoint


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_ROOT = PROJECT_ROOT + '../input/'


@dataclass
class DataCollatorSpeechSeq2SeqWithOnTheFlyAudio:
    processor: WhisperProcessor
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Union[List[int], str]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        label_features = []

        for feature in features:
            audio_path = feature["input_values"]
            try:
                speech_array, sr = sf.read(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                speech_array = np.zeros(self.sampling_rate)

            extracted_features = self.processor.feature_extractor(
                speech_array,
                sampling_rate=self.sampling_rate,
                return_tensors="pt"
            ).input_features[0]

            input_features.append({"input_features": extracted_features})
            label_features.append({"input_ids": feature["labels"]})

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class ValidCheckpointCallback(TrainerCallback):
    def on_save(self, args: Seq2SeqTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        print(f"\n[ValidCheckpointCallback] Checkpoint saved: {ckpt_dir}")

        score = valid_checkpoint(ckpt_dir)
        try:
            os.rename(ckpt_dir, ckpt_dir + f'_cer_{score:.4f}')
        except Exception as e:
            print("Error renaming: ", str(e))


def make_preprocess_batch(processor):
    def preprocess_text_only(examples):
        processed_batch = {}
        processed_batch["labels"] = [
            processor.tokenizer(text=ex).input_ids for ex in examples["phonetic_text"]
        ]
        processed_batch['input_values'] = examples["audio_path"]
        processed_batch["duration"] = examples["duration"]
        return processed_batch

    return preprocess_text_only


if __name__ == '__main__':
    def load_data(jsonl_name):
        in1 = open(DATA_ROOT + f"/{jsonl_name}", "r", encoding="utf-8")
        lines = in1.readlines()
        res = []
        for l in lines:
            item = json.loads(l.strip())
            audio = item['audio_filepath']
            if not os.path.isfile(audio):
                print(f"No file: {audio}")
                exit()
            res.append([audio, item['text'], item['duration']])
        return pd.DataFrame(res, columns=['audio_path', 'phonetic_text', 'duration'])


    df_train = load_data("train.jsonl")
    df_valid = load_data("valid.jsonl")

    schema = Features({
        "phonetic_text": Value("string"),
        "audio_path": Value("string"),
        "duration": Value("float"),
    })

    dataset_train = Dataset.from_pandas(df_train.reset_index(drop=True), features=schema)
    dataset_valid = Dataset.from_pandas(df_valid.reset_index(drop=True), features=schema)

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3", language="english", task="transcribe"
    )

    processed_dataset_train = dataset_train.map(
        make_preprocess_batch(processor),
        batched=True,
        num_proc=4,
        remove_columns=dataset_train.column_names
    )
    processed_dataset_valid = dataset_valid.map(
        make_preprocess_batch(processor),
        batched=True,
        num_proc=4,
        remove_columns=dataset_valid.column_names
    )

    print(f"Train: {len(processed_dataset_train)}, Eval: {len(processed_dataset_valid)}")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    data_collator = DataCollatorSpeechSeq2SeqWithOnTheFlyAudio(processor=processor)

    def make_compute_metrics(processor):
        cer_metric = evaluate.load("cer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

            return {"cer": cer_metric.compute(references=label_str, predictions=pred_str)}

        return compute_metrics


    output_dir = str(Path(PROJECT_ROOT) / "models" / "whisper-large-v3-finetune")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # Увеличено под 96GB
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        learning_rate=1e-5,
        num_train_epochs=20,

        eval_strategy="steps",
        eval_steps=3000,
        save_steps=3000,


        logging_steps=100,
        warmup_steps=500,
        lr_scheduler_type="linear",
        bf16=True,  # Включено для архитектуры Blackwell
        tf32=True,  # Ускоряет матричные вычисления на Ampere+
        predict_with_generate=True,  # Обязательно для оценки генерации
        generation_max_length=225,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_total_limit=15,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset_train,
        eval_dataset=processed_dataset_valid,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
        callbacks=[ValidCheckpointCallback()],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print("Evaluation Results:")
    print(f"  CER: {eval_results['eval_cer']:.4f}")
    print(f"  Loss: {eval_results['eval_loss']:.4f}")

    save_dir = Path(PROJECT_ROOT) / "models" / "whisper-final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    processor.save_pretrained(str(save_dir))
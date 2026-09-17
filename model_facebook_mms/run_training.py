import os
os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_cache/'

import sys
from dataclasses import dataclass
import json
from pathlib import Path
import soundfile as sf
import evaluate
from typing import Dict, List, Union, Optional

from loguru import logger
import numpy as np
import pandas as pd

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from datasets import Dataset, Features, Value
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from run_validation import valid_checkpoint

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_ROOT = PROJECT_ROOT + '../input/'
MODEL_ID = "facebook/mms-1b-all"

# "adapter" or "full"
TRAIN_MODE = str(sys.argv[1])
VALID_MANIFEST = DATA_ROOT + "valid.jsonl"
if __name__ == '__main__':
    print("Train mode: {}".format(TRAIN_MODE))

def read_manifest(path):
    res = []
    with open(path, "r", encoding="utf-8") as in1:
        for l in in1:
            item = json.loads(l.strip())
            audio = item['audio_filepath']
            if not os.path.isfile(audio):
                print("No file: {}".format(audio))
                exit()
            res.append([audio, item['text'], item['duration']])
    return pd.DataFrame(res, columns=['audio_path', 'phonetic_text', 'duration'])


@dataclass
class DataCollatorCTCWithOnTheFlyAudio:
    processor: Wav2Vec2Processor
    sampling_rate: int = 16000
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], str]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = []
        label_features = []

        for feature in features:
            audio_path = feature["input_values"]
            try:
                speech_array, sr = sf.read(audio_path)
            except Exception as e:
                print(audio_path)
                continue

            extracted_features = self.processor(
                speech_array,
                sampling_rate=self.sampling_rate
            ).input_values[0]

            input_features.append({"input_values": extracted_features})
            label_features.append({"input_ids": feature["labels"]})

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


class ValidCheckpointCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        print(f"\n[ValidCheckpointCallback] Чекпоинт сохранен: {ckpt_dir}")
        print("[ValidCheckpointCallback] Запускаю процесс валидации...")

        score_cer, score_wer = valid_checkpoint(ckpt_dir, VALID_MANIFEST)
        try:
            os.rename(ckpt_dir, ckpt_dir + '_cer_{:.4f}_wer_{:.4f}'.format(score_cer, score_wer))
        except Exception as e:
            print("Error: ", str(e))


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


VALID_CHARS = ['a', 'i', 'e', 'n', 'o', 'k', 'm', 'u', 'r', 'b', 't', 'z', 's', 'y', 'w', 'l', 'v', 'h', 'd',
               'g', 'p', '.', 'c', 'f', ',', 'M', 'P', 'N', 'j', 'A', 'V', 'K', 'B', 'I', 'C', 'Z', 'E', 'S', 'U',
               'D', 'R', '-', 'é', 'L', 'T', '\n', 'O', 'H', "'", 'Y', 'q', 'F', 'W', 'x', '"', 'G', '!', 'J', '(',
               ')', 'î', 'è', 'í', ';', 'ô', 'à', 'ñ', 'â', 'ú', 'ê', 'ï', '?', 'ç', 'É', 'ó', '&', '`', 'ì', 'á',
               'X', 'ù', 'ĺ', 'ò', 'û', '\xa0', 'ü', '=', 'Q', 'ĝ', 'þ', 'œ', '«', '»', 'Ķ', '\\', 'ķ', '“', '”',
               'Ĺ', 'Œ', '/', 'ā', ':']


if __name__ == '__main__':
    df_train = read_manifest(DATA_ROOT + "/train.jsonl")
    df_valid = read_manifest(DATA_ROOT + "/valid.jsonl")

    SR = 16000
    print(df_train.head())

    schema = Features(
        {
            "phonetic_text": Value("string"),
            "audio_path": Value("string"),
            "duration": Value("float"),
        }
    )

    dataset_train = Dataset.from_pandas(df_train.reset_index(drop=True), features=schema)
    dataset_valid = Dataset.from_pandas(df_valid.reset_index(drop=True), features=schema)

    unk_tok, pad_tok, space_tok = "[UNK]", "[PAD]", "|"

    all_toks = sorted([char for char in VALID_CHARS if char != " "]) + [unk_tok, pad_tok, space_tok]
    vocab_dict = {char: idx for idx, char in enumerate(all_toks)}

    vocab_path = DATA_ROOT + "/vocab_facebook4/facebook_mms_vocab.json"
    os.makedirs(DATA_ROOT + "/vocab_facebook4/", exist_ok=True)
    with open(vocab_path, "w") as out:
        json.dump(vocab_dict, out)

    print('Vocab length: {}'.format(len(vocab_dict)))

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token=unk_tok,
        pad_token=pad_tok,
        word_delimiter_token=space_tok
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    data_collator = DataCollatorCTCWithOnTheFlyAudio(
        processor=processor,
        sampling_rate=SR,
        padding=True
    )

    if TRAIN_MODE == "adapter":
        start_checkpoint = MODEL_ID
    elif TRAIN_MODE == "full":
        start_checkpoint = PROJECT_ROOT + 'models/facebook_mms_adapter_final/'
    else:
        print("Unknown train mode. Use adapter or full!")
        exit()

    model = Wav2Vec2ForCTC.from_pretrained(
        start_checkpoint,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
        vocab_size=len(processor.tokenizer),
    )

    if TRAIN_MODE == "adapter":
        model.init_adapter_layers()
        model.freeze_base_model()          # freeze model.wav2vec2, lm_head stays trainable
        for param in model._get_adapters().values():
            param.requires_grad = True
    else:
        model.freeze_feature_encoder()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("Trainable params: {:,} / {:,} ({:.3f}%)".format(trainable, total, 100 * trainable / total))

    def make_preprocess_batch(processor):
        def preprocess_text_only(examples):
            processed_batch = {}
            processed_batch["labels"] = [
                processor(text=ex.replace(" ", "|")).input_ids for ex in examples["phonetic_text"]
            ]
            processed_batch['input_values'] = examples["audio_path"]
            processed_batch["duration"] = examples["duration"]
            return processed_batch
        return preprocess_text_only

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

    before_filter = len(processed_dataset_train)

    def is_valid_ctc_sample(example):
        DOWNSAMPLE = 320
        input_len = int(example["duration"] * 16000)
        label_len = len(example["labels"])
        return (input_len // DOWNSAMPLE) > label_len and label_len > 0 and input_len > 0

    processed_dataset_train = processed_dataset_train.filter(is_valid_ctc_sample, num_proc=4)
    print(
        f"CTC filter: {before_filter} -> {len(processed_dataset_train)} samples "
        f"({before_filter - len(processed_dataset_train)} removed)"
    )

    train_dataset = processed_dataset_train
    eval_dataset = processed_dataset_valid
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    def make_compute_metrics(processor):
        cer_metric = evaluate.load("cer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids)
            label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
            return {"cer": cer_metric.compute(references=label_str, predictions=pred_str)}

        return compute_metrics

    output_dir = str(Path(PROJECT_ROOT) / "models" / "facebook_mms_{}".format(TRAIN_MODE))

    from transformers.trainer_callback import ProgressCallback
    class TrainOnlyProgressCallback(ProgressCallback):
        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
            pass

    # ============================================================
    # TRAINING ARGUMENTS
    # ============================================================

    if TRAIN_MODE == "adapter":
        learning_rate = 1e-3
        num_train_epochs = 4
        batch_size = 4
        gradient_accumulation_steps = 6
        eval_steps = 3000
    else:
        learning_rate = 1e-5
        num_train_epochs = 10
        batch_size = 12
        gradient_accumulation_steps = 2
        eval_steps = 1000

    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=0.0,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=100,
        warmup_steps=500,
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_total_limit=15,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=make_compute_metrics(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            ValidCheckpointCallback()
        ],
    )

    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(TrainOnlyProgressCallback)

    if 'checkpoint' in start_checkpoint and 0:
        trainer.train(resume_from_checkpoint=start_checkpoint)
    else:
        trainer.train()

    eval_results = trainer.evaluate()
    print("Evaluation Results:")
    print(f"  CER: {eval_results['eval_cer']:.4f}")
    print(f"  Loss: {eval_results['eval_loss']:.4f}")

    save_dir = Path(PROJECT_ROOT) / "models" / "facebook_mms_{}_final".format(TRAIN_MODE)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(save_dir))
    processor.save_pretrained(str(save_dir))
    processor.feature_extractor.save_pretrained(str(save_dir))
    torch.save(training_args, save_dir / "training_args.pt")

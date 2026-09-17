import os
import json
import os
import gc
import shutil
import time
import evaluate
import numpy as np

from loguru import logger
from tqdm import tqdm
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

warnings.filterwarnings("ignore")

MAX_BATCH_SIZE = 8
MEMORY_BUDGET = 18
ATTENTION_PENALTY = 0.0125
INFERENCE_DTYPE = torch.bfloat16


def get_dynamic_batches(items):
    total = len(items)
    i = 0

    while i < total:
        l = items[i]['duration']
        if l > 200:
            batch_size = 1
        else:
            batch_size = int(MEMORY_BUDGET / (l + ATTENTION_PENALTY * (l ** 2)))
            if batch_size <= 0:
                batch_size = 1
            batch_size = min(batch_size, MAX_BATCH_SIZE)

        yield items[i:i + batch_size]
        i += batch_size


class AudioBatchDataset(Dataset):
    def __init__(self, batches, data_dir):
        self.batches = list(batches)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]
        audios = []

        for item in batch:
            audio_name = item["audio_filepath"]
            waveform, sr = librosa.load(str(audio_name), sr=16000, mono=True)
            audios.append(waveform)

        return {
            "audios": audios,
            "batch_items": batch
        }


def identity_collate_fn(x):
    return x


def inference_checkpoint(model_path, manifest_path):
    start_time = time.time()

    processor = Wav2Vec2Processor.from_pretrained(model_path)

    model = Wav2Vec2ForCTC.from_pretrained(
        model_path,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
        vocab_size=len(processor.tokenizer),
        torch_dtype=INFERENCE_DTYPE,
    )
    model.to('cuda')
    model.eval()

    root_path = os.path.dirname(os.path.dirname(__file__)) + '/'
    data_dir = root_path + "input/"

    with open(manifest_path, "r", encoding='utf8') as fr:
        items = [json.loads(line) for line in fr]

    items.sort(key=lambda x: x["duration"], reverse=True)

    logger.info(f"Processing {len(items)} utterances")
    batches = list(get_dynamic_batches(items))
    logger.info("Number of batches: {}".format(len(batches)))
    dataset = AudioBatchDataset(batches, data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        prefetch_factor=2,
        collate_fn=identity_collate_fn,
    )

    from pyctcdecode import build_ctcdecoder

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    labels = list(sorted_vocab_dict.keys())

    labels[processor.tokenizer.pad_token_id] = ""

    word_delimiter = processor.tokenizer.word_delimiter_token
    if word_delimiter is not None and word_delimiter in labels:
        labels[labels.index(word_delimiter)] = " "
    elif "|" in labels:
        labels[labels.index("|")] = " "

    seen = set()
    for i in range(len(labels)):
        if labels[i] in seen:
            labels[i] = f"<dummy_{i}>"
        seen.add(labels[i])

    decoder = build_ctcdecoder(
        labels=labels,
        # kenlm_model_path="path/to/your/language_model.arpa",
    )

    predictions = {}
    total = 0

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Valid", disable=True)

        for batch_data in pbar:
            audios = batch_data["audios"]
            batch = batch_data["batch_items"]

            inputs = processor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_values = inputs.input_values.to('cuda', dtype=INFERENCE_DTYPE)
            attention_mask = inputs.attention_mask.to('cuda')

            logits = model(input_values, attention_mask=attention_mask).logits
            logits_np = logits.to(torch.float32).cpu().numpy()

            transcription = []
            for b_idx in range(logits_np.shape[0]):
                text = decoder.decode(logits_np[b_idx], beam_width=500)
                transcription.append(text)

            for i, r in enumerate(transcription):
                item = batch[i]
                predictions[item["audio_filepath"]] = r
                total += 1

            pbar.set_postfix({
                "bs": len(batch),
                "dur": f"{batch[0]['duration']:.1f}s",
                "processed": total
            })

            del input_values, attention_mask, logits

    logger.success("Transcription complete in {:.2f} sec".format(
        time.time() - start_time
    ))

    submission_path = root_path + "submission/submission_facebook_mms.jsonl"
    os.makedirs(root_path + "/submission/", exist_ok=True)

    all_real_text = []
    all_pred_text = []
    with open(submission_path, "w", encoding='utf8') as fw:
        for item in items:
            all_real_text.append(item['text'])
            phonetic = predictions.get(item["audio_filepath"], "")
            all_pred_text.append(phonetic)
            item["text"] = phonetic
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    manifest_path = root_path + "input/test2.jsonl"
    # model_path = root_path + 'model_facebook_mms/models/checkpoint-12450_cer_0.0767_wer_0.3252/'
    model_path = 'ZFTurbo/facebook_mms_1b_lingala_shona'
    inference_checkpoint(model_path, manifest_path)


"""
checkpoint-12450_cer_0.0767_wer_0.3252 - WER: 0.3209 CER: 0.0762 (Beam 100)
WER: 0.3207 CER: 0.0761 - WER: 0.3207 CER: 0.0761 (Beam 500)
WER: 0.3204 CER: 0.0760 (Beam 5000)
"""
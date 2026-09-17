import os

if __name__ == '__main__':
    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    os.environ['PYTHONWARNINGS'] = "ignore"
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_cache/'


from itertools import islice
import json
import os
import gc
import shutil
import time
import evaluate
import numpy as np
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import logging
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

warnings.filterwarnings("ignore")

MAX_BATCH_SIZE = 20

def get_dynamic_batches(items):
    total = len(items)
    i = 0
    memory_budget = 50
    attention_penalty = 0.0125

    while i < total:
        l = items[i]['duration']
        if items[i]['duration'] > 200:
            batch_size = 1
        else:
            batch_size = int(memory_budget / (l + attention_penalty * (l ** 2)))
            if batch_size <= 0:
                batch_size = 1
            # batch_size = int(2000 / items[i]['audio_duration_sec']) + 1
            batch_size = min(batch_size, MAX_BATCH_SIZE)

        # print(batch_size)
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
        languages = []
        max_new_tokens = 0

        for item in batch:
            audio_name = item["audio_filepath"]
            waveform, sr = librosa.load(str(audio_name), sr=16000, mono=True)
            audios.append(waveform)
            languages.append("English")
            max_new_tokens = max(max_new_tokens, int(item["duration"] * 20))

        return {
            "audios": audios,
            "languages": languages,
            "max_new_tokens": max_new_tokens,
            "batch_items": batch
        }


def identity_collate_fn(x):
    return x


def valid_checkpoint(model_path, manifest_path):
    start_time = time.time()
    # Load pretrained Wav2Vec2 model
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_path,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,  # Replace inf CTC loss with 0 to prevent NaN gradients
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
        vocab_size=len(processor.tokenizer),
    )
    model.to('cuda')
    model.eval()

    # =========================
    # LOAD DATA
    # =========================

    root_path = os.path.dirname(os.path.dirname(__file__)) + '/'
    data_dir = "../../input/"

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
    cur_time = time.time()

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Valid", disable=True)

        for batch_data in pbar:
            audios = batch_data["audios"]
            batch = batch_data["batch_items"]

            input_values = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to('cuda')

            logits = model(input_values).logits

            logits_np = logits.to(torch.float32).cpu().numpy()

            transcription = []
            for b_idx in range(logits_np.shape[0]):
                text = decoder.decode(logits_np[b_idx], beam_width=500)
                transcription.append(text)

            for i, r in enumerate(transcription):
                item = batch[i]
                predictions[item["audio_filepath"]] = r
                total += 1

            # Обновляем информацию рядом с прогресс-баром
            pbar.set_postfix({
                "bs": len(batch),
                "dur": f"{batch[0]['duration']:.1f}s",
                "processed": total
            })

            del input_values, logits

    logger.success("Transcription complete in {:.2f} sec".format(
        time.time() - start_time
    ))

    # =========================
    # WRITE SUBMISSION
    # =========================

    submission_path = root_path + "/submission/submission.jsonl"
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

    try:
        cer_metric = evaluate.load("cer")
        wer_metric = evaluate.load("wer")
        cer = cer_metric.compute(references=all_real_text, predictions=all_pred_text)
        wer = wer_metric.compute(references=all_real_text, predictions=all_pred_text)
        print("WER: {:.4f} CER: {:.4f}".format(wer, cer))
    except Exception as e:
        cer = -1.0
        wer = -1.0
        print("Score error", str(e))

    shutil.copy(
        submission_path,
        os.path.dirname(submission_path) + '/submission_{}_{}_cer_{:.4f}_wer_{:.4f}.jsonl'.format(
            os.path.basename(manifest_path).split('_')[0], 'wav2vec-300m', cer, wer)
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return wer, cer


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    manifest_path = root_path + "input/valid.jsonl"
    # manifest_path = root_path + "input/test2.jsonl"
    # model_path = root_path + 'model_wav2vec/models/wav2vec2-300m-final-full/'
    model_path = 'ZFTurbo/facebook_wav2vec2_xls_r_300m_lingala_shona'
    valid_checkpoint(model_path, manifest_path)

import os

if __name__ == '__main__':
    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    os.environ['PYTHONWARNINGS'] = "ignore"
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_cache/'


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


def valid_checkpoint(model_path, manifest_path):
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
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)

            for i, r in enumerate(transcription):
                item = batch[i]
                predictions[item["audio_filepath"]] = r
                total += 1

            pbar.set_postfix({
                "bs": len(batch),
                "dur": f"{batch[0]['duration']:.1f}s",
                "processed": total
            })

            del input_values, attention_mask, logits, predicted_ids

    logger.success("Transcription complete in {:.2f} sec".format(
        time.time() - start_time
    ))

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
            os.path.basename(manifest_path).split('_')[0], 'mms-1b-all', cer, wer)
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cer, wer


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    manifest_path = root_path + "input/valid.jsonl"
    # manifest_path = root_path + "input/test2.jsonl"
    model_path = 'ZFTurbo/facebook_mms_1b_lingala_shona'
    valid_checkpoint(model_path, manifest_path)

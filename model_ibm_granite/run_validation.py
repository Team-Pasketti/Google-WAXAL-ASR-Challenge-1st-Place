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

from transformers import (
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechProcessor,
)

warnings.filterwarnings("ignore")

MAX_BATCH_SIZE = 10

ASR_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."


def get_dynamic_batches(items):
    total = len(items)
    i = 0
    memory_budget = 250
    attention_penalty = 0.0125

    while i < total:
        l = items[i]['duration']
        if items[i]['duration'] > 200:
            batch_size = 1
        else:
            batch_size = int(memory_budget / (l + attention_penalty * (l ** 2)))
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

    processor = GraniteSpeechProcessor.from_pretrained(model_path)
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16
    )
    model.to('cuda')
    model.eval()

    chat_prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": ASR_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )

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

    predictions = {}
    total = 0

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Valid")

        for batch_data in pbar:
            audios = batch_data["audios"]
            batch = batch_data["batch_items"]
            prompts = [chat_prompt] * len(audios)

            processed = processor(
                prompts,
                audios,
                return_tensors="pt",
                padding=True,
                padding_side="left"
            ).to('cuda')

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if 1:
                    out = model.generate(
                        **processed,
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=3,
                    )

            out = out[:, processed.input_ids.shape[1]:]

            transcription = processor.tokenizer.batch_decode(out, skip_special_tokens=True)

            for i, r in enumerate(transcription):
                item = batch[i]
                predictions[item["audio_filepath"]] = r.strip()
                total += 1

            pbar.set_postfix({
                "bs": len(batch),
                "dur": f"{batch[0]['duration']:.1f}s",
                "processed": total
            })

            del processed, out

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
            all_real_text.append(item['text'].strip())
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
            os.path.basename(manifest_path).split('_')[0], 'ibm_granite', cer, wer)
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return wer, cer


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    manifest_path = root_path + "input/valid.jsonl"
    # manifest_path = root_path + "input/test2.jsonl"
    model_path = root_path + 'model_ibm_granite/models/granite-speech/checkpoint-37340_wer_0.4243_cer_0.1357/'
    valid_checkpoint(model_path, manifest_path)

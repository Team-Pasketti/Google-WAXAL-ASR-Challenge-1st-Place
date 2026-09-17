import os
import sys

if __name__ == '__main__':
    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_use
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_cache/'

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import json
import shutil
import gc
import glob
import evaluate

from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import normalize_audios, normalize_language_name, validate_language, split_audio_into_chunks, AudioChunk
from qwen_asr.inference.utils import MAX_ASR_INPUT_SECONDS, MAX_FORCE_ALIGN_INPUT_SECONDS, SAMPLE_RATE
# from lib.score import score_jsonl

MULTI_COEFF = 1
MAX_BATCH_SIZE = 15
TOKENS_PER_SECOND = 20
MAX_NEW_TOKENS_OVERALL = MAX_ASR_INPUT_SECONDS * TOKENS_PER_SECOND

def get_dynamic_batches(items):
    total = len(items)
    i = 0

    while i < total:
        if items[i]['duration'] >= 200:
            batch_size = MULTI_COEFF * 1
        elif 150 < items[i]['duration'] <= 200:
            batch_size = MULTI_COEFF * 2
        elif 100 < items[i]['duration'] <= 150:
            batch_size = MULTI_COEFF * 4
        elif 40 < items[i]['duration'] <= 100:
            batch_size = MULTI_COEFF * 8
        elif 30 < items[i]['duration'] <= 40:
            batch_size = MULTI_COEFF * 15
        elif 20 < items[i]['duration'] <= 30:
            batch_size = MULTI_COEFF * 30
        elif 15 < items[i]['duration'] <= 20:
            batch_size = MULTI_COEFF * 45
        elif 10 < items[i]['duration'] <= 15:
            batch_size = MULTI_COEFF * 50
        elif 7 < items[i]['duration'] <= 10:
            batch_size = MULTI_COEFF * 64
        elif 5 < items[i]['duration'] <= 7:
            batch_size = MULTI_COEFF * 80
        elif 3 < items[i]['duration'] <= 5:
            batch_size = MULTI_COEFF * 96
        elif 2 < items[i]['duration'] <= 3:
            batch_size = MULTI_COEFF * 110
        elif 1 < items[i]['duration'] <= 2:
            batch_size = MULTI_COEFF * 128
        elif 0 <= items[i]['duration'] <= 1:
            batch_size = MULTI_COEFF * 256
        else:
            batch_size = int(MULTI_COEFF * 900 / items[i]['duration']) + 1
            batch_size = min(batch_size, MAX_BATCH_SIZE)

        batch_size = 5
        yield items[i:i + batch_size]
        i += batch_size


# 1. Создаем Dataset для параллельного чтения данных в фоне
class AudioBatchDataset(Dataset):
    def __init__(self, batches, data_dir, processor):
        self.batches = list(batches)
        self.data_dir = data_dir
        # self.model = model
        self.processor = processor

    def __len__(self):
        return len(self.batches)

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        """
        Build the string prompt for one request.

        If force_language is provided, "language X<asr_text>" is appended after the generation prompt
        to request text-only output.
        """
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            base = base + f"language {force_language}{'<asr_text>'}"
        return base

    def __getitem__(self, idx):
        batch = self.batches[idx]
        audios = []
        languages = []
        max_new_tokens = 0

        for item in batch:
            try:
                path = str(item["audio"])
            except:
                path = str(item["audio_filepath"])

            waveform, sr = librosa.load(path, sr=16000, mono=True)

            audios.append((waveform, sr))
            languages.append(None)
            max_new_tokens = max(max_new_tokens, int(item["duration"] * TOKENS_PER_SECOND))

        return_time_stamps = False
        context = ""
        language = languages
        # Normalization removed from Qwen code
        wavs = normalize_audios(audios)
        n = len(wavs)

        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")

        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(f"Batch size mismatch: audio={n}, language={len(langs_in)}")

        langs_norm: List[Optional[str]] = []
        for l in langs_in:
            if l is None or str(l).strip() == "":
                langs_norm.append(None)
            else:
                # ln = normalize_language_name(str(l))
                ln = str(l)
                validate_language(ln)
                langs_norm.append(ln)

        max_chunk_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS

        # chunk audios and record mapping
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=max_chunk_sec,
            )
            for j, (cwav, offset_sec) in enumerate(parts):
                chunks.append(AudioChunk(orig_index=i, chunk_index=j, wav=cwav, sr=SAMPLE_RATE, offset_sec=offset_sec))

        # run ASR on chunks
        chunk_ctx: List[str] = [ctxs[c.orig_index] for c in chunks]
        chunk_lang: List[Optional[str]] = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs: List[np.ndarray] = [c.wav for c in chunks]

        texts = [self._build_text_prompt(context=c, force_language=None) for c, fl in zip(chunk_ctx, chunk_lang)]
        batch_size1 = len(texts)

        all_inputs1 = []
        for ii in range(0, len(texts), batch_size1):
            sub_text = texts[ii: ii + batch_size1]
            sub_wavs = chunk_wavs[ii: ii + batch_size1]
            inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
            all_inputs1.append(inputs)

        return {
            "audios": audios,
            "chunks": chunks,
            "texts": texts,
            "chunk_lang": chunk_lang,
            "all_inputs": all_inputs1,
            "max_new_tokens": max_new_tokens,
            "batch_items": batch
        }


def identity_collate_fn(x):
    return x


def calc_metrics(
        target_file,
        preds_file,
        lang='en',
        verbose=True,
        normalize=False,
):
    from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    normalizer = None
    if normalize:
        if lang == 'en':
            normalizer = EnglishTextNormalizer()
        else:
            normalizer = BasicMultilingualTextNormalizer()

    lines = open(target_file, 'r', encoding="utf-8").readlines()
    target = [json.loads(line) for line in lines]
    lines = open(preds_file, 'r', encoding="utf-8").readlines()
    preds = [json.loads(line) for line in lines]

    if verbose:
        print('Target entries: {} Prediction entries: {}'.format(len(target), len(preds)))

    target_ids = set()
    target_dict = {}
    for t in target:
        try:
            target_ids |= set([t['audio_filepath']])
            target_dict[t['audio_filepath']] = t['text']
        except:
            target_ids |= set([t['audio']])
            target_dict[t['audio']] = t['text']


    preds_ids = set()
    preds_dict = {}
    for t in preds:
        try:
            preds_ids |= set([t['audio_filepath']])
            preds_dict[t['audio_filepath']] = t['text']
        except:
            preds_ids |= set([t['audio']])
            preds_dict[t['audio']] = t['text']

    check = target_ids - preds_ids
    if len(check) != 0:
        print("Some problem here. Some ids wasn't predicted! {}".format(len(check)))
        print(list(check)[:5])
        print(target_file, preds_file)

    references1 = []
    hypotheses1 = []
    for audio_id in list(target_ids):
        references1.append(target_dict[audio_id])
        hypotheses1.append(preds_dict[audio_id])

    if normalizer:
        references = [normalizer(ref) for ref in references1]
        hypotheses = [normalizer(pred) for pred in hypotheses1]
    else:
        references = references1
        hypotheses = hypotheses1

    references_fixed = []
    hypotheses_fixed = []
    for r, p in zip(references, hypotheses):
        if r == '':
            references_fixed.append('a')
            if p == '':
                hypotheses_fixed.append('a')
            else:
                hypotheses_fixed.append(p)
        else:
            references_fixed.append(r)
            hypotheses_fixed.append(p)

    references = references_fixed
    hypotheses = hypotheses_fixed

    score_wer = wer_metric.compute(
        references=references,
        predictions=hypotheses
    )
    score_wer = round(100 * score_wer, 6)

    score_cer = cer_metric.compute(
        references=references,
        predictions=hypotheses
    )
    score_cer = round(100 * score_cer, 6)

    return score_wer, score_cer


def valid_checkpoint(
    model_path,
    manifest_path="../../input/valid_qwen.jsonl",
    subm_name="submission.jsonl"
):
    start_time = time.time()
    logging.set_verbosity_error()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

    try:
        shutil.copy(root_path + 'qwen3_asr_training/qwen3-checkpoints/checkpoint-2000_1/chat_template.json', model_path)
        shutil.copy(root_path + 'qwen3_asr_training/qwen3-checkpoints/checkpoint-2000_1/preprocessor_config.json', model_path)
    except Exception as e:
        print('Error!', str(e))

    logger.info(f"Load model: {model_path}")
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=1024,
        max_new_tokens=-1,
    )

    logger.info("Model loaded")

    # =========================
    # LOAD DATA
    # =========================
    data_dir = root_path + "../../input/dataset/"
    logger.info("Data dir: {}", data_dir)
    submission_path = root_path + "/submission/" + subm_name

    with open(manifest_path, "r", encoding='utf8') as fr:
        items = [json.loads(line) for line in fr]

    items.sort(key=lambda x: x["duration"], reverse=True)

    logger.info(f"Processing {len(items)} utterances")

    batches = list(get_dynamic_batches(items))
    logger.info("Number of batches: {}".format(len(batches)))
    dataset = AudioBatchDataset(batches, data_dir, model.processor)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=identity_collate_fn
    )

    predictions = {}
    total = 0
    cur_time = time.time()

    with torch.inference_mode():
        for batch_data in dataloader:
            audios = batch_data["audios"]
            chunks = batch_data["chunks"]
            chunk_lang = batch_data["chunk_lang"]
            texts = batch_data["texts"]
            all_inputs = batch_data["all_inputs"]
            max_new_tokens = batch_data["max_new_tokens"]
            batch = batch_data["batch_items"]

            if max_new_tokens > MAX_NEW_TOKENS_OVERALL:
                max_new_tokens = MAX_NEW_TOKENS_OVERALL

            print(
                "Batch size:", len(batch),
                "Duration:", batch[0]["duration"],
                "Processed:", total,
                "Max tokens:", max_new_tokens,
                "Time: {:.2f} sec".format(time.time() - cur_time),
                "Total: {:.2f} sec".format(time.time() - start_time)
            )
            cur_time = time.time()

            if len(batch) >= 0:
                results = model.transcribe_reduced(
                    audio=audios,
                    chunks=chunks,
                    texts=texts,
                    chunk_lang=chunk_lang,
                    all_inputs=all_inputs,
                    max_new_tokens=max_new_tokens,

                    num_beams=5,
                    # do_sample=True,
                    # repetition_penalty=1.2,
                    # early_stopping=True,
                    # length_penalty=1.5,
                )
            else:
                results = model.transcribe_reduced(
                    audio=audios,
                    chunks=chunks,
                    texts=texts,
                    chunk_lang=chunk_lang,
                    all_inputs=all_inputs,
                    max_new_tokens=max_new_tokens,
                    # repetition_penalty=1.0,
                )

            for i, r in enumerate(results):
                item = batch[i]
                txt = r.text
                if '<asr_text>' in txt:
                    txt = txt.split('<asr_text>')[-1]
                try:
                    predictions[item["audio"]] = txt
                except:
                    predictions[item["audio_filepath"]] = txt
                total += 1

    logger.success("Transcription complete in {:.2f} sec".format(
        time.time() - start_time
    ))

    with open(submission_path, "w", encoding='utf8') as fw:
        for i in range(len(items)):
            item = items[i]
            try:
                item["text"] = predictions.get(item["audio"], "")
            except:
                item["text"] = predictions.get(item["audio_filepath"], "")
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    score_wer, score_cer = calc_metrics(manifest_path, submission_path, lang='xx', normalize=False)
    print("No normalize. Score WER: {:.4f} Score CER: {:.4f}".format(score_wer, score_cer))
    score_wer1, score_cer1 = calc_metrics(manifest_path, submission_path, lang='xx', normalize=True)
    print("Normalize. Score WER: {:.4f} Score CER: {:.4f}".format(score_wer1, score_cer1))

    shutil.copy(
        submission_path,
        os.path.dirname(submission_path) + '/submission_{}_{}_cer_{:.4f}_wer_{:.4f}.jsonl'.format(
            os.path.basename(manifest_path)[:-6], 'qwen', score_cer, score_wer
        )
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.success("Done.")
    return score_wer, score_cer


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    manifest_path = root_path + "input/valid.jsonl"
    # manifest_path = root_path + "input/test2.jsonl"
    # model_path = root_path + 'model_qwen3_asr/qwen3-checkpoints-last-v3/averaged_qwen/'
    model_path = 'ZFTurbo/Qwen3-ASR_lingala_shona'
    valid_checkpoint(
        model_path,
        manifest_path=manifest_path,
        subm_name=root_path + "submission/subm_qwen.jsonl",
    )

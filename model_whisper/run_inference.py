import os
import json
import gc
import shutil
import time
import evaluate
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
import librosa
from loguru import logger
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

warnings.filterwarnings("ignore")


def get_dynamic_batches(items):
    total = len(items)
    i = 0
    while i < total:
        if i < 100:
            batch_size = 8
        else:
            batch_size = 16
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


def inference_checkpoint(
    model_path,
    manifest_path,
):
    start_time = time.time()
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to('cuda')
    model.eval()

    root_path = os.path.dirname(os.path.dirname(__file__)) + '/'
    data_dir = "./input/"

    with open(manifest_path, "r", encoding='utf8') as fr:
        items = [json.loads(line) for line in fr]

    items.sort(key=lambda x: x["duration"], reverse=True)

    logger.info(f"Processing {len(items)} utterances")
    batches = list(get_dynamic_batches(items))
    logger.info(f"Number of batches: {len(batches)}")
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

    with (torch.inference_mode()):
        pbar = tqdm(dataloader, desc="Valid")
        for batch_data in pbar:
            audios = batch_data["audios"]
            batch = batch_data["batch_items"]

            inputs = processor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True
            )

            input_features = inputs.input_features.to('cuda', dtype=torch.float16)
            attention_mask = inputs.attention_mask.to('cuda')

            if 1:
                predicted_ids = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    # language="english",
                    task="transcribe",
                    max_new_tokens=428,
                    return_timestamps=False,

                    num_beams=5,
                    # no_repeat_ngram_size=3,
                    # condition_on_prev_tokens=False,
                )
            else:
                predicted_ids = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    # language="english",
                    task="transcribe",
                    max_new_tokens=428,
                    return_timestamps=False,
                )
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            for i, r in enumerate(transcription):
                item = batch[i]
                predictions[item["audio_filepath"]] = r
                total += 1

            pbar.set_postfix({"bs": len(batch), "processed": total})
            del input_features, predicted_ids

    logger.success(f"Transcription complete in {time.time() - start_time:.2f} sec")

    submission_path = root_path + "/submission/submission_whisper.jsonl"
    os.makedirs(root_path + "/submission/", exist_ok=True)

    all_real_text = []
    all_pred_text = []
    with open(submission_path, "w") as fw:
        for item in items:
            all_real_text.append(item['text'])
            phonetic = predictions.get(item["audio_filepath"], "")
            all_pred_text.append(phonetic)
            item["text"] = phonetic
            fw.write(json.dumps(item) + "\n")

    if 0:
        try:
            cer_metric = evaluate.load("cer")
            wer_metric = evaluate.load("wer")
            cer = cer_metric.compute(references=all_real_text, predictions=all_pred_text)
            wer = wer_metric.compute(references=all_real_text, predictions=all_pred_text)
            print(f"CER: {cer:.4f} WER: {wer:.4f} ")
            scr = cer
        except Exception as e:
            scr = -1.0
            cer = 0
            wer = 0
            print("Score error", str(e))

        shutil.copy(
            submission_path,
            os.path.dirname(submission_path) + '/submission_{}_whisper_cer_{:.4f}_wer_{:.4f}.jsonl'.format(
                os.path.basename(manifest_path).split('_')[0], cer, wer
            )
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    manifest_path = root_path + "input/test2.jsonl"
    # model_path = root_path + 'model_whisper/models/checkpoint-12000_cer_0.4560/'
    model_path = 'ZFTurbo/whisper_large_v3_lingala_shona'
    inference_checkpoint(model_path, manifest_path)

"""
checkpoint-12000_cer_0.4560 CER: 0.0846 WER: 0.3394 (chosen)
"""

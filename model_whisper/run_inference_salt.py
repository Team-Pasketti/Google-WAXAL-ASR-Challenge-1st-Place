import os
import json
import gc
import shutil
import sys
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


SAMPLE_RATE = 16000

LANGUAGE_TOKENS_WHISPER = {
    # Existing languges codes from Whisper
    "eng": 50259, "fra": 50265, "swa": 50318, "sna": 50324, "yor": 50325, "som": 50326,
    "afr": 50327, "amh": 50334, "mlg": 50349, "lin": 50353, "hau": 50354,
    # Overwrite unused language tokens
    "ach": 50357, "aka": 50356, "bam": 50355, "bem": 50352, "ber": 50351,
    "cgg": 50350, "dag": 50348, "dga": 50347, "ewe": 50346, "ful": 50345,
    "ibo": 50344, "kab": 50343, "kau": 50342, "kik": 50341, "kin": 50340,
    "kln": 50339, "koo": 50338, "kpo": 50337, "led": 50336, "lgg": 50335,
    "lth": 50333, "lug": 50332, "luo": 50331, "luy": 50330, "myx": 50329,
    "nbl": 50328, "nya": 50323, "nyn": 50322, "orm": 50321, "pcm": 50320,
    "ruc": 50319, "rwm": 50317, "sot": 50316, "teo": 50315, "tsn": 50314,
    "ttj": 50313, "wol": 50312, "xho": 50311, "xog": 50310, "zul": 50309
}

LANGUAGE_NAMES = {
    'Acholi': 'ach', 'Afrikaans': 'afr', 'Akan': 'aka', 'Amharic': 'amh', 'Ateso': 'teo',
    'Bambara': 'bam', 'Bemba': 'bem', 'Berber': 'ber', 'Chichewa': 'nya', 'Dagaare': 'dga',
    'Dagbani': 'dag', 'English': 'eng', 'Ewe': 'ewe', 'French': 'fra', 'Fulani': 'ful',
    'Hausa': 'hau', 'Igbo': 'ibo', 'Ikposo': 'kpo', 'Kabyle': 'kab', 'Kalenjin': 'kln',
    'Kanuri': 'kau', 'Kikuyu': 'kik', 'Kinyarwanda': 'kin', 'Kwamba': 'rwm', 'Lendu': 'led',
    'Lingala': 'lin', 'Luganda': 'lug', 'Lugbara': 'lgg', 'Luhya': 'luy', 'Lumasaba': 'myx',
    'Luo': 'luo', 'Lusoga': 'xog', 'Malagasy': 'mlg', 'Ndebele': 'nbl', 'Nigerian Pidgin': 'pcm',
    'Oromo': 'orm', 'Rukiga': 'cgg', 'Rukonjo': 'koo', 'Runyankole': 'nyn', 'Ruruuli': 'ruc',
    'Rutooro': 'ttj', 'Shona': 'sna', 'Somali': 'som', 'Sotho': 'sot', 'Swahili': 'swa',
    'Thur': 'lth', 'Tswana': 'tsn', 'Wolof': 'wol', 'Xhosa': 'xho', 'Yoruba': 'yor', 'Zulu': 'zul'
}



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
    token
):
    start_time = time.time()
    processor = WhisperProcessor.from_pretrained(
        # model_path,
        "openai/whisper-large-v3",
        token=token,
        # use_fast=False,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        token=token,
    ).to('cuda')
    model.eval()

    root_path = os.path.dirname(os.path.dirname(__file__)) + '/'
    data_dir = "../../input/"

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

    lang_tok = LANGUAGE_TOKENS_WHISPER[LANGUAGE_NAMES['Runyankole']]
    transcribe_tok = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_tok = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    forced_decoder_ids = [
        (1, lang_tok),
        (2, transcribe_tok),
        (3, notimestamps_tok),
    ]

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
                    forced_decoder_ids=forced_decoder_ids,
                    # language="english",
                    task="transcribe",
                    max_new_tokens=428,
                    return_timestamps=False,

                    num_beams=5,
                )
            else:
                predicted_ids = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    # language="english",
                    forced_decoder_ids=forced_decoder_ids,
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

    submission_path = root_path + "/submission/submission_whisper_salt.jsonl"
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
    if len(sys.argv) < 2:
        print("Obtain token at https://huggingface.co/settings/tokens and got access to weights at https://huggingface.co/Sunbird/asr-whisper-51-african-languages")
        exit()
    token = str(sys.argv[1]).strip()
    print("Token: {}".format(token))
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    manifest_path = root_path + "input/test2.jsonl"
    model_path = "Sunbird/asr-whisper-51-african-languages"
    inference_checkpoint(model_path, manifest_path, token)

"""
2 langs:
CER: 0.0788 WER: 0.3189
"""

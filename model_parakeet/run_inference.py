import os
import json
import gc
import time
import torch
import warnings
from loguru import logger

warnings.filterwarnings("ignore")


import nemo.collections.asr as nemo_asr
import copy
from omegaconf import open_dict

def inference_checkpoint(model_path, manifest_path):
    start_time = time.time()

    logger.info(f"Loading Parakeet model from {model_path}")
    model = nemo_asr.models.ASRModel.restore_from(model_path)
    model.eval()

    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy_batch"
        if "greedy" not in decoding_cfg:
            decoding_cfg.greedy = {}
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        decoding_cfg.greedy.loop_labels = True
    model.change_decoding_strategy(decoding_cfg)

    root_path = os.path.dirname(os.path.dirname(__file__)) + '/'

    with open(manifest_path, "r", encoding='utf8') as fr:
        items = [json.loads(line) for line in fr]

    audio_paths = []
    all_real_text = []

    for item in items:
        try:
            audio_name = item["audio_filepath"]
        except:
            audio_name = item["audio"]
        audio_paths.append(audio_name)
        all_real_text.append(item['text'])

    logger.info(f"Processing {len(items)} utterances (supports > 30 seconds duration)")

    with torch.inference_mode():
        hypotheses = model.transcribe(audio_paths, batch_size=2)
        transcriptions = [hyp.text for hyp in hypotheses]

    logger.success(f"Transcription complete in {time.time() - start_time:.2f} sec")

    submission_path = root_path + "submission/submission_parakeet.jsonl"
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    all_pred_text = []
    with open(submission_path, "w", encoding='utf8') as fw:
        for i, item in enumerate(items):
            pred_text = transcriptions[i]
            all_pred_text.append(pred_text)
            item["text"] = pred_text
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    manifest_path = root_path + "input/test2.jsonl"
    nemo_model_path = root_path + "model_parakeet/models/parakeet-step_step=4627-val_wer_val_wer=0.4007.nemo"
    inference_checkpoint(nemo_model_path, manifest_path)

"""
parakeet-step_step=4627-val_wer_val_wer=0.4007.nemo CER: 0.1168 WER: 0.4013
averaged_parakeet_7.nemo CER: 0.1129 WER: 0.3965
"""
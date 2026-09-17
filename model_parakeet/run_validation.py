import os
import sys

if __name__ == '__main__':
    gpu_use = "0"
    print(f'GPU use: {gpu_use}')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_use}"
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_cache/'

    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


import json
import gc
import shutil
import time
import evaluate
import torch
import warnings
from loguru import logger

warnings.filterwarnings("ignore")


import nemo.collections.asr as nemo_asr
import copy
from omegaconf import open_dict

def valid_parakeet_checkpoint(model_path, manifest_path):
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

    submission_path = root_path + "submission/submission.jsonl"
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    all_pred_text = []
    with open(submission_path, "w", encoding='utf8') as fw:
        for i, item in enumerate(items):
            pred_text = transcriptions[i]
            all_pred_text.append(pred_text)
            item["text"] = pred_text
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

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
        os.path.dirname(submission_path) + '/submission_{}_parakeet_cer_{:.4f}_wer_{:.4f}.jsonl'.format(
            os.path.basename(manifest_path).split('_')[0], cer, wer
        )
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return scr


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    val_manifest = root_path + "input/valid.jsonl"
    # val_manifest = root_directory + "input/test2.jsonl"
    # nemo_model_path = root_path + "/model_parakeet/parakeet_finetuned_step_full.nemo"
    nemo_model_path = "ZFTurbo/ibm_granite_speech_4.1_2b_lingala_shona"
    valid_parakeet_checkpoint(nemo_model_path, val_manifest)

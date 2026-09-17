import os

if __name__ == '__main__':
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_models_cache/'


import librosa
from tqdm import tqdm
import soundfile as sf
import json
import numpy as np
import random


def store_test_dataset_as_files_unique(dataset, out_dir1, name='test'):
    os.makedirs(out_dir1, exist_ok=True)
    out_dir = out_dir1 + name + '/'
    os.makedirs(out_dir, exist_ok=True)
    output_jsonl_file = os.path.join(out_dir, "markdown.jsonl")

    if os.path.isfile(output_jsonl_file):
        print("Dataset already created!")
        return output_jsonl_file
    out = open(output_jsonl_file, 'w', encoding='utf-8')
    print(dataset)
    print("Dataset length: {}".format(len(dataset[name])))
    for i in tqdm(range(len(dataset[name]))):
        # print(dataset[name][i])
        if dataset[name][i]["audio"]["path"] is None:
            orig_name = '{}.wav'.format(i)
        else:
            part = dataset[name][i]["audio"]["path"][:-4]
            part = part.replace(":", "")
            orig_name = os.path.basename(part + '_{}.wav'.format(i))
        audio = dataset[name][i]["audio"]["array"]
        sr = dataset[name][i]["audio"]["sampling_rate"]
        if 1:
            if sr != 16000:
                print('Resample: {} -> {}'.format(sr, 16000))
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # print(out_dir, orig_name, os.path.join(os.path.abspath(out_dir), orig_name))
        sf.write(os.path.join(out_dir, orig_name), audio, 16000, 'FLOAT')
        res = {
            'id': dataset[name][i]['id'],
            'audio': orig_name,
            'text': dataset[name][i]['transcription'],
            'duration': len(audio) / 16000,
        }
        out.write(json.dumps(res, ensure_ascii=False) + '\n')
    out.close()
    return output_jsonl_file


def download_dataset_WaxalNLP_clean():
    from datasets import load_dataset
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

    if 0:
        langs = ['lug_asr', 'lin_asr', 'sna_asr', 'ach_asr', 'aka_asr', 'amh_asr', 'dag_asr', 'dga_asr', 'ewe_asr',
        'ful_asr', 'kpo_asr', 'mas_asr', 'mlg_asr', 'nyn_asr', 'orm_asr', 'sid_asr', 'tir_asr', 'sog_asr', 'wal_asr']

    langs = [ 'lin_asr', 'sna_asr']

    for tp in langs:
        print(tp)
        asr_data = load_dataset("enesssssw/waxal-cleaned-16k", tp)

        # Access splits
        train = asr_data['train']
        val = asr_data['validation']
        test = asr_data['test']
        print(len(train), len(val), len(test))

        for split in ["train", "validation", "test"]:
            out_dir1 = root_path + 'input/dataset_cleaned/' + tp + '/'
            os.makedirs(out_dir1, exist_ok=True)
            store_test_dataset_as_files_unique(asr_data, out_dir1, split)


def gen_training_data():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    out = open(root_path + 'input/train.jsonl', 'w', encoding='utf-8')

    duration_arr = []
    len_arr = []

    folders_waxal = ['lin_asr', 'sna_asr']

    for tr_type in ['train', 'validation']:
        for tp in folders_waxal:
            current_path = root_path + 'input/dataset_cleaned/{}/{}/'.format(tp, tr_type)
            language = os.path.basename(os.path.dirname(current_path[:-1])).split('_')[0]
            full_path = os.path.abspath(current_path)
            full_path = full_path.replace("\\", "/") + '/'

            jsonl_dataset = current_path + 'markdown.jsonl'
            in1 = open(jsonl_dataset, 'r', encoding='utf-8')
            lines = in1.readlines()
            in1.close()
            for line in lines:
                arr = json.loads(line)
                if arr['duration'] > 50:
                    continue
                a = {
                    'audio_filepath': full_path + arr['audio'],
                    'text': arr['text'],
                    'duration': arr['duration'],
                }
                duration_arr.append(arr['duration'])
                len_arr.append(len(arr['text']))
                out.write(json.dumps(a, ensure_ascii=False) + '\n')

    out.close()


# In reality we use test with markdown
def gen_valid_data(limit_per_lang):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    out = open(root_path + 'input/valid.jsonl', 'w', encoding='utf-8')

    duration_arr = []
    len_arr = []

    folders_waxal = ['lin_asr', 'sna_asr']

    for tr_type in ['test']:
        for tp in folders_waxal:
            current_path = root_path + 'input/dataset_cleaned/{}/{}/'.format(tp, tr_type)
            language = os.path.basename(os.path.dirname(current_path[:-1])).split('_')[0]
            full_path = os.path.abspath(current_path)
            full_path = full_path.replace("\\", "/") + '/'

            jsonl_dataset = current_path + 'markdown.jsonl'
            in1 = open(jsonl_dataset, 'r', encoding='utf-8')
            lines = in1.readlines()
            in1.close()

            random.shuffle(lines)

            for line in lines[:limit_per_lang]:
                arr = json.loads(line)
                if arr['duration'] > 50:
                    continue
                a = {
                    'id': arr['id'],
                    'audio_filepath': full_path + arr['audio'],
                    'text': arr['text'],
                    'lang': language,
                    'duration': arr['duration'],
                }
                duration_arr.append(arr['duration'])
                len_arr.append(len(arr['text']))
                out.write(json.dumps(a, ensure_ascii=False) + '\n')

    out.close()


def convert_for_qwen():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    jsonl_dataset = root_path + "input/train.jsonl"
    out_dataset = jsonl_dataset[:-6] + '_qwen.jsonl'
    in1 = open(jsonl_dataset, 'r', encoding='utf-8')
    out = open(out_dataset, 'w', encoding='utf-8')
    lines = in1.readlines()
    in1.close()
    for line in lines:
        arr = json.loads(line)
        pth = arr['audio_filepath']

        a1 = {
            'text': arr['text'],
            'audio': pth,
        }
        out.write(json.dumps(a1, ensure_ascii=False) + '\n')
    out.close()

    jsonl_dataset = root_path + "input/valid.jsonl"
    out_dataset = jsonl_dataset[:-6] + '_qwen.jsonl'
    in1 = open(jsonl_dataset, 'r', encoding='utf-8')
    out = open(out_dataset, 'w', encoding='utf-8')
    lines = in1.readlines()
    in1.close()
    for line in lines:
        arr = json.loads(line)
        pth = arr['audio_filepath']

        a1 = {
            'text': arr['text'],
            'audio': pth,
        }
        out.write(json.dumps(a1, ensure_ascii=False) + '\n')
    out.close()


if __name__ == "__main__":
    download_dataset_WaxalNLP_clean()
    gen_training_data()
    gen_valid_data(100)
    convert_for_qwen()
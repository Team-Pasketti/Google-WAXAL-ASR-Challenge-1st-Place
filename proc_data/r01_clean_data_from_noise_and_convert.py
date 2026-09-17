import os
import sys
import librosa
from tqdm import tqdm
import soundfile as sf
import glob
import json
import time
import urllib.request


project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def clean_test_phase_2_with_vocal_model_new():
    from msst.inference import proc_folder

    code_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    in_audio_folder = code_path + 'input/newaudios/'
    out_audio_folder = code_path + 'input/newaudios_cleaned/'
    os.makedirs(out_audio_folder, exist_ok=True)
    files = glob.glob(in_audio_folder + '*.*')
    print(in_audio_folder, out_audio_folder, len(files))

    if not os.path.isfile(code_path + "msst/bs_6stem_fixed.ckpt"):
        urllib.request.urlretrieve(
            "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_fixed.ckpt?download=true",
            code_path + "msst/bs_6stem_fixed.ckpt"
        )

    # https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_fixed.ckpt?download=true
    # https://huggingface.co/noblebarkrr/mvsepless_resources/raw/main/bs_roformer/bs_6stem_fixed_config.yaml
    args = {
        'model_type': 'bs_roformer',
        "config_path": code_path + "msst/bs_6stem_fixed_config.yaml",
        "start_check_point": code_path + "msst/bs_6stem_fixed.ckpt",
        "store_dir": out_audio_folder,
        "input_folder": in_audio_folder,
        "device_ids": "0",
    }

    proc_folder(args)


def convert_back_to_mono_stage_2_new():
    code_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

    in_audio_folder = code_path + 'input/newaudios_cleaned/'
    out_audio_folder = code_path + 'input/newaudios_cleaned_mono/'
    os.makedirs(out_audio_folder, exist_ok=True)
    files = glob.glob(in_audio_folder + '*/vocals.*')
    for f in tqdm(files):
        vocals_path = out_audio_folder + os.path.basename(os.path.dirname(f)) + '.wav'
        cleaned, orig_sr = sf.read(f)
        cleaned = cleaned.mean(axis=-1)
        cleaned = librosa.resample(
            cleaned,
            orig_sr=orig_sr,
            target_sr=16000,
            res_type='soxr_vhq'
        )
        sf.write(vocals_path, cleaned, 16000, 'FLOAT')


def gen_test_phase_2_data_json():
    code_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

    out = open(code_path + 'input/test2.jsonl', 'w', encoding='utf-8')

    current_path = code_path + 'input/newaudios_cleaned_mono/'
    full_path = os.path.abspath(current_path) + '/'
    full_path = full_path.replace("\\", "/")

    files = glob.glob(current_path + '*.wav')
    for f in files:
        wav, sr = sf.read(f)
        if sr != 16000:
            print('Error!', sr)
            exit()
        duration = wav.shape[0] / sr
        a = {
            'id':  os.path.basename(f)[:-4],
            'audio_filepath': full_path + os.path.basename(f),
            'lang': '',
            'text': '',
            'duration': duration,
        }
        out.write(json.dumps(a, ensure_ascii=False) + '\n')

    out.close()


if __name__ == "__main__":
    start_time = time.time()
    clean_test_phase_2_with_vocal_model_new()
    convert_back_to_mono_stage_2_new()
    gen_test_phase_2_data_json()
    print("Time: {:.2f} sec".format(time.time() - start_time))

# Google WAXAL ASR Challenge (1st place solution)

1st place solution for [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) hosted by Zindi. Code was prepared by team [Roman Solovyev](https://github.com/ZFTurbo) and [Enes](https://github.com/enes3774).

The task was: Build an automatic speech recognition (ASR) system using the [WAXAL dataset](https://huggingface.co/datasets/google/WaxalNLP) that generalises to previously unseen speech data, focusing on three languages: Lingala, Shona, and Luganda.

Voice technologies have reshaped how people access digital services, but most African languages remain underserved because large-scale, high-quality speech datasets simply don't exist for them. WAXAL addresses this gap: developed over several years through a collaboration between Google Research and African academic and community organisations, it is one of the largest openly accessible speech resources for the continent, covering 27 languages spoken by more than 100 million people, with thousands of hours of natural speech alongside high-quality recordings aimed at both speech recognition and speech generation research.

Scoring uses a multi-metric approach - the weighted mean of Word Error Rate (WER) and Character Error Rate (CER), each weighted 0.5. The combination balances word-level accuracy against character-level robustness, which matters given how much spelling varies across African languages. 

## Solution documentation
* [Full solution documentation](docs/README.md)

## Requirements
* Python 3.10+
* All experiments were run on a single NVIDIA A6000 Blackwell 96 GB. However, the pipeline should work on GPUs with less memory - you just need to adjust the batch sizes accordingly.

## Inference
Quick start: just run `run_inference.sh` in a bash Linux shell. Below you can find the step-by-step inference:

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Unzip the test data into the `./input/newaudios/` folder parallel to this code. There must be 892 wav files. Also, put 
`Test_Phase2.csv` in the `./input` folder.

3. Run [MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training) to clean the speech data from noise. We use the [weights for bs_roformer](https://huggingface.co/noblebarkrr/mvsepless_resources/tree/main/bs_roformer). 
Cleaning is probably not needed, because the final test data wasn't noisy. After cleaning, convert the data to 16000 Hz mono.
```bash
python proc_data/r01_clean_data_from_noise_and_convert.py
```

4. Run inference for all the independent models:
```bash
python model_facebook_mms/run_inference.py
python model_ibm_granite/run_inference.py
python model_parakeet/run_inference.py
python model_qwen3_asr/run_inference.py
python model_wav2vec/run_inference.py
python model_whisper/run_inference.py
python model_whisper_salt_student/run_inference.py
```

5. Run inference for the 3rd-party model. You need to visit the [model page](https://huggingface.co/Sunbird/asr-whisper-51-african-languages) and request access, and also [generate a token on Hugging Face](https://huggingface.co/settings/tokens).
Then run:
```bash
python model_whisper/run_inference_salt.py "your hf token"
```

6. After steps 4 and 5, you will have 8 .jsonl files in the submission folder. Next, you need to convert them to the contest CSV format and ensemble the results.
```bash
python proc_data/r02_convert_to_csv.py
python proc_data/r03_ensemble_subms.py
```

7. Apply the orthography-aware consensus postprocessor. It preserves the
character median unless language-structured boundary rules or independently
verified member evidence support a correction:
```bash
python postprocess/run_postprocess.py \
  --ensemble submission/submission_final_8.csv \
  --out submission/submission_final_postprocessed.csv
```

The final submission file is
`submission/submission_final_postprocessed.csv`. See
[`postprocess/README.md`](postprocess/README.md) for the method, included member
hypotheses, audits.

## Training
Six models were trained on different architectures. To train the models you need to:

1. Prepare the data. There were only 2 languages in the final test set, so we trained models for these 2 languages only: lin (Lingala) and sna (Shona). The code should work similarly for any set of languages.
```bash
python proc_data/t01_download_and_preapre_data_for_training.py
```

2. Train all the models. Some of them need to be trained twice (with a frozen encoder + full training):
```bash
python model_facebook_mms/run_training.py adapter
python model_facebook_mms/run_training.py full
python model_ibm_granite/run_training.py
# Optional 2nd run
# python model_ibm_granite/run_training.py --model-name <last checkpoint path> --train-last-n-layers 60
python model_parakeet/run_training.py adapter
python model_parakeet/run_training.py full
python model_qwen3_asr/run_training.py
python model_wav2vec/run_training.py adapter
python model_wav2vec/run_training.py full
python model_whisper/run_training.py
```

You can choose the final checkpoints based on their validation scores.

3. Train SALT noisy-student (LoRA). Guide is available [in independent document](model_whisper_salt_student/README.md)
pip install -r requirements.txt
python proc_data/r01_clean_data_from_noise_and_convert.py
python model_facebook_mms/run_inference.py
python model_ibm_granite/run_inference.py
python model_parakeet/run_inference.py
python model_qwen3_asr/run_inference.py
python model_wav2vec/run_inference.py
python model_whisper/run_inference.py
python model_whisper_salt_student/run_inference.py
python model_whisper/run_inference_salt.py "your hf token"
python proc_data/r02_convert_to_csv.py
python proc_data/r03_ensemble_subms.py

# Adapter weights

Place the trained LoRA adapter here as `step00500/`:

```
models/step00500/adapter_config.json
models/step00500/adapter_model.safetensors
```

61 MB total. This is a PEFT adapter, not a full model — it is applied on top of
`Sunbird/asr-whisper-51-african-languages`, which is gated and needs an approved
HF token, the same requirement as Model 7.

`run_inference.py --adapter` and `run_validation.py --adapter` both accept
either this local path or an HF repo id, so the weights can be pulled from the
Hub instead of vendored.

Reproducing it from scratch takes the four stages in `../README.md`; the run
that produced this adapter was 2000 steps and the shipped checkpoint is step
500.

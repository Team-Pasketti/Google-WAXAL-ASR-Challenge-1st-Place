# export NUMBA_CUDA_USE_NVIDIA_BINDING=1 - can help in case of errors under WSL2
# python run_training.py

import os
import sys
import glob

if __name__ == '__main__':
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_cache/'


if __name__ == '__main__':
    TRAIN_MODE = str(sys.argv[1])

    project_root = os.path.abspath(os.path.dirname(__file__))

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import lightning.pytorch as pl
    from omegaconf import OmegaConf
    import nemo.collections.asr as nemo_asr

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

    TRAIN_MANIFEST = root_path + "input/train.jsonl"
    VAL_MANIFEST = root_path + "/input/valid.jsonl"

    if TRAIN_MODE == 'adapter':
        MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
        model = nemo_asr.models.ASRModel.from_pretrained(
            MODEL_NAME
        )
    elif TRAIN_MODE == 'full':
        MODEL_NAME = project_root + "/parakeet_finetuned_step_adapter.nemo"
        model = nemo_asr.models.ASRModel.restore_from(
            MODEL_NAME
        )
    else:
        print("Unknown train mode. Use adapter or full!")
        exit()

    import copy
    from omegaconf import open_dict

    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy_batch"
        if "greedy" not in decoding_cfg:
            decoding_cfg.greedy = {}
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        decoding_cfg.greedy.loop_labels = True

    model.change_decoding_strategy(decoding_cfg)

    def kill_cuda_graphs(decoding_obj):
        if decoding_obj is None:
            return
        impl = getattr(decoding_obj, "decoding", None)
        if impl is None:
            return
        if hasattr(impl, "use_cuda_graph_decoder"):
            impl.use_cuda_graph_decoder = False
        comp = getattr(impl, "decoding_computer", None)
        if comp is not None:
            if hasattr(comp, "disable_cuda_graphs"):
                comp.disable_cuda_graphs()
            if hasattr(comp, "cuda_graphs_mode"):
                comp.cuda_graphs_mode = None


    kill_cuda_graphs(getattr(model, "decoding", None))
    kill_cuda_graphs(getattr(getattr(model, "wer", None), "decoding", None))
    kill_cuda_graphs(getattr(getattr(model.joint, "wer", None), "decoding", None))

    print("cuda_graphs_mode =",
          getattr(model.decoding.decoding.decoding_computer, "cuda_graphs_mode", "n/a"))

    if TRAIN_MODE == 'adapter':
        for param in model.encoder.parameters():
            param.requires_grad = False

        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.joint.parameters():
            param.requires_grad = True

    sp_processor = model.tokenizer.tokenizer

    vocab_size = sp_processor.get_piece_size()
    print(f"\nВсего токенов в словаре: {vocab_size}")

    vocab = [sp_processor.id_to_piece(i) for i in range(vocab_size)]

    cfg = OmegaConf.to_container(model.cfg)
    print(cfg['train_ds'])

    cfg['train_ds']['manifest_filepath'] = TRAIN_MANIFEST
    cfg['train_ds']['use_lhotse'] = False
    if 'batch_duration' in cfg['train_ds']:
        del cfg['train_ds']['batch_duration']
    cfg['train_ds']['max_duration'] = 16.0
    cfg['train_ds']['min_duration'] = 0.1
    cfg['train_ds']['batch_size'] = 4
    cfg['train_ds']['text_field'] = 'text'
    cfg['train_ds']['num_workers'] = 8
    cfg['train_ds']['min_tps'] = 0
    cfg['train_ds']['max_tps'] = 1000
    cfg['train_ds']['pin_memory'] = True
    cfg['train_ds']['use_bucketing'] = True
    cfg['train_ds']['num_buckets'] = 6
    print(cfg['train_ds'])

    print(cfg['validation_ds'])
    cfg['validation_ds']['manifest_filepath'] = VAL_MANIFEST
    cfg['validation_ds']['use_lhotse'] = False
    cfg['validation_ds']['batch_size'] = 4
    cfg['validation_ds']['min_duration'] = 0.3
    cfg['validation_ds']['num_workers'] = 8
    print(cfg['validation_ds'])

    print(cfg['optim'])
    cfg['optim']['lr'] = 5e-5
    cfg['optim']['sched']['min_lr'] = 5e-6

    model.setup_training_data(cfg['train_ds'])
    model.setup_validation_data(cfg['validation_ds'])
    model.setup_optimization(cfg['optim'])

    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="parakeet_checkpoints/",
        filename="parakeet-step_{step}-val_wer_{val_wer:.4f}",
        save_on_train_epoch_end=False,
        save_top_k=15,
        monitor="val_wer",
        mode="min"
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=100,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,

        val_check_interval=1321,
        callbacks=[checkpoint_callback],

        enable_checkpointing=True,
        logger=False,
        log_every_n_steps=1000
    )

    # In case of failed run restore from last point
    ckpts = glob.glob("parakeet_checkpoints/*.ckpt")
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        print(f"Found checkpoint: {latest_ckpt}, resuming...")
        trainer.fit(model, ckpt_path=latest_ckpt)
    else:
        print("No checkpoints found, training from scratch...")
        trainer.fit(model)

    model.save_to("parakeet_finetuned_step_{}.nemo".format(TRAIN_MODE))

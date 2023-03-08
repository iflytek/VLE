import pytorch_lightning as pl
from pytorch_lightning.plugins import DeepSpeedPlugin
import torch
import json
import copy
import os
os.environ["NCCL_DEBUG"] = "INFO"
import argparse

from vqav2_datamodule import VQAv2DataModule
from vqav2_train_module import VLEForVQA_PL
from models.VLE.processing_vle import VLEProcessor


def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"], workers=True)
    vle_processor = VLEProcessor.from_pretrained(_config["model_dir"])
    if vle_processor.image_processor.size["shortest_edge"] != _config["image_size"]:
        vle_processor.image_processor.crop_size["height"] = _config["image_size"]
        vle_processor.image_processor.crop_size["width"] = _config["image_size"]
        vle_processor.image_processor.size["shortest_edge"] = _config["image_size"]
    dm = VQAv2DataModule(vle_processor, _config)

    model = VLEForVQA_PL(_config)
    exp_name = 'VQAv2'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=_config["save_top_k"],
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=_config["save_weights_only"]
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["model_dir"].split("/")[-1]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else -1

    if _config["use_deepspeed"]:
        deepspeed_config = _config["deepspeed_config"]
        if _config["precision"] != "16" and _config["precision"] != 16:
            deepspeed_config["fp16"]["enabled"] = False
        if _config["precision"] == "bf16":
            deepspeed_config["bf16"] = {"enabled": True}
        ds_plugin = DeepSpeedPlugin(config=deepspeed_config)
        strategy = ds_plugin
    else:
        strategy = "ddp"

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        strategy=strategy,
        benchmark=True,
        deterministic=False,
        max_epochs=_config["max_epoch"] if max_steps is -1 else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=True,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        num_sanity_val_steps=0,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        torch.cuda.empty_cache()
        trainer.test(ckpt_path="best", datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args for finetuning VLE on VQAv2.")
    parser.add_argument("--train_config_file", type=str, default="vqa_train_config.json", help="Config file for training.")
    args = parser.parse_args()
    train_config_file = args.train_config_file
    train_config = json.load(open(train_config_file, 'r'))

    main(train_config)
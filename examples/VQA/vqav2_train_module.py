import torch
import torch.nn as nn
import pytorch_lightning as pl
import json
import os
import glob

from torchmetrics.metric import Metric
from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from models.VLE import VLEForVQA
from models.VLE.modeling_vle import extend_position_embedding


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar.item()
        self.total += 1

    def compute(self):
        if self.total.item() == 0:
            return 0
        return self.scalar.item() / self.total.item()

class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum().item()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


class VLEForVQA_PL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = VLEForVQA.from_pretrained(config["model_dir"])

        if config["image_size"] != self.model.config.vision_config.image_size:
            patch_size = self.model.config.vision_config.patch_size
            position_length_after = (config["image_size"]//self.model.config.vision_config.patch_size)**2 + 1
            position_embed_dim = self.model.vle.vision_model.vision_model.embeddings.position_embedding.embedding_dim
            
            new_state_dict = extend_position_embedding(self.model.state_dict(), patch_size, config["image_size"])
            self.model.vle.vision_model.vision_model.embeddings.position_embedding = nn.Embedding(position_length_after, position_embed_dim)
            self.model.vle.vision_model.vision_model.embeddings.register_buffer("position_ids", torch.arange(position_length_after).expand((1, -1)))
            self.model.load_state_dict(new_state_dict)

        for split in ["train", "val"]:
            setattr(self, f"{split}_vqa_score", VQAScore())
            setattr(self, f"{split}_vqa_loss", Scalar())

    def forward(self, batch):
        ret = dict()
        model_inputs = batch["inputs"]
        model_outputs = self.model(**model_inputs,vqa_labels=batch["vqa_labels"], vqa_scores=batch["vqa_scores"], return_loss=True)
        vqa_logits = model_outputs["logits"]
        vqa_loss = model_outputs["loss"]

        vqa_targets = torch.zeros(vqa_logits.size()[:2]).to(vqa_logits.device)
        vqa_labels = batch["vqa_labels"]
        vqa_scores = batch["vqa_scores"]
        for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(_label, _score):
                vqa_targets[i, l] = s

        ret = {
            "vqa_loss": vqa_loss,
            "vqa_logits": vqa_logits,
            "vqa_targets": vqa_targets,
            "vqa_labels": vqa_labels,
            "vqa_scores": vqa_scores,
        }

        phase = "train" if self.training else "val"
        loss = getattr(self, f"{phase}_vqa_loss")(ret["vqa_loss"])
        score = getattr(self, f"{phase}_vqa_score")(
            ret["vqa_logits"], ret["vqa_targets"]
        )
        self.log(f"vqa/{phase}/loss", loss, batch_size=self.hparams.config["per_gpu_batchsize"])
        self.log(f"vqa/{phase}/score", score, batch_size=self.hparams.config["per_gpu_batchsize"])

        return ret

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        self.epoch_wrapup()

    def validation_step(self, batch, batch_idx):
        output = self(batch)

    def validation_epoch_end(self, outs):
        self.epoch_wrapup()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        ret = dict()
        # update vqa answer
        id2label = self.model.config.id2label
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2label[pred.item()] for pred in vqa_preds]
        questions = batch["text"]
        qids = batch["qid"]
        ret.update({"qids": qids, "questions": questions, "preds": vqa_preds})

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["model_dir"].split("/")[-1]
        save_dir = self.trainer.logger.log_dir
        rank = torch.distributed.get_rank()
        if rank != 0:
            save_dir_id = int(save_dir.split('_')[-1]) - 1
            save_dir = '_'.join(save_dir.split('_')[:-1]+[str(save_dir_id)])
        qids, preds = list(), list()
        for out in outs:
            qids += out["qids"]
            preds += out["preds"]

        rets = list()
        for qid, pred in zip(qids, preds):
            rets.append({"question_id": qid, "answer": pred})
        with open(os.path.join(save_dir, f"vqa_submit_{rank}.json"), "w") as fp:
            json.dump(rets, fp, indent=4)

        torch.distributed.barrier()

        if rank == 0:
            jsons = list()
            paths = list(glob.glob(os.path.join(save_dir,"vqa_submit_*.json")))
            for path in paths:
                with open(path, "r") as fp:
                    jsons += json.load(fp)
            os.makedirs(os.path.join(save_dir,"result"), exist_ok=True)
            with open(os.path.join(save_dir, f"result/vqa_submit_{model_name}.json"), "w") as fp:
                json.dump(jsons, fp, indent=4)

        torch.distributed.barrier()
        os.remove(os.path.join(save_dir, f"vqa_submit_{rank}.json"))

        self.epoch_wrapup(test_mode=True)

    def configure_optimizers(self):
        lr = self.hparams.config["learning_rate"]
        wd = self.hparams.config["weight_decay"]

        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
        ]
        head_names = ["vqa_classifier"]
        cross_modal_names = ['cross_modal']
        lr_mult_head = self.hparams.config["lr_mult_head"]
        lr_mult_cross_modal = self.hparams.config["lr_mult_cross_modal"]
        end_lr = self.hparams.config["end_lr"]
        decay_power = self.hparams.config["decay_power"]
        optim_type = self.hparams.config["optim_type"]
        all_grad_parameters = [(n,p) for n,p in self.named_parameters()]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in all_grad_parameters
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in all_grad_parameters
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in all_grad_parameters
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_head,
            },
            {
                "params": [
                    p
                    for n, p in all_grad_parameters
                    if any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_head,
            },
            {
                "params": [
                    p
                    for n, p in all_grad_parameters
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_cross_modal,
            },
            {
                "params": [
                    p
                    for n, p in all_grad_parameters
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_cross_modal,
            },
        ]

        if optim_type == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
            )
        elif optim_type == "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optim_type == "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

        if self.trainer.max_steps is -1:
            max_steps = (
                len(self.trainer.datamodule.train_dataloader())
                * self.trainer.max_epochs
                // self.trainer.accumulate_grad_batches
            )
        else:
            max_steps = self.trainer.max_steps

        warmup_steps = self.hparams.config["warmup_steps"]
        if isinstance(self.hparams.config["warmup_steps"], float):
            warmup_steps = int(max_steps * warmup_steps)

        if decay_power == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
            )
        else:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=end_lr,
                power=decay_power,
            )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )

    def epoch_wrapup(self, test_mode=False):
        phase = "train" if self.training else "val"
        loss_name = 'vqa'
        value = getattr(self, f"{phase}_{loss_name}_score").compute()
        self.log(f"{loss_name}/{phase}/score_epoch", value)
        getattr(self, f"{phase}_{loss_name}_score").reset()
        self.log(
            f"{loss_name}/{phase}/loss_epoch",
            getattr(self, f"{phase}_{loss_name}_loss").compute(),
        )
        getattr(self, f"{phase}_{loss_name}_loss").reset()

        self.log(f"{phase}/the_metric", value)
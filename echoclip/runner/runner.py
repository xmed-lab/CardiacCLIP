import json
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random
import cv2

from echoclip.models import MODELS
from echoclip.models.echoclip import EchoCLIP
from echoclip.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights
import sys
import numpy

from utils import compute_regression_metric
from sklearn.metrics import roc_auc_score


import wandb
import random
from scipy.stats import shapiro
from scipy.stats import lognorm
from scipy.stats import gamma
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

wandb.init(
    project="EchoCLIP",
)

logger = get_logger(__name__)


class Runner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(
            ce_loss=1.0,
            kl_loss=1.0,
            reg_loss=1.0
        ),
        ckpt_path="",
    ) -> None:
        super().__init__()
        self.module = MODELS.build(model_cfg)

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = nn.KLDivLoss(reduction="sum")
        self.loss_weights = loss_weights
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.num_ranks = 5
        self.register_buffer("rank_output_value_array", torch.arange(0, self.num_ranks).float(), persistent=False)


    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.forward_text_only()

    def run_step(self, batch, batch_idx):

        x, y = batch
        predictions, logits = self.module(x)

        losses = self.compute_losses(predictions, logits, y)
        loss = sum([weight * losses[k] for k, weight in self.loss_weights.items()])


        metrics_exp = self.compute_per_example_metrics(logits, predictions, y, "exp")
        metrics_max = self.compute_per_example_metrics(logits, predictions, y, "max")


        return {"loss": loss, **losses, **metrics_exp, **metrics_max}

    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        self.logging(outputs, "train", on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        return outputs


    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {pg["name"]: (pg["lr"], len(list(pg["params"]))) for pg in self.optimizers().param_groups}
        logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {param_group_lrs}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(f"{run_type}_{k}", v.mean(), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)


    def compute_losses(self, predictions, logits, y):       

        losses = {}
        y_label = self.num2label_fine(y)
        logits_prompt = logits

  
        # losses["rank_loss"] = torch.from_numpy(np.zeros(1)).to(y.device)

      
        losses["ce_loss"] = self.ce_loss_func(logits_prompt, y_label) 
        losses["kl_loss"] = self.compute_kl_loss(logits_prompt, y_label) 
        losses["reg_loss"] = (torch.abs(predictions-y)).mean() 


        # losses["dist_loss"] =  torch.from_numpy(np.zeros(1)).to(logits.device)
       

        wandb.log({"ce_loss": losses["ce_loss"], "kl_loss":losses["kl_loss"],"reg_loss":losses["reg_loss"]})
        return losses
    

   
    def num2label_fine(self, y):

        y = torch.floor(y).to(torch.int64).to(y.device)
        y_num = y.detach().cpu().numpy()
        y_label = np.zeros(len(y_num))
        
        for i in range(len(y_num)):
            if 0<= y_num[i] <=30:
                y_label[i] = 0
            elif 31<= y_num[i] <=45:
                y_label[i] = 1
            elif 46<= y_num[i] <=55:
                y_label[i] = 2
            elif 56<= y_num[i] <=70:
                y_label[i] = 3
            else:
                y_label[i] = 4

        y_label = torch.from_numpy(y_label).cuda().type(y.dtype)
        return y_label



    def compute_kl_loss(self, logits, y):
        y_t = F.one_hot(y, self.num_ranks).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        # .softmax函数将tensor的每个元素都缩放到(0,1)之间，且和为1.
        y_t_reduction = (y_t * 10.0).softmax(-1)

        y_t_reduction[y_t_row_ind <= 0] = 0

        logits_t = logits.t()
        kl_loss = self.kl_loss_func(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots

        return kl_loss


    
    def compute_per_example_metrics(self, logits, predictions, y, gather_type="exp"):

        dtype = logits.dtype

        probs = F.softmax(logits, -1)
        if gather_type == "exp":
        
            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y_label = torch.sum(probs * rank_output_value_array, dim=-1)


        elif gather_type == "max":
            predict_y_label = torch.argmax(probs, dim=-1).type(dtype)

        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        acc = (torch.round(predict_y_label) == y).type(logits.dtype)
        mae = (torch.abs(predictions-y)).mean()

        return {f"mae_{gather_type}_metric": mae,  "predict_y": predict_y_label, "GT": y}



    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_prompt_learner_weights=None,
        init_image_encoder_weights=None,
        init_text_encoder_weights=None,
    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return
        if init_prompt_learner_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.prompt_learner, init_prompt_learner_weights)
        if init_image_encoder_weights is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        if init_text_encoder_weights is not None:
            self._custom_logger.info("init_text_encoder_weights")
            load_pretrained_weights(self.module.text_encoder, init_text_encoder_weights)
        return

    def build_param_dict(
        self,
        lr_prompt_learner_context,
        lr_prompt_learner_ranks,
        lr_image_encoder,
        lr_text_encoder,
        lr_attention_layer,
        lr_logit_scale,
        staged_lr_image_encoder,
    ):
        param_dict_ls = []

        # if lr_prompt_learner_context > 0 and self.module.prompt_learner is not None:
        #     param_dict_ls.append(
        #         {
        #             "params": self.module.prompt_learner.context_embeds,
        #             "lr": lr_prompt_learner_context,
        #             "init_lr": lr_prompt_learner_context,
        #             "name": "lr_prompt_learner_context",
        #         }
        #     )
        # else:
        #     self._custom_logger.info("freeze_param(self.model.prompt_learner.context_embeds)")
        #     try:
        #         freeze_param(self.module.prompt_learner.context_embeds)
        #     except AttributeError:
        #         pass

        # if lr_prompt_learner_ranks > 0 and self.module.prompt_learner is not None:
        #     param_dict_ls.append(
        #         {
        #             "params": self.module.prompt_learner.rank_embeds,
        #             "lr": lr_prompt_learner_ranks,
        #             "init_lr": lr_prompt_learner_ranks,
        #             "name": "lr_prompt_learner_ranks",
        #         }
        #     )
        # else:
        #     self._custom_logger.info("freeze_param(self.model.prompt_learner.rank_embeds)")
        #     try:
        #         freeze_param(self.module.prompt_learner.rank_embeds)
        #     except AttributeError:
        #         pass
        
        # =============================================================================================
        
        # if lr_image_encoder > 0 and self.module.image_encoder is not None:
        #     if staged_lr_image_encoder is not None:
        #         self._custom_logger.info("staged_lr_image_encoder activated")
        #         image_encoder_param_groups = build_staged_lr_param_groups(
        #             model=self.module.image_encoder,
        #             lr=lr_image_encoder,
        #             **staged_lr_image_encoder,
        #         )
        #         param_dict_ls.extend(image_encoder_param_groups)
        #     else:
        #         param_dict_ls.append(
        #             {
        #                 "params": self.module.image_encoder.parameters(),
        #                 "lr": lr_image_encoder,
        #                 "init_lr": lr_image_encoder,
        #                 "name": "image_encoder",
        #             }
        #         )

        # else:
        #     self._custom_logger.info("freeze_param(self.model.image_encoder)")
        #     freeze_param(self.module.image_encoder)


# ---------------------------------------------------------------------------------------------
        
        if lr_image_encoder > 0 and self.module.image_clip_fea_encoder is not None:
            if staged_lr_image_encoder is not None:
                self._custom_logger.info("staged_lr_image_encoder activated")
                image_encoder_param_groups = build_staged_lr_param_groups(
                    model=self.module.image_clip_fea_encoder,
                    lr=lr_image_encoder,
                    **staged_lr_image_encoder,
                )
                param_dict_ls.extend(image_encoder_param_groups)
            else:
                param_dict_ls.append(
                    {
                        "params": self.module.image_clip_fea_encoder.parameters(),
                        "lr": lr_image_encoder,
                        "init_lr": lr_image_encoder,
                        "name": "image_clip_fea_encoder",
                    }
                )

        else:
            self._custom_logger.info("freeze_param(self.model.image_clip_fea_encoder)")
            freeze_param(self.module.image_clip_fea_encoder)



        if lr_image_encoder > 0 and self.module.head is not None:
            if staged_lr_image_encoder is not None:
                self._custom_logger.info("staged_lr_image_encoder activated")
                image_encoder_head_param_groups = build_staged_lr_param_groups(
                    model=self.module.head,
                    lr=lr_image_encoder,
                    **staged_lr_image_encoder,
                )
                param_dict_ls.extend(image_encoder_head_param_groups)
            else:
                param_dict_ls.append(
                    {
                        "params": self.module.head.parameters(),
                        "lr": lr_image_encoder,
                        "init_lr": lr_image_encoder,
                        "name": "head",
                    }
                )

        else:
            self._custom_logger.info("freeze_param(self.model.head)")
            freeze_param(self.module.head)



 
        if lr_attention_layer > 0 and self.module.abmil is not None:
            param_dict_ls.append(
                {
                    "params": self.module.abmil.parameters(),
                    "lr": lr_attention_layer,
                    "init_lr": lr_attention_layer,
                    "name": "abmil",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.abmil)")
            freeze_param(self.module.abmil)


        if lr_text_encoder > 0 and self.module.text_encoder is not None:
            param_dict_ls.append(
                {
                    "params": self.module.text_encoder.parameters(),
                    "lr": lr_text_encoder,
                    "init_lr": lr_text_encoder,
                    "name": "text_encoder",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.text_encoder)")
            freeze_param(self.module.text_encoder)



        if lr_logit_scale > 0 and self.module.logit_scale is not None:
            param_dict_ls.append(
                {
                    "params": self.module.logit_scale,
                    "lr": lr_logit_scale,
                    "init_lr": lr_logit_scale,
                    "name": "logit_scale",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.logit_scale)")
            freeze_param(self.module.logit_scale)


    
        return param_dict_ls




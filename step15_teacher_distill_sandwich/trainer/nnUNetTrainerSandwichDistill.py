#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step15_teacher_distill_sandwich/trainer/nnUNetTrainerSandwichDistill.py

SQ2 sandwich squeeze + feature-level distillation from a frozen UltraFedFM
teacher.  There is NO student network.

    L = L_sandwich (base + cautious + amplified, stock DC+CE)
      + w * (1 - cosine(aux_head(h4), PCA16(teacher(x)))),   w = 0.05

The teacher modulates the stage-4 features through a zero-initialised FiLM and
never receives gradients.  `aux_head` is a 1x1 conv used only by the loss; it
produces no segmentation output.

environment:
    US_DISTILL_W      weight of the cosine term (default 0.05)
    US_DISTILL_STATS  prior_stats.npz path (default ./prior_stats.npz)
    US_FM_CHECKPOINT  UltraFedFM checkpoint (required, not in this repo)

Registration: nnunetv2 2.6.2 resolves `-tr <Name>` by scanning the files in
<site-packages>/nnunetv2/training/nnUNetTrainer; scripts/70_install_trainer_shim.sh
installs the generated shim that re-exports this class.
"""
import os

import torch
import torch.nn.functional as F
from torch import autocast

from nnunetv2.utilities.helpers import dummy_context

from nnUNetTrainerSandwich import nnUNetTrainerSandwich
from sandwich_config import env_or


class nnUNetTrainerSandwichDistill(nnUNetTrainerSandwich):
    ARM_TAG = "EXP2_teacher_distill"
    ARM_DESC = ("stock sandwich + frozen UltraFedFM teacher FiLM at stage 4 + "
                "PCA-16 cosine distillation (no student network)")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        from sandwich_net import SandwichHeadUNet, tie_seg_heads
        from us_distill import DistillUNet

        if int(num_output_channels) != 2:
            raise ValueError("sandwich needs binary segmentation, got %s" % num_output_channels)

        base = get_network_from_plans(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, 6, deep_supervision=enable_deep_supervision)
        net = DistillUNet(base)
        tie_seg_heads(base, head_bias=float(env_or("STEP6_HEAD_BIAS", 1.0)))
        return SandwichHeadUNet(net, forward_mode="base")

    # ------------------------------------------------------------------ helpers
    def _distill_module(self):
        n = self.network
        for _ in range(3):
            if n is not None and n.__class__.__name__ == "DistillUNet":
                return n
            n = (getattr(n, "_orig_mod", None)
                 or getattr(n, "module", None)
                 or getattr(n, "net", None))
        return None

    # ------------------------------------------------------------------ train step
    def train_step(self, batch):
        """Stock nnU-Net step + the cosine distillation term.

        Copied from nnunetv2 2.6.2 nnUNetTrainer.train_step so the sandwich
        objective (computed by super().loss) is untouched; the only addition is
        the auxiliary feature-alignment term before backward.
        """
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' \
                else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

            w = float(env_or("US_DISTILL_W", 0.05))
            mod = self._distill_module()
            if w > 0 and mod is not None and mod._aux is not None:
                pred, tgt = mod._aux
                pred = F.normalize(pred.float(), dim=1)
                tgt = F.normalize(tgt.float(), dim=1)
                l = l + w * (1.0 - (pred * tgt).sum(1)).mean()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

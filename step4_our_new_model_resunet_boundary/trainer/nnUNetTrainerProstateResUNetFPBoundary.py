#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/nnUNetTrainerProstateResUNetFPBoundary.py

ARM B -- "everything on".

Network : `ResidualEncoderUNet` (nnU-Net's own updated recommendation, printed
          by this dataset's planner) with a light residual profile
          (1, 2, 2, 3, 3, 3, 3, 3) instead of the official ResEnc L profile
          (1, 3, 4, 6, 6, 6, 6, 6): on 71 training images the extra depth buys
          little while multiplying the epoch time.  The fingerprint's stages,
          features, strides and kernels are kept exactly.
Loss    : the same fingerprint-driven compound loss as arm A, PLUS the angular
          direction weighting from `angular_profile` (sectors where the
          fingerprint measured the boundary to be weakest get more weight --
          those are exactly the sectors where the failure analysis measured the
          highest FN rate, +75/+105 deg, rel edge strength 0.65-0.67).

Arm A isolates the loss; this arm adds the two other changes the redesign
argues for, so the comparison A vs B vs stock nnU-Net says whether they stack.
"""
import json
import os

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from fp_boundary_losses import FPCompoundLoss
from fp_config import load_fp_config

RESENC_UNET_CLASS = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"


def residual_blocks_per_stage(n_stages: int):
    """Light residual profile: (1, 2, 2, 3, 3, 3, 3, 3) for n_stages = 8."""
    profile = [1, 2, 2] + [3] * max(0, n_stages - 3)
    return tuple(profile[:n_stages])


class nnUNetTrainerProstateResUNetFPBoundary(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("STEP4_EPOCHS", 500))
        self.fp = load_fp_config()

    # ------------------------------------------------------------------ network
    # NOTE: signature is nnU-Net **2.6.2**'s
    #   (architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
    #    num_input_channels, num_output_channels, enable_deep_supervision)
    # which is what `nnUNetTrainer.initialize()` passes in the zk environment.
    # (nnU-Net >= 2.8 replaced the first three with plans_manager/configuration_manager.)
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        arch_kwargs = dict(arch_init_kwargs)
        n_stages = int(arch_kwargs["n_stages"])
        if "n_blocks_per_stage" in arch_kwargs:
            pass                                             # plans already residual
        else:
            arch_kwargs.pop("n_conv_per_stage", None)        # not accepted by ResEnc
            arch_kwargs["n_blocks_per_stage"] = residual_blocks_per_stage(n_stages)

        return get_network_from_plans(
            RESENC_UNET_CLASS,
            arch_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

    # --------------------------------------------------------------------- loss
    def _build_loss(self):
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

        base_loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
             'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            base_loss = DeepSupervisionWrapper(base_loss, weights)

        loss = FPCompoundLoss(base_loss, self.fp, is_full_resolution=True)
        loss.set_epoch(0)
        return loss

    # ------------------------------------------------------------------- logging
    def on_train_start(self):
        super().on_train_start()
        for line in self.fp.summary_lines():
            self.print_to_log_file(line)
        self.print_to_log_file(
            "[fp] arm B (ResidualEncoderUNet + direction weighting) | epochs %d | "
            "network %s | output_folder %s"
            % (self.num_epochs, type(self.network).__name__, self.output_folder))
        try:
            with open(os.path.join(self.output_folder, "fp_config_used.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"arm": "B_resenc_fp_loss_direction",
                           "num_epochs": self.num_epochs,
                           "fingerprint_source": self.fp.source,
                           "derived_loss_config": self.fp.loss,
                           "use_direction": bool(self.fp.use_direction),
                           "disabled": bool(self.fp.disabled),
                           "network_class": type(self.network).__name__},
                          fh, indent=2)
        except Exception as e:
            self.print_to_log_file("[fp] could not write fp_config_used.json: %s" % e)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if hasattr(self.loss, "set_epoch"):
            self.loss.set_epoch(self.current_epoch)
            if self.current_epoch < 3 or self.current_epoch % 25 == 0:
                d = self.fp.loss
                self.print_to_log_file(
                    "[fp] epoch %d | boundary multiplier %.4f | tol_band %d w %.3f | "
                    "sharp band %d beta %.1f w %.4f | length w %.4f | dir %s"
                    % (self.current_epoch, self.loss.multiplier,
                       d["tolerant_band"], d["weight_tolerant_bdice"],
                       d["bce_band"], d["bce_beta"], d["weight_boundary_bce"],
                       d["weight_perimeter"],
                       bool(self.fp.use_direction)))

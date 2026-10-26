#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/nnUNetTrainerProstateFPBoundary.py

ARM A -- "the loss, and nothing else".

Network : stock `PlainConvUNet`, i.e. the *same* architecture stock nnU-Net
          built for this dataset from its fingerprint (8 stages, 32..512,
          2 conv/stage, InstanceNorm+LeakyReLU).  Untouched on purpose: this arm
          isolates the effect of the new objective.
Loss    : stock Dice+CE (deep-supervision weighted) + the fingerprint-driven
          boundary terms (tolerant boundary Dice, sharp boundary BCE, length),
          see fp_boundary_losses.py and fp_config.py.

Everything else (fingerprint, plans, preprocessing, augmentation, optimiser,
PolyLR, TTA at inference, postprocessing) is stock nnU-Net, so a difference in
the final numbers is attributable to the loss.

500 epochs by default (STEP4_EPOCHS), which is also what PolyLR will be
scheduled over -- i.e. the schedule completes, unlike running a 1000-epoch
schedule for only half its length.

Registration: nnU-Net 2.8.1 looks trainers up *by class name*; this file lives
outside the nnunetv2 package, so `scripts/env_step4.sh` exports
`nnUNet_extTrainer=<step4>/trainer`.
"""
import json
import os

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from fp_boundary_losses import FPCompoundLoss
from fp_config import load_fp_config


class nnUNetTrainerProstateFPBoundary(nnUNetTrainer):
    # Overridable so that a subclass can keep the log and the
    # fp_config_used.json honest without copying on_train_start.
    ARM_TAG = "A_stock_arch_fp_loss"
    ARM_DESC = "arm A (stock PlainConvUNet)"


    # __init__ must mirror nnUNetTrainer's signature EXACTLY: nnUNetTrainer
    # builds self.my_init_kwargs from locals() of the *subclass* signature, so
    # an extra parameter raises KeyError.  Configuration is env-driven.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 500 epochs as requested; PolyLR is scheduled over exactly this number.
        self.num_epochs = int(os.environ.get("STEP4_EPOCHS", 500))
        self.fp = load_fp_config()

    # ------------------------------------------------------------------ network
    # Arm A keeps nnU-Net's own network: no override at all.

    # --------------------------------------------------------------------- loss
    def _build_loss(self):
        """Stock Dice+CE, rebuilt here so the deep-supervision weights are
        visible and identical to stock, wrapped with the fingerprint terms."""
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
            "[fp] %s | epochs %d | network %s | output_folder %s"
            % (self.ARM_DESC, self.num_epochs, type(self.network).__name__,
               self.output_folder))
        try:
            with open(os.path.join(self.output_folder, "fp_config_used.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"arm": self.ARM_TAG,
                           "arm_label": self.ARM_DESC,
                           "num_epochs": self.num_epochs,
                           "fingerprint_source": self.fp.source,
                           "derived_loss_config": self.fp.loss,
                           "use_direction": bool(self.fp.use_direction),
                           "disabled": bool(self.fp.disabled),
                           "network_class": type(self.network).__name__},
                          fh, indent=2)
        except Exception as e:                                   # never fatal
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

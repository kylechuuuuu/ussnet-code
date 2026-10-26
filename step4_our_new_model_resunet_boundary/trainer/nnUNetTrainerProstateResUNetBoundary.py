#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/nnUNetTrainerProstateResUNetBoundary.py

.. deprecated:: 2026-10-06
   SUPERSEDED.  This was the first design (ResEnc + a hand-tuned
   boundary-weighted BCE, `step4/DESIGN.md`).  It is not registered in
   `scripts/53_install_trainer_shim.sh` and is not launched by any
   current script; the experiments that replaced it are the
   fingerprint-loss arms A/B/C/D driven by `scripts/51_train_fp.sh`.
   It is kept only so the DESIGN.md-era code stays readable.  Its
   `build_network_architecture` has been ported to the nnU-Net 2.6.2
   signature so that it is runnable rather than instantly broken.

The redesigned model, implemented as an nnU-Net custom trainer so that it
inherits the whole fingerprint-driven pipeline (dataset fingerprint -> plans ->
preprocessing -> deep supervision -> sliding-window inference with mirroring TTA
-> evaluation).  Only the two things the step-3 analysis says are worth changing
are changed:

  1. ARCHITECTURE (encoder)
     stock nnU-Net 2d uses `PlainConvUNet`, i.e. 2 plain convolutions per stage
     and no residual connections.  nnU-Net's own updated recommendation (printed
     by the planner of this very dataset: "You are using the old nnU-Net default
     planner ... please consider using those instead!" -> resenc_presets.md) is
     `ResidualEncoderUNet`: a residual encoder with a stem, He initialisation and
     the last BatchNorm of every residual block zero-initialised.  We keep the
     fingerprint's stages / features / strides / kernels exactly (the dataset
     fingerprint decided those) and only change the block type and count, so the
     comparison against stock nnU-Net isolates the architecture, not the budget.

  2. LOSS
     stock Dice+CE plus a boundary-weighted BCE term on the full-resolution head,
     ramped in over `warmup` epochs (see trainer/boundary_losses.py).

Everything else - augmentation, optimiser, schedule, 1000-epoch recipe, TTA,
postprocessing - is deliberately left at nnU-Net's defaults so that any
difference in the final numbers is attributable to those two changes.

Registration: nnU-Net 2.8.1 has no module-level registration constant; trainers
are looked up *by class name*.  This file lives outside the nnunetv2 package, so
scripts/env_step4.sh exports

    nnUNet_extTrainer=<step4>/trainer

which nnU-Net searches as a fallback (utilities/find_objects.py).  The class name
below is the string passed to `nnUNetv2_train -tr`, and it is also what decides
the results folder name and what `nnUNetv2_predict` re-imports.
"""
import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from boundary_losses import BoundaryWeightedBCE, EdgeAwareCompoundLoss

# dynamic_network_architectures' residual encoder; same module nnU-Net's own
# ResEnc planners emit (`nnUNetPlannerResEncM/L/XL`).
RESENC_UNET_CLASS = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"


def residual_blocks_per_stage(n_stages: int):
    """Residual blocks per encoder stage.

    nnU-Net's ResEnc planners use (1, 3, 4, 6, 6, ...).  That is ~4x the
    convolution count of the default plain 2-conv-per-stage encoder; on this
    71-image dataset the extra depth buys little while multiplying the epoch
    time (which directly eats the budget for the second experiment), so we use a
    lighter profile of the same shape: shallow at high resolution, deeper once
    the feature maps are small.

        n_stages = 8  ->  (1, 2, 2, 3, 3, 3, 3, 3)
    """
    profile = [1, 2, 2] + [3] * max(0, n_stages - 3)
    return tuple(profile[:n_stages])


class nnUNetTrainerProstateResUNetBoundary(nnUNetTrainer):
    # NOTE: __init__ mirrors nnUNetTrainer's signature EXACTLY (5 positional
    # params).  nnUNetTrainer.__init__ builds self.my_init_kwargs by reading
    # locals() for every parameter name of the *subclass* signature, so adding a
    # parameter here raises KeyError.  Configuration therefore comes from the
    # environment.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = int(os.environ.get("STEP4_EPOCHS", 1000))
        self.boundary_weight = float(os.environ.get("STEP4_BOUNDARY_WEIGHT", 0.25))
        self.boundary_warmup_epochs = int(os.environ.get("STEP4_BOUNDARY_WARMUP", 100))
        self.boundary_band = int(os.environ.get("STEP4_BOUNDARY_BAND", 3))
        self.boundary_beta = float(os.environ.get("STEP4_BOUNDARY_BETA", 4.0))

    # ------------------------------------------------------------------ network
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Same fingerprint plans, residual encoder instead of plain conv encoder.

        NOTE (2026-10-06): written for the nnU-Net **2.6.2** call form, which is
        what this project's zk env ships -- `nnUNetTrainer.initialize` calls
        `build_network_architecture(class_name, init_kwargs, req_import,
        num_input_channels, num_output_channels, enable_deep_supervision)`.
        nnU-Net >= 2.8 passes `plans_manager, configuration_manager, ...`
        instead; a trainer written for one form raises TypeError on the other.
        Keep this signature byte-for-byte in sync with
        `nnUNetTrainerProstateResUNetFPBoundary.build_network_architecture`.
        """
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
        """nnU-Net's stock Dice+CE (including its deep-supervision weighting),
        wrapped so that a boundary term can be added on the full-resolution head.

        We rebuild the base loss instead of calling super()._build_loss() so that
        the deep-supervision weights and the MemoryEfficientSoftDiceLoss choice
        are visible here and identical to stock.
        """
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        import numpy as np

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

        loss = EdgeAwareCompoundLoss(
            base_loss=base_loss,
            boundary_loss=BoundaryWeightedBCE(beta=self.boundary_beta, band=self.boundary_band),
            boundary_weight=self.boundary_weight,
            warmup_epochs=self.boundary_warmup_epochs,
            is_full_resolution=True,
        )
        loss.set_epoch(0)
        return loss

    # ------------------------------------------------------------------- epochs
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if hasattr(self.loss, "set_epoch"):
            self.loss.set_epoch(self.current_epoch)
            if self.current_epoch < 3 or self.current_epoch % 50 == 0:
                self.print_to_log_file(
                    f"[step4] boundary term multiplier = {self.loss.boundary_loss.multiplier:.6f} "
                    f"(weight {self.boundary_weight}, warmup {self.boundary_warmup_epochs}, "
                    f"band {self.boundary_band}, beta {self.boundary_beta})"
                )

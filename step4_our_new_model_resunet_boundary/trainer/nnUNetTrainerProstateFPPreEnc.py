#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/nnUNetTrainerProstateFPPreEnc.py

ARM C -- "a different inductive bias": ImageNet-pretrained ResNet18 encoder,
the same fingerprint-driven loss as arms A and B.

Motivation (see DESIGN_NEXT.md and preenc_unet.py): arms A/B tie with stock
nnU-Net and every one of the eleven §5.x ceiling experiments failed because they
all reweighted or post-processed the *same* signal.  A pretrained encoder is the
only remaining lever that adds information the 71 training images cannot supply.

Three differences from arm A, all deliberate and all logged:

1. NETWORK.  `PreEncUNet` -- pretrained resnet trunk at strides 1/2/4/8,
   1x1 projections onto the *fingerprint's* feature budget
   (32/64/128/256/512/512/512/512), plain nnU-Net conv stages 4..7 and
   nnU-Net's own UNetDecoder.  The decoder is therefore exactly what stock
   nnU-Net would build; only the encoder changes.
2. FROZEN EARLY LAYERS.  STEP4_PREENC_FREEZE (default 2) freezes the stem and
   resnet layer1/layer2 -- the stride-1 and stride-2 layers.  They are the
   expensive ones, the most generic ones, and on 71 images the largest
   overfitting surface.  Frozen parameters run under `torch.no_grad()`, which
   is where most of the speed-up comes from (93 -> 55 s/epoch at patch
   640x1024, batch 4).
3. LOWER LR ON THE PRETRAINED PART.  layer3/layer4 and the projections get
   STEP4_PREENC_LR_SCALE (default 0.1) times the base LR, i.e. 1e-3 instead of
   1e-2.  nnU-Net's PolyLRScheduler applies one global rate to every group, so
   this trainer carries a small per-group PolyLR replacement.

Everything else (fingerprint loss, plans, preprocessing, augmentation,
500-epoch schedule, TTA at inference, postprocessing) is identical to arms A/B,
so a difference in the final numbers is attributable to the encoder.

Registration: identical to arms A/B -- the class is re-exported by the shim
installed by scripts/53_install_trainer_shim.sh (zk ships nnU-Net 2.6.2, which
has no nnUNet_extTrainer support).
"""
import json
import os

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnUNetTrainerProstateFPBoundary import nnUNetTrainerProstateFPBoundary
from preenc_unet import PreEncUNet


class _PolyLRPerGroup:
    """PolyLR that respects each parameter group's own starting lr.

    nnU-Net's PolyLRScheduler.step() overwrites every group with
    `initial_lr * (1 - step/max)**exponent`, using the *global* initial_lr, so a
    lower-rate group would be silently promoted to the global rate after the
    first step.  This keeps per-group bases.
    """

    def __init__(self, optimizer, num_epochs: int, exponent: float = 0.9):
        self.optimizer = optimizer
        self.num_epochs = int(num_epochs)
        self.exponent = float(exponent)
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.ctr = 0
        self._last_lr = list(self.base_lrs)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        f = (1.0 - float(current_step) / float(self.num_epochs)) ** self.exponent
        for g, b in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = b * f
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class nnUNetTrainerProstateFPPreEnc(nnUNetTrainerProstateFPBoundary):

    ARM_TAG = "C_preenc_resnet18_fp_loss"
    ARM_DESC = "arm C (ImageNet-pretrained ResNet18 encoder + fingerprint loss)"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.backbone = os.environ.get("STEP4_PREENC_BACKBONE", "resnet18")
        self.freeze_upto = int(os.environ.get("STEP4_PREENC_FREEZE", "2"))
        self.trunk_lr_scale = float(os.environ.get("STEP4_PREENC_LR_SCALE", "0.1"))
        # the label/tag follow the backbone, so arm D (resnet34) logs honestly
        self.arm_label = (os.environ.get("STEP4_PREENC_LABEL")
                          or ("arm C" if self.backbone == "resnet18" else "arm D"))
        self.arm_tag = "preenc_%s_fp_loss" % self.backbone
        self.arm_desc = ("%s (ImageNet-pretrained %s encoder + fingerprint loss)"
                         % (self.arm_label, self.backbone))

    # ------------------------------------------------------------------ network
    # nnU-Net 2.6.2 six-argument signature (identical to arm B's).
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        import pydoc
        ak = dict(arch_init_kwargs)

        def _loc(v):
            return pydoc.locate(v) if isinstance(v, str) else v

        net = PreEncUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            features_per_stage=ak["features_per_stage"],
            kernel_sizes=ak["kernel_sizes"],
            strides=ak["strides"],
            n_conv_per_stage=ak["n_conv_per_stage"],
            n_conv_per_stage_decoder=ak["n_conv_per_stage_decoder"],
            norm_op=_loc(ak.get("norm_op")),
            norm_op_kwargs=ak.get("norm_op_kwargs"),
            dropout_op=_loc(ak.get("dropout_op")),
            dropout_op_kwargs=ak.get("dropout_op_kwargs"),
            nonlin=_loc(ak.get("nonlin")),
            nonlin_kwargs=ak.get("nonlin_kwargs"),
            conv_bias=ak.get("conv_bias", True),
            deep_supervision=enable_deep_supervision,
            backbone=os.environ.get("STEP4_PREENC_BACKBONE", "resnet18"),
            pretrained=os.environ.get("STEP4_PREENC_PRETRAINED", "1") == "1",
            freeze_upto=int(os.environ.get("STEP4_PREENC_FREEZE", "2")),
        )
        # He-init the projections / plain stages / decoder; the pretrained trunk
        # must keep its ImageNet weights.
        PreEncUNet.initialize(net)
        return net

    # --------------------------------------------------------------- optimiser
    def configure_optimizers(self):
        enc = self.network.encoder
        low_ids = {id(p) for p in enc.low_lr_parameters()}
        trainable = [p for p in self.network.parameters() if p.requires_grad]
        high = [p for p in trainable if id(p) not in low_ids]
        low = [p for p in trainable if id(p) in low_ids]

        weight_decay = getattr(self, "weight_decay", 3e-5)
        groups = [{"params": high, "lr": self.initial_lr}]
        if low:
            groups.append({"params": low, "lr": self.initial_lr * self.trunk_lr_scale})
        self.optimizer = torch.optim.SGD(groups, lr=self.initial_lr, weight_decay=weight_decay,
                                         momentum=0.99, nesterov=True)
        if low:
            self.lr_scheduler = _PolyLRPerGroup(self.optimizer, self.num_epochs)
        else:
            from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
            self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.num_epochs)
        return self.optimizer, self.lr_scheduler

    # ----------------------------------------------------------------- logging
    def on_train_start(self):
        # grandparent: nnU-Net's own on_train_start (initialises if needed)
        nnUNetTrainer.on_train_start(self)
        for line in self.fp.summary_lines():
            self.print_to_log_file(line)
        enc = self.network.encoder
        self.print_to_log_file(
            "[fp] %s | epochs %d | network %s | %s"
            % (self.arm_desc, self.num_epochs, type(self.network).__name__, enc.load_report()))
        self.print_to_log_file(
            "[fp] %s config: backbone %s | freeze_upto %d | trunk lr scale %.3f | "
            "params %.2f M | trainable %.2f M"
            % (self.arm_label, self.backbone, self.freeze_upto, self.trunk_lr_scale,
               sum(p.numel() for p in self.network.parameters()) / 1e6,
               sum(p.numel() for p in self.network.parameters() if p.requires_grad) / 1e6))
        self.print_to_log_file(
            ("[fp] %s param groups: " % self.arm_label) + " | ".join(
                "n=%d lr=%.2e" % (len(g["params"]), g["lr"]) for g in self.optimizer.param_groups))
        try:
            with open(os.path.join(self.output_folder, "fp_config_used.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"arm": self.arm_tag,
                           "arm_label": self.arm_label,
                           "num_epochs": self.num_epochs,
                           "fingerprint_source": self.fp.source,
                           "derived_loss_config": self.fp.loss,
                           "use_direction": bool(self.fp.use_direction),
                           "disabled": bool(self.fp.disabled),
                           "network_class": type(self.network).__name__,
                           "backbone": self.backbone,
                           "freeze_upto": self.freeze_upto,
                           "trunk_lr_scale": self.trunk_lr_scale,
                           "pretrained_load": enc.load_report()},
                          fh, indent=2)
        except Exception as e:                                   # never fatal
            self.print_to_log_file("[fp] could not write fp_config_used.json: %s" % e)

    def on_train_epoch_start(self):
        nnUNetTrainer.on_train_epoch_start(self)
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

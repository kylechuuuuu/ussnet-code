#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/trainer/nnUNetTrainerSandwich.py

SQ2 "sandwich squeeze" trainer: ONE stock nnU-Net backbone produces three
versions of the segmentation -- base / cautious / amplified (sandwich_net.py),
trained with three stock objectives (sandwich_losses.py).  The final boundary
is DETERMINED at inference inside the interval by image-evidence arbitration
(tools/63_arbitrate.py), not by a head vote.

Everything else -- plans, preprocessing, augmentation, optimiser, PolyLR,
deep supervision, TTA, checkpoint selection (EMA pseudo-Dice of the BASE
version) -- is stock nnU-Net, so the numbers stay directly comparable with the
step-1 nnU-Net baseline, the step-4 arms and the step-5 squeeze arms.

Registration: zk's nnunetv2 2.6.2 resolves `-tr <Name>` only by scanning the
files inside <site-packages>/nnunetv2/training/nnUNetTrainer, so this class is
made visible through scripts/70_install_trainer_shim.sh (same mechanism as
steps 4 and 5).
"""
import json
import os

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from sandwich_config import env_or, load_sandwich_config
from sandwich_losses import SandwichLoss
from sandwich_net import SandwichHeadUNet, tie_seg_heads


class nnUNetTrainerSandwich(nnUNetTrainer):
    ARM_TAG = "SQ2_sandwich_squeeze"
    ARM_DESC = ("base + cautious(erode) + amplified(dilate), fixed error-scale "
                "radii (3/4 px), angular targets from the boundary fingerprint; "
                "nesting enforced at inference; boundary determined by "
                "interval-clamped image-evidence snapping")

    # __init__ must mirror nnUNetTrainer's signature EXACTLY (nnUNetTrainer
    # builds self.my_init_kwargs from locals() of the *subclass* signature).
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(env_or("STEP6_EPOCHS", 500))
        self.cfg = load_sandwich_config()
        self._last_diag = {}

    # ------------------------------------------------------------------ network
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Stock architecture, 3 binary versions instead of 1.

        This is also the function nnU-Net's predictor calls when it rebuilds
        the network from the checkpoint, so training and inference agree on
        the architecture by construction.
        """
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        if int(num_output_channels) != 2:
            raise ValueError(
                "the sandwich model is defined for binary segmentation "
                "(background + 1 foreground); got num_output_channels=%s" % num_output_channels)

        net = get_network_from_plans(architecture_class_name, arch_init_kwargs,
                                     arch_init_kwargs_req_import, num_input_channels,
                                     6, deep_supervision=enable_deep_supervision)
        tie_seg_heads(net, head_bias=float(env_or("STEP6_HEAD_BIAS", 1.0)))
        return SandwichHeadUNet(net, forward_mode="base")

    # --------------------------------------------------------------------- loss
    def _build_loss(self):
        """Three stock Dice+CE objectives (stock deep-supervision weights).

        The base version therefore sees *exactly the same objective* nnU-Net
        used for the baseline; the cautious/amplified objectives only add the
        two bounds of the squeeze interval.
        """
        def factory():
            return DC_and_CE_loss(
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
        else:
            weights = np.array([1.0])

        return SandwichLoss(factory, weights, self.cfg, device=self.device)

    # ------------------------------------------------------------------ logging
    @staticmethod
    def _unwrap_wrapper(net):
        """Find the SandwichHeadUNet behind torch.compile wrappers."""
        for _ in range(3):
            if isinstance(net, SandwichHeadUNet):
                return net
            nxt = getattr(net, "_orig_mod", None) or getattr(net, "module", None)
            if nxt is None:
                return None
            net = nxt
        return None

    def _set_forward_mode(self, mode):
        mod = self._unwrap_wrapper(self.network)
        if mod is None:
            return False
        mod.set_forward_mode(mode)
        return True

    def on_train_start(self):
        super().on_train_start()
        for line in self.cfg.summary_lines():
            self.print_to_log_file(line)
        self.print_to_log_file(
            "[sw] %s | epochs %d | head layout [0:2]=base [2:4]=cautious [4:6]=amplified | "
            "output_folder %s" % (self.ARM_DESC, self.num_epochs, self.output_folder))
        if not self._set_forward_mode("all"):
            raise RuntimeError("[sw] cannot switch the network to 6-channel training mode")
        try:
            with open(os.path.join(self.output_folder, "sandwich_config_used.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({
                    "arm": self.ARM_TAG,
                    "arm_label": self.ARM_DESC,
                    "num_epochs": self.num_epochs,
                    "fingerprint_source": self.cfg.source,
                    "r_cautious_px": self.cfg.r_c,
                    "r_amplified_px": self.cfg.r_a,
                    "w_cautious": self.cfg.w_c,
                    "w_amplified": self.cfg.w_a,
                    "angular": bool(self.cfg.angular),
                    "sector_scale": self.cfg.sector_scale,
                    "sector_deg": self.cfg.sector_deg,
                    "head_bias": self.cfg.head_bias,
                    "network_class": type(self.network).__name__,
                }, fh, indent=2)
        except Exception as e:                                        # never fatal
            self.print_to_log_file("[sw] could not write run metadata: %s" % e)

    def train_step(self, batch):
        out = super().train_step(batch)
        self._last_diag = dict(getattr(self.loss, "last_diag", {}) or {})
        return out

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        d = self._last_diag
        if d and (self.current_epoch < 3
                  or self.current_epoch % self.cfg.log_every == 0
                  or self.current_epoch == self.num_epochs - 1):
            self.print_to_log_file(
                "[sw] interval | cautious-outside-GT %.4f | GT-outside-amplified %.4f | "
                "nesting viol %.4f/%.4f | true boundary inside band %.3f | width %.2f px | "
                "dice(caut/target) %.3f dice(amp/target) %.3f | parts base %.3f c %.3f a %.3f"
                % (d.get("caut_leak", float('nan')), d.get("amp_miss", float('nan')),
                   d.get("viol_in", float('nan')), d.get("viol_out", float('nan')),
                   d.get("bd_contain", float('nan')), d.get("width_px", float('nan')),
                   d.get("dice_c", float('nan')), d.get("dice_a", float('nan')),
                   d.get("l_base", float('nan')), d.get("l_c", float('nan')),
                   d.get("l_a", float('nan'))))

    # ---------------------------------------------------- end-of-training validation
    def perform_actual_validation(self, save_probabilities: bool = False):
        """nnU-Net's end-of-training validation with the BASE version active.

        The cautious/amplified versions are exported by
        tools/62_predict_sandwich.py (one predictor pass per head); the stock
        validation path only needs the 2-channel base decision.
        """
        if not self._set_forward_mode("base"):
            raise RuntimeError("could not find the SandwichHeadUNet for validation")
        self.print_to_log_file("[sw] end-of-training validation uses the BASE version")
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            self._set_forward_mode("all")

    # ---------------------------------------------------------------- validation
    def validation_step(self, batch: dict) -> dict:
        """Identical to the stock step, except the pseudo-Dice is computed on the
        BASE version: with six output channels a plain argmax would silently
        score a 3-way segmentation instead of the boundary."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(target, (list, tuple)):
            target = target[0]

        logits = output[:, 0:2]                             # the base (基础) version
        fg = (torch.softmax(logits, dim=1)[:, 1:2] > 0.5).to(torch.float32)
        predicted_segmentation_onehot = torch.cat([1.0 - fg, fg], dim=1)

        axes = [0] + list(range(2, output.ndim))
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target,
                                        axes=axes, mask=None)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard,
                'fn_hard': fn_hard}

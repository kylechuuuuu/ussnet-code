#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/trainer/nnUNetTrainerProstateSqueeze.py

The squeeze-theorem (夹逼定理) trainer, ARM SQ-A (angular band, the default arm).

    network : stock nnU-Net architecture, ONLY the final segmentation conv is
              widened from 2 to 6 channels = three stock binary heads
              (main / inner / outer), see squeeze_net.SqueezeHeadUNet.
    loss    : per-head Dice+CE with the *stock* deep-supervision weighting
              (so the main head is trained by exactly the baseline objective)
              + ordering hinge  p_in <= p_main <= p_out
              + squeeze gap     (bounds collapse where the middle head is undecided)
              + the annealed band radius r(t): R0 -> R_min, which drives the two
              bound targets erode/dilate(GT, r(t)) onto GT itself.
              See squeeze_losses.py for the theorem-to-term mapping.
    everything else (plans, preprocessing, augmentation, optimiser, PolyLR,
    TTA at inference, postprocessing) is stock nnU-Net - identical to the
    step-1 baseline and to the step-4 arms, so the comparison is fair.

The trainer also LOGS the measurable claims of the squeeze whatever it does:
ordering violation rate, predicted interval width, whether the outer bound
still contains GT, whether the inner bound stays inside GT, and which fraction
of the true boundary lies inside the predicted interval.  Those are the curves
that show whether the pinch actually closes (tools/61_squeeze_diagnostics.py
recomputes them on the validation split with TTA).

Registration: zk's nnunetv2 2.6.2 resolves `-tr <Name>` only by scanning the
files inside <site-packages>/nnunetv2/training/nnUNetTrainer, so this class is
made visible through scripts/70_install_trainer_shim.sh (same mechanism as
step 4).
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

from squeeze_config import env_or, load_squeeze_config
from squeeze_losses import SqueezeLoss
from squeeze_net import SqueezeHeadUNet, tie_seg_heads


class nnUNetTrainerProstateSqueeze(nnUNetTrainer):
    ARM_TAG = "SQ-A_angular_squeeze"
    ARM_DESC = ("squeeze theorem (夹逼): 3 heads (main/inner/outer) + ordering hinge + "
                "squeeze gap + annealed band r(t): 6 -> 1 px, angular band from the fingerprint")
    DEFAULT_ANGULAR = True

    # __init__ must mirror nnUNetTrainer's signature EXACTLY (nnUNetTrainer builds
    # self.my_init_kwargs from locals() of the *subclass* signature).
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(env_or("STEP5_EPOCHS", 500))
        self.cfg = load_squeeze_config(angular_default=self.DEFAULT_ANGULAR)
        self._last_diag = {}

    # ------------------------------------------------------------------ network
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Stock architecture, 3 x binary heads instead of 1.

        This is also the function nnU-Net's predictor calls when it rebuilds the
        network from the checkpoint, so training and inference are guaranteed to
        agree on the architecture.
        """
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        if int(num_output_channels) != 2:
            raise ValueError(
                "the squeeze model is defined for binary segmentation "
                "(background + 1 foreground); got num_output_channels=%s" % num_output_channels)

        net = get_network_from_plans(architecture_class_name, arch_init_kwargs,
                                     arch_init_kwargs_req_import, num_input_channels,
                                     6, deep_supervision=enable_deep_supervision)
        tie_seg_heads(net, head_bias=float(env_or("STEP5_HEAD_BIAS", 1.0)))
        return SqueezeHeadUNet(net, fusion="train")

    # --------------------------------------------------------------------- loss
    def _build_loss(self):
        """Stock per-head Dice+CE (deep supervision weighted exactly like stock)
        + the squeeze terms.  The main head therefore sees *the same objective
        nnU-Net used for the baseline*; the extra machinery only adds the bounds.
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

        loss = SqueezeLoss(factory, weights, self.cfg, device=self.device)
        loss.set_epoch(0)
        return loss

    # ------------------------------------------------------------------ logging
    def on_train_start(self):
        super().on_train_start()
        for line in self.cfg.summary_lines():
            self.print_to_log_file(line)
        self.print_to_log_file(
            "[sq] %s | epochs %d | network %s | head layout "
            "[0:2]=main [2:4]=inner [4:6]=outer | output_folder %s"
            % (self.ARM_DESC, self.num_epochs, type(self.network).__name__, self.output_folder))
        try:
            with open(os.path.join(self.output_folder, "squeeze_config_used.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({
                    "arm": self.ARM_TAG,
                    "arm_label": self.ARM_DESC,
                    "num_epochs": self.num_epochs,
                    "fingerprint_source": self.cfg.source,
                    "R0": self.cfg.R0,
                    "R_min": self.cfg.R_min,
                    "warmup": self.cfg.warmup,
                    "schedule": self.cfg.schedule,
                    "lambda_order": self.cfg.lambda_order,
                    "lambda_gap": self.cfg.lambda_gap,
                    "w_in": self.cfg.w_in,
                    "w_out": self.cfg.w_out,
                    "angular": bool(self.cfg.angular),
                    "sector_scale": self.cfg.sector_scale,
                    "sector_deg": self.cfg.sector_deg,
                    "head_bias": self.cfg.head_bias,
                    "network_class": type(self.network).__name__,
                }, fh, indent=2)
            with open(os.path.join(self.output_folder, "squeeze_radius_schedule.csv"), "w",
                      encoding="utf-8") as fh:
                fh.write("epoch,r_px\n")
                for e, r in enumerate(self.cfg.radius_schedule()):
                    fh.write("%d,%.6f\n" % (e, r))
        except Exception as e:                                        # never fatal
            self.print_to_log_file("[sq] could not write run metadata: %s" % e)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if hasattr(self.loss, "set_epoch"):
            self.loss.set_epoch(self.current_epoch)
            if self.current_epoch < 3 or self.current_epoch % self.cfg.log_every == 0:
                self.print_to_log_file("[sq] epoch %d | band radius r = %.3f px"
                                       % (self.current_epoch, self.loss.r))

    def train_step(self, batch):
        out = super().train_step(batch)
        self._last_diag = dict(getattr(self.loss, "last_diag", {}) or {})
        return out

    # ---------------------------------------------------- end-of-training validation
    @staticmethod
    def _unwrap_wrapper(net):
        """Find the SqueezeHeadUNet behind torch.compile / DDP wrappers."""
        for _ in range(3):
            if isinstance(net, SqueezeHeadUNet):
                return net
            nxt = getattr(net, "_orig_mod", None) or getattr(net, "module", None)
            if nxt is None:
                return None
            net = nxt
        return None

    def _set_fusion(self, mode):
        mod = self._unwrap_wrapper(self.network)
        if mod is None:
            return False
        mod.set_fusion(mode)
        return True

    def perform_actual_validation(self, save_probabilities: bool = False):
        """nnU-Net's end-of-training validation, with the squeeze fusion active.

        The stock implementation hands `self.network` to nnUNetPredictor, whose
        sliding window preallocates `label_manager.num_segmentation_heads` (= 2)
        channels.  While training our network emits 6 (three heads), so the
        fusion must be switched on BEFORE the predictor runs - and switched back
        afterwards, so nothing else ever sees a fused network.  STOCK behaviour
        is otherwise untouched (same TTA, same sliding window, same metrics), so
        the `validation/` folder is directly comparable with the baseline's.
        """
        mode = str(env_or("STEP5_VAL_FUSION", "median"))
        if not self._set_fusion(mode):
            raise RuntimeError("could not find the SqueezeHeadUNet for validation")
        self.print_to_log_file(
            "[sq] end-of-training validation uses the fused decision rule '%s' "
            "(override with STEP5_VAL_FUSION in {main,median,inner,outer})" % mode)
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            self._set_fusion("train")

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        d = self._last_diag
        if d and (self.current_epoch < 3
                  or self.current_epoch % self.cfg.log_every == 0
                  or self.current_epoch == self.num_epochs - 1):
            self.print_to_log_file(
                "[sq] squeeze | viol_mag %.4f | viol>0.1 %.4f | gap %.4f (at boundary %.4f) | "
                "GT-outside-outer %.4f | inner-outside-GT %.4f | nesting %.4f | "
                "true boundary inside interval %.3f | parts main %.3f in %.3f out %.3f ord %.4f gap %.4f"
                % (d.get("viol_mag", float('nan')), d.get("viol_big", float('nan')),
                   d.get("gap", float('nan')),
                   d.get("gap_mid", float('nan')), d.get("outer_miss", float('nan')),
                   d.get("inner_leak", float('nan')), d.get("nesting", float('nan')),
                   d.get("bd_contain", float('nan')), d.get("l_main", float('nan')),
                   d.get("l_in", float('nan')), d.get("l_out", float('nan')),
                   d.get("l_ord", float('nan')), d.get("l_gap", float('nan'))))

    # ---------------------------------------------------------------- validation
    def validation_step(self, batch: dict) -> dict:
        """Identical to the stock step, except that the pseudo-Dice is computed on
        the MAIN (pinched) head: with three heads in one output tensor a plain
        argmax would silently score a 3-way segmentation instead of the boundary.
        """
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

        logits = output[:, 0:2]                             # main (夹逼中值) head
        fg = (torch.softmax(logits, dim=1)[:, 1:2] > 0.5).to(torch.float32)
        predicted_segmentation_onehot = torch.cat([1.0 - fg, fg], dim=1)

        axes = [0] + list(range(2, output.ndim))
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=None)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard,
                'fn_hard': fn_hard}

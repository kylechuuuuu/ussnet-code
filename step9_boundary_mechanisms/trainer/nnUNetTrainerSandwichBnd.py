#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/trainer/nnUNetTrainerSandwichBnd.py

MECHANISM 2 of step 9: SQ2res-w25 fine-tuned with a boundary-metric objective.

    L = L_base + 0.25 * L_cautious + 0.25 * L_amplified      (step 6, frozen)
        + w_band * (L_out + L_in)                            (step 9, NEW)

`boundary_losses.py` carries the full rationale.  In one line: Dice+CE is an
area functional and cannot see a 20-100 px long, 20 px deep anterior excursion
that costs 14 % of the boundary points; the band term is a symmetric,
distance-weighted tolerance-band penalty that does.

WHY FINE-TUNE INSTEAD OF TRAIN FROM SCRATCH

  * The 500-epoch from-scratch recipe costs 6.5 h on this GPU and, on this
    dataset, every 500-epoch arm landed within 0.3 bdice@1px of every other
    one (step 6 REPORT).  The question here is "does the OBJECTIVE change the
    boundary", not "does a longer schedule help", so the honest and cheap
    experiment is to change exactly one variable on top of a frozen reference.
  * `checkpoint_final.pth` of nnUNetTrainerSandwichResencW25 is the strongest
    base this project has (raw bdice@1px 35.33 / Dice 96.38 on val), so any
    movement is measured against the best available starting point.

WEIGHT LOADING

  nnU-Net's own `-pretrained_weights` route is NOT usable here: its
  `load_pretrained_weights` deliberately SKIPS every key containing
  `.seg_layers.` (compound_losses / load_pretrained_weights.py), which for this
  network is exactly the 1x1 head that produces the 6 channels -- i.e. the base
  head would restart from random noise.  We therefore load the checkpoint
  ourselves, in place, after `initialize()` has built the network and the
  optimizer (the optimizer holds parameter objects, not copies, so loading in
  place is correct).

ENVIRONMENT KNOBS

    STEP9_PRETRAINED  checkpoint to start from (default: the SQ2res-w25 final)
    STEP9_EPOCHS      fine-tune length             (default 150)
    STEP9_LR          peak LR of the PolyLR        (default 2e-4)
    STEP9_BAND_D      tolerance band width  D, px  (default 12)
    STEP9_BAND_W      weight of the band term      (default 1.0)
    STEP9_SKIP_BAND   1 = fine-tune with the stock objective only
                      (the control arm: isolates "LR schedule / more epochs"
                       from "the new objective")
"""
import os

import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.helpers import empty_cache

from nnUNetTrainerSandwichResencW25 import nnUNetTrainerSandwichResencW25
from sandwich_config import env_or
from boundary_losses import BoundaryBandLoss

USS_ROOT = "/media/72TB_disk/personal/zk/USSnet"

DEFAULT_PRETRAINED = os.path.join(
    USS_ROOT, "step6_sandwich_squeeze", "data", "nnUNet_results",
    "Dataset001_UltrasoundProstate",
    "nnUNetTrainerSandwichResencW25__nnUNetPlans__2d", "fold_0",
    "checkpoint_final.pth")


class nnUNetTrainerSandwichBnd(nnUNetTrainerSandwichResencW25):
    ARM_TAG = "SQ2res_w25_bnd_sandwich_squeeze"
    ARM_DESC = ("SQ2res-w25 fine-tuned with a symmetric distance-weighted "
                "tolerance-band boundary loss on the base head "
                "(boundary-metric objective, mechanism 2)")

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(env_or("STEP9_EPOCHS", 150))
        self.ft_lr = float(env_or("STEP9_LR", 2e-4))
        self.pretrained_path = env_or("STEP9_PRETRAINED", DEFAULT_PRETRAINED)
        self.band_D = float(env_or("STEP9_BAND_D", 12.0))
        self.band_w = float(env_or("STEP9_BAND_W", 1.0))
        self.skip_band = str(env_or("STEP9_SKIP_BAND", "0")) in ("1", "true", "True")
        self._loaded = 0

    # -------------------------------------------------------------- optimiser
    def configure_optimizers(self):
        """Fine-tune schedule: SGD at a small constant-then-decayed LR.

        nnU-Net's default (lr 1e-2, PolyLR over the full run) is the from-
        scratch schedule; restarting it for 150 epochs would blow the
        checkpoint apart in the first epoch.
        """
        optimizer = torch.optim.SGD(self.network.parameters(), self.ft_lr,
                                    weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.ft_lr, self.num_epochs)
        return optimizer, lr_scheduler

    # ------------------------------------------------------------------- loss
    def _build_loss(self):
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

        def factory():
            return DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
                 'do_bg': False, 'ddp': self.is_ddp},
                {},
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        from sandwich_losses import SandwichLoss
        loss = SandwichLoss(factory, weights, self.cfg, device=self.device)
        if not self.skip_band:
            loss.extra = BoundaryBandLoss(D=self.band_D, weight=self.band_w,
                                          scales=(0, 1, 2))
        self._ds_weights = weights
        return loss

    # ------------------------------------------------------------ initialise
    def initialize(self):
        super().initialize()
        self._load_pretrained_in_place()

    def _load_pretrained_in_place(self):
        path = self.pretrained_path
        if not path or not os.path.isfile(path):
            raise RuntimeError("STEP9_PRETRAINED not found: %r" % path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        src = ckpt.get("network_weights", ckpt)
        src = {str(k).replace("_orig_mod.", "").replace("module.", ""): v
               for k, v in src.items()}

        net = self.network
        if hasattr(net, "_orig_mod"):
            net = net._orig_mod
        dst = net.state_dict()
        loaded, skipped = 0, []
        for k, v in src.items():
            if k in dst and tuple(dst[k].shape) == tuple(v.shape):
                dst[k] = v
                loaded += 1
            else:
                skipped.append(k)
        if loaded == 0:
            raise RuntimeError("no parameter matched the pretrained checkpoint %r" % path)
        net.load_state_dict(dst)
        self._loaded = loaded
        print("[step9-bnd] loaded %d/%d tensors from %s"
              % (loaded, len(dst), path), flush=True)
        if skipped:
            print("[step9-bnd] skipped %d keys, e.g. %s" % (len(skipped), skipped[:5]),
                  flush=True)
        if self._loaded < len(dst):
            print("[step9-bnd] WARNING: %d tensors kept their random init"
                  % (len(dst) - loaded), flush=True)
        del ckpt, src
        empty_cache(self.device)

    # --------------------------------------------------------------- logging
    def _bnd_diag_line(self):
        d = dict(getattr(self, "_last_diag", {}) or {})
        if not d:
            return None
        keys = ["l_extra", "l_base"] + sorted(k for k in d if k.startswith("bnd_"))
        return "[step9-bnd] band | " + " | ".join(
            "%s %s" % (k, ("%.5f" % d[k]) if isinstance(d[k], float) else str(d[k]))
            for k in keys if k in d)

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if (self.current_epoch < 3 or self.current_epoch % self.cfg.log_every == 0
                or self.current_epoch == self.num_epochs - 1):
            line = self._bnd_diag_line()
            if line:
                self.print_to_log_file(line)

    # --------------------------------------------------------------- logging
    def on_train_start(self):
        super().on_train_start()
        msg = ("[step9-bnd] fine-tune: epochs=%d lr=%g D=%g w_band=%g "
               "skip_band=%s pretrained=%s"
               % (self.num_epochs, self.ft_lr, self.band_D, self.band_w,
                  self.skip_band, os.path.basename(self.pretrained_path)))
        self.print_to_log_file(msg)
        print(msg, flush=True)

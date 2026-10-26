#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/trainer/nnUNetTrainerSandwichW25.py

SQ2 with the bound-head loss weights reduced from 0.5/0.5 to 0.25/0.25.

Motivation (measured in this project):
    SQ2 (w = 0.5, 500 ep)   base raw bdice@1px 34.23
    fpA stock-arch 500 ep   base raw bdice@1px 34.61   (+ fingerprint loss)
    armB resenc   500 ep    base raw bdice@1px 35.17
    nnU-Net      1000 ep    base raw bdice@1px 35.23

At the SAME 500-epoch budget the stock architecture sits at ~34.5-34.6, so the
three-head objective costs the base head roughly 0.3-0.4 boundary-Dice points.
The most likely mechanism is the effective gradient scale: base + 0.5 + 0.5
doubles the objective the shared backbone is optimised with (nnU-Net's LR
schedule is tuned for the stock 1x scale).  0.25/0.25 brings the total to 1.5x
while keeping both bounds trained well enough for the interval (containment of
the true boundary was 0.678 on val / 1.000 on train for the 0.5/0.5 model; the
bound heads quickly saturate their targets, so less weight is affordable).

Use TWO class names on purpose: the output folder of an nnU-Net run is keyed by
the trainer name, and this lets a 1000-epoch run and a 500-epoch reference run
coexist without clobbering each other.

    nnUNetTrainerSandwichW25    eps from STEP6_EPOCHS (default 500)
    nnUNetTrainerSandwichW25s   hard-wired 500-epoch reference run
"""
import torch

from nnUNetTrainerSandwich import nnUNetTrainerSandwich
from sandwich_config import env_or


class nnUNetTrainerSandwichW25(nnUNetTrainerSandwich):
    ARM_TAG = "SQ2w25_sandwich_squeeze"
    ARM_DESC = ("SQ2 sandwich with bound-head loss weights 0.25/0.25 "
                "(base-preserving fix of the 3-head gradient dilution)")

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cfg.w_c = float(env_or("STEP6_WC", 0.25))
        self.cfg.w_a = float(env_or("STEP6_WA", 0.25))


class nnUNetTrainerSandwichW25s(nnUNetTrainerSandwichW25):
    """500-epoch reference twin of W25 (distinct output folder on purpose)."""
    ARM_TAG = "SQ2w25s_sandwich_squeeze"
    ARM_DESC = ("SQ2w25 at exactly 500 epochs (same-schedule A/B reference "
                "against the original w=0.5 SQ2 run)")

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(env_or("STEP6_EPOCHS_W25S", 500))

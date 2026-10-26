#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/trainer/nnUNetTrainerProstateStock200.py

CONTROL ARM -- the *stock* nnU-Net recipe, nothing added, only the schedule
length matched to step 5's expedited run (STEP5_EPOCHS, default 200).

Why it exists
-------------
The squeeze arm lands at Dice 96.02 / boundary Dice@1px 32.77 while the step-1
nnU-Net baseline (1000 epochs, same data, same fold, same everything) sits at
96.11 / 35.16.  Two very different explanations fit that:

    H1 "the squeeze machinery costs boundary accuracy"   (a method effect)
    H2 "200 epochs is simply less training than 1000"    (a budget effect)

This control decides between them: identical architecture, identical loss,
identical augmentation, identical preprocessing, identical split - only 200
epochs.  Whatever boundary Dice@1px it reaches IS the budget-matched reference
the squeeze arm has to be compared against.  It is registered through the same
shim and predicted with the same nnU-Net pipeline, then scored by the same
evaluator, so the comparison is apples to apples.
"""
import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from squeeze_config import env_or


class nnUNetTrainerProstateStock200(nnUNetTrainer):
    ARM_TAG = "CTRL-stock200"
    ARM_DESC = ("control: STOCK nnU-Net (2-channel head, stock Dice+CE), "
                "only the schedule is matched to the squeezed run")

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(env_or("STEP5_EPOCHS", 200))

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file("[ctrl] STOCK nnU-Net control run | epochs %d | network %s"
                               % (self.num_epochs, type(self.network).__name__))

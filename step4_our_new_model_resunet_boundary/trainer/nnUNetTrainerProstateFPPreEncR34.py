#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step4_our_new_model_resunet_boundary/trainer/nnUNetTrainerProstateFPPreEncR34.py

ARM D -- identical network and training recipe to arm C, but a DISTINCT trainer
class name.

Why a subclass that adds nothing
--------------------------------
nnU-Net derives the results folder purely from the trainer CLASS NAME:

    <nnUNet_results>/Dataset001_UltrasoundProstate/
        nnUNetTrainerProstateFPPreEnc__nnUNetPlans__2d/fold_0/

Two concurrently running arms that share a trainer class therefore share one
output folder and would overwrite each other's checkpoints and training logs.
Arm D exists to test a second pretrained trunk (resnet34 instead of resnet18)
AT THE SAME TIME on the other authorised GPU, so it needs its own name -- and
that is the only thing this file changes.

The trunk itself is chosen by the environment variable STEP4_PREENC_BACKBONE,
which scripts/51_train_fp.sh sets per arm (resnet18 for C, resnet34 for D).
No __init__ override: nnU-Net reconstructs `my_init_kwargs` from the signature's
locals, so the inherited signature must stay untouched.
"""
from nnUNetTrainerProstateFPPreEnc import nnUNetTrainerProstateFPPreEnc


class nnUNetTrainerProstateFPPreEncR34(nnUNetTrainerProstateFPPreEnc):
    """Arm D: pretrained resnet34 trunk + fingerprint loss (see module docstring)."""

    ARM_TAG = "D_preenc_resnet34_fp_loss"
    ARM_DESC = "arm D (ImageNet-pretrained ResNet34 encoder + fingerprint loss)"

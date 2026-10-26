#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/trainer/nnUNetTrainerProstateSqueezeIso.py

ARM SQ-I -- the isotropic ablation of the squeeze model.

Identical in every respect to arm SQ-A (nnUNetTrainerProstateSqueeze) except
that the band half-width is the SAME in all directions:

    r(theta) = r(t)          instead of      r(theta) = r(t) * s(theta)

where s(theta) is the fingerprint's angular edge-strength profile (wider band
where the ultrasound evidence is weaker, i.e. around +75/+105 deg for this
dataset).  Everything else - architecture, three heads, ordering hinge, squeeze
gap, radius curriculum - is untouched, so the difference between SQ-A and SQ-I
isolates whether letting the interval open up in the directions where the image
does not support a sharp edge is worth anything.

The class exists (instead of an env flag on SQ-A) because nnU-Net names the
results folder after the trainer class, and the two arms must not overwrite each
other.
"""
import os

from nnUNetTrainerProstateSqueeze import nnUNetTrainerProstateSqueeze


class nnUNetTrainerProstateSqueezeIso(nnUNetTrainerProstateSqueeze):
    ARM_TAG = "SQ-I_isotropic_squeeze"
    ARM_DESC = ("squeeze theorem (夹逼): 3 heads + ordering hinge + squeeze gap + "
                "annealed band r(t): 6 -> 1 px, ISOTROPIC band (ablation of the angular half)")
    DEFAULT_ANGULAR = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device=None):
        import torch
        super().__init__(plans, configuration, fold, dataset_json,
                         device if device is not None else torch.device('cuda'))
        # STEP5_ANGULAR, if exported, still wins (it is read inside load_squeeze_config);
        # this class only changes the *default*.
        from squeeze_config import env_or
        assert int(env_or("STEP5_ANGULAR", 0)) == 0, (
            "arm SQ-I is the isotropic arm but STEP5_ANGULAR=1 was exported; "
            "unset it or use nnUNetTrainerProstateSqueeze")

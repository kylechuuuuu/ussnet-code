#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/trainer/nnUNetTrainerSandwichResencW25.py

SQ2w25 on top of the STRONGEST base architecture this project has measured:
the light ResidualEncoderUNet profile of step-4 arm B.

Measured same-split base qualities (raw bdice@1px, val, TTA):
    fpA  stock arch + fingerprint loss  500 ep : 34.61
    armB ResidualEncoderUNet + fp loss  500 ep : 35.17   <- this architecture
    nnU-Net stock                      1000 ep : 35.23
    SQ2  stock arch, 3 heads, w=0.5     500 ep : 34.23

armB showed the residual encoder buys roughly +0.55 boundary-Dice at the same
500-epoch budget; SQ2 showed the three-headed objective costs the base head
roughly 0.3-0.4 (fpA per-budget reference), which the w=0.25 variant is
designed to recover.  Combining the two is therefore the most promising
configuration for the final target: base >= 35.0 so that the interval-clamped
image-evidence arbitration can climb past the strongest reference points
(nnU-Net 1000 ep + free snap: 36.59; armB + free snap: 36.67).

The network builder mirrors step-4 arm B's build_network_architecture exactly
(same class, same plan kwargs, same light residual profile), then widens the
segmentation head to 6 channels and ties the three versions.  The objective is
the sandwich objective of this step (three stock DC+CE terms, w = 0.25/0.25),
NOT step 4's fingerprint loss: the comparison here is about the boundary
determination stack, one variable at a time.
"""
import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from nnUNetTrainerSandwichW25 import nnUNetTrainerSandwichW25
from sandwich_config import env_or
from sandwich_net import SandwichHeadUNet, tie_seg_heads

RESENC_UNET_CLASS = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"


def residual_blocks_per_stage(n_stages: int):
    """Light residual profile (1, 2, 2, 3, 3, 3, 3, 3) -- step-4 arm B's choice."""
    profile = [1, 2, 2] + [3] * max(0, n_stages - 3)
    return tuple(profile[:n_stages])


class nnUNetTrainerSandwichResencW25(nnUNetTrainerSandwichW25):
    ARM_TAG = "SQ2res_w25_sandwich_squeeze"
    ARM_DESC = ("SQ2w25 on the ResidualEncoderUNet profile of step-4 arm B "
                "(strongest base architecture measured in this project)")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        if int(num_output_channels) != 2:
            raise ValueError("binary segmentation expected; got %s" % num_output_channels)

        arch_kwargs = dict(arch_init_kwargs)
        n_stages = int(arch_kwargs["n_stages"])
        if "n_blocks_per_stage" not in arch_kwargs:
            arch_kwargs.pop("n_conv_per_stage", None)
            arch_kwargs["n_blocks_per_stage"] = residual_blocks_per_stage(n_stages)

        net = get_network_from_plans(
            RESENC_UNET_CLASS,
            arch_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            6,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        tie_seg_heads(net, head_bias=float(env_or("STEP6_HEAD_BIAS", 1.0)))
        return SandwichHeadUNet(net, forward_mode="base")

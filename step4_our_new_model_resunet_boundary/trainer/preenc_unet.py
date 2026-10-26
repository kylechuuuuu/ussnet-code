#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/preenc_unet.py

ARM C -- "a different inductive bias": a U-Net whose encoder is an
ImageNet-pretrained ResNet trunk.

Why
---
Arms A and B (fingerprint loss on stock / residual PlainConvUNet) tie with
stock nnU-Net (96.11 / 96.16 vs 96.11 Dice) and the eleven ceiling experiments
in DESIGN_NEXT.md §5.1-§5.11 all failed.  Every one of those experiments
*reweighted or post-processed the same signal*; none of them added information.
A pretrained encoder is the one remaining lever that does: its filters were
trained on 1.2M natural images and encode generic edge/blob/texture detectors
that 71 ultrasound images cannot learn from scratch.

What this network is
--------------------
The nnU-Net plan for this dataset is 8 stages, features
[32,64,128,256,512,512,512,512], strides [1,2,2,2,2,2,2,2], 3x3 kernels,
InstanceNorm2d + LeakyReLU, patch 640x1024.

    stage 0  stride  1   resnet stem (7x7 conv, stride 1) + layer1  -> proj  32 ch
    stage 1  stride  2   resnet layer2                              -> proj  64 ch
    stage 2  stride  4   resnet layer3                              -> proj 128 ch
    stage 3  stride  8   resnet layer4                              -> proj 256 ch
    stage 4  stride 16   plain nnU-Net conv block                          512 ch
    stage 5  stride 32   plain nnU-Net conv block                          512 ch
    stage 6  stride 64   plain nnU-Net conv block                          512 ch
    stage 7  stride 128  plain nnU-Net conv block                          512 ch

The 1x1 projections keep the *fingerprint's feature budget* for every skip, so
the decoder is exactly the size stock nnU-Net would have built; only the
encoder's inductive bias changes.  (A residual encoder arm that ignored the
budget measured 135 s/epoch and 65.6 M parameters at stride 1; keeping the
budget roughly halves both.)

Three deliberate adaptations:

* the resnet stem keeps `conv1`'s pretrained 7x7 filters but its stride is set
  to 1 and `maxpool` is dropped, so stage 0 really is at stride 1 -- the
  decoder must return a full-resolution logit or the deep-supervision targets
  cannot line up.  The 3-channel filter is folded to `input_channels` by taking
  the channel mean, which keeps the response scale of a z-scored grayscale
  image comparable to that of a z-scored RGB one.
* every BatchNorm2d in the trunk becomes InstanceNorm2d with the BN affine
  parameters copied in.  nnU-Net's 2D fingerprint asks for InstanceNorm, and
  BatchNorm with patch-based batch 4 uses noisy batch statistics.  The
  pretrained convolution *filters* are what we want; the normalisation is
  re-fit in the regime the rest of the network uses.
* `freeze_upto` freezes the first N trunk stages (parameters AND gradient flow,
  they run under no_grad).  The stride-1 and stride-2 layers are the expensive
  ones and also the most generic, which is the classic transfer-learning
  recipe: freeze the generic early filters, fine-tune the task-specific deep
  ones.  On 71 training images it also removes most of the overfitting surface.

The decoder is `dynamic_network_architectures`' own `UNetDecoder`, driven by a
duck-typed encoder that only has to expose output_channels / strides /
kernel_sizes / norm_op / ... .  Reusing the real decoder guarantees the deep
supervision list (7 outputs, full resolution first) and
`network.decoder.deep_supervision` behave exactly as nnU-Net expects.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Type

import numpy as np
import torch
import torch.nn as nn

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

TORCH_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")

# torchvision IMAGENET1K_V1 checkpoint file names (cached on this host)
BACKBONES = {
    "resnet18": ("resnet18-f37072fd.pth", (64, 128, 256, 512)),
    "resnet34": ("resnet34-333f7ec4.pth", (64, 128, 256, 512)),
}


def _bn_to_in(module: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d with an InstanceNorm2d that keeps the affine
    parameters (and drops the running statistics)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            new = nn.InstanceNorm2d(child.num_features, eps=1e-5, affine=True,
                                    track_running_stats=False)
            with torch.no_grad():
                new.weight.copy_(child.weight.detach())
                new.bias.copy_(child.bias.detach())
            setattr(module, name, new)
        else:
            _bn_to_in(child)
    return module


class PreEncEncoder(nn.Module):
    """Duck-typed encoder: same public attributes as `PlainConvEncoder`."""

    def __init__(self,
                 input_channels: int = 1,
                 backbone: str = "resnet18",
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 512, 512, 512, 512),
                 kernel_sizes: Sequence[Sequence[int]] = None,
                 strides: Sequence[Sequence[int]] = None,
                 n_conv_per_stage: Sequence[int] = None,
                 conv_op: Type[nn.Module] = nn.Conv2d,
                 conv_bias: bool = True,
                 norm_op: Type[nn.Module] = nn.InstanceNorm2d,
                 norm_op_kwargs: dict = None,
                 dropout_op=None,
                 dropout_op_kwargs=None,
                 nonlin: Type[nn.Module] = nn.LeakyReLU,
                 nonlin_kwargs: dict = None,
                 pretrained: bool = True,
                 freeze_upto: int = 2):
        super().__init__()
        norm_op_kwargs = {"eps": 1e-5, "affine": True} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {"inplace": True} if nonlin_kwargs is None else nonlin_kwargs
        assert conv_op is nn.Conv2d, "this encoder is 2D only"

        n_stages = len(features_per_stage)
        assert n_stages == 8, "the resnet trunk covers 4 stages, so n_stages must be 8"
        if kernel_sizes is None:
            kernel_sizes = [[3, 3]] * n_stages
        if strides is None:
            strides = [[1, 1]] + [[2, 2]] * (n_stages - 1)
        if n_conv_per_stage is None:
            n_conv_per_stage = [2] * n_stages

        # --- attributes the UNetDecoder reads -------------------------------
        self.conv_op = conv_op
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.output_channels = list(features_per_stage)
        self.strides = [list(s) for s in strides]
        self.kernel_sizes = [list(k) for k in kernel_sizes]
        self.n_conv_per_stage = list(n_conv_per_stage)
        self.input_channels = int(input_channels)
        self.backbone_name = backbone
        self.freeze_upto = int(freeze_upto)

        # --- pretrained resnet trunk (taps at strides 1,2,4,8) --------------
        import torchvision
        if backbone not in BACKBONES:
            raise ValueError("unknown backbone %r" % backbone)
        ckpt, raw_ch = BACKBONES[backbone]
        ref = getattr(torchvision.models, backbone)(weights=None)

        w3 = ref.conv1.weight.detach()                       # (64, 3, 7, 7)
        new_conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7,
                              stride=1, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight.copy_(w3.mean(dim=1, keepdim=True).repeat(1, self.input_channels, 1, 1))

        self.conv1 = new_conv1
        self.bn1 = ref.bn1
        self.relu = ref.relu
        self.layer1 = ref.layer1          # -> stride 1,  64 ch
        self.layer2 = ref.layer2          # -> stride 2,  128 ch
        self.layer3 = ref.layer3          # -> stride 4,  256 ch
        self.layer4 = ref.layer4          # -> stride 8,  512 ch
        del ref

        if pretrained:
            try:
                sd = torch.load(os.path.join(TORCH_CACHE, ckpt), map_location="cpu")
            except RuntimeError:
                # resnet34-333f7ec4.pth is still in the LEGACY .tar format, which
                # torch >= 2.6 refuses to load with the default weights_only=True
                # ("Cannot use ``weights_only=True`` with files saved in the legacy
                # .tar format").  It is a trusted torchvision IMAGENET1K_V1 file
                # from the local cache, so falling back is safe.
                sd = torch.load(os.path.join(TORCH_CACHE, ckpt), map_location="cpu",
                                weights_only=False)
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            sd.pop("conv1.weight", None)                      # 1-channel version is set above
            missing, unexpected = self.load_state_dict(sd, strict=False)
            self._load_report = ("pretrained %s: %d missing, %d unexpected"
                                 % (backbone, len(missing), len(unexpected)))
            bad = [u for u in unexpected if "running_" not in u and "num_batches_tracked" not in u]
            assert not bad, "unexpected keys while loading the trunk: %r" % bad[:5]
        else:
            self._load_report = "backbone %s NOT pretrained" % backbone

        # BN -> InstanceNorm, after the pretrained weights are in
        _bn_to_in(self)

        # --- 1x1 projections onto the fingerprint's feature budget ----------
        self.proj = nn.ModuleList()
        for s in range(4):
            ch = int(self.output_channels[s])
            block = nn.Sequential()
            block.add_module("conv", conv_op(raw_ch[s], ch, 1, 1, 0, bias=True))
            block.add_module("norm", norm_op(ch, **norm_op_kwargs))
            block.add_module("act", nonlin(**nonlin_kwargs))
            self.proj.append(block)

        # --- plain stages 4..7 ----------------------------------------------
        self.stages_plain = nn.ModuleList()
        for s in range(4, n_stages):
            in_ch = int(self.output_channels[s - 1])
            out_ch = int(self.output_channels[s])
            self.stages_plain.append(StackedConvBlocks(
                int(n_conv_per_stage[s]), conv_op, in_ch, out_ch,
                int(self.kernel_sizes[s][0]), int(self.strides[s][0]),
                conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, False))

        self._apply_freezing()

    # ------------------------------------------------------------------ helpers
    def _apply_freezing(self):
        blocks = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        # freeze_upto = 1 -> conv1/bn1/layer1, 2 -> + layer2, ...
        frozen = [self.conv1, self.bn1]
        if self.freeze_upto >= 1:
            frozen.append(self.layer1)
        if self.freeze_upto >= 2:
            frozen.append(self.layer2)
        if self.freeze_upto >= 3:
            frozen.append(self.layer3)
        if self.freeze_upto >= 4:
            frozen.append(self.layer4)
        if self.freeze_upto > 0:
            for m in frozen:
                for p in m.parameters():
                    p.requires_grad_(False)
        self._frozen_modules = frozen
        del blocks

    def load_report(self) -> str:
        return self._load_report

    def low_lr_parameters(self):
        """Pretrained-but-trainable parameters: layer3, layer4 and the projections."""
        for mod in (self.layer3, self.layer4, self.proj):
            for p in mod.parameters():
                yield p

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ forward
    def _stem(self, x):
        return self.layer1(self.relu(self.bn1(self.conv1(x))))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        fz = self.freeze_upto
        if fz >= 1:
            with torch.no_grad():
                r0 = self._stem(x)
        else:
            r0 = self._stem(x)

        if fz >= 2:
            with torch.no_grad():
                r1 = self.layer2(r0)
        else:
            r1 = self.layer2(r0)

        if fz >= 3:
            with torch.no_grad():
                r2 = self.layer3(r1)
        else:
            r2 = self.layer3(r1)

        if fz >= 4:
            with torch.no_grad():
                r3 = self.layer4(r2)
        else:
            r3 = self.layer4(r2)

        skips = [self.proj[0](r0), self.proj[1](r1), self.proj[2](r2), self.proj[3](r3)]
        for blk in self.stages_plain:
            skips.append(blk(skips[-1]))
        return skips

    def compute_conv_feature_map_size(self, input_size):
        """Same bookkeeping as PlainConvEncoder: only used for reporting."""
        assert len(input_size) == len(self.strides[0])
        size = [int(i) for i in input_size]
        out = np.int64(0)
        for s in range(len(self.output_channels)):
            if s > 0:
                size = [int(i // j) for i, j in zip(size, self.strides[s])]
            npx = int(np.prod([self.output_channels[s], *size], dtype=np.int64))
            n_conv = int(self.n_conv_per_stage[s])
            out += np.int64(npx * max(n_conv, 1))
        return out


class PreEncUNet(AbstractDynamicNetworkArchitectures):
    """8-stage U-Net: pretrained resnet trunk + nnU-Net's own decoder."""

    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 features_per_stage: Sequence[int],
                 kernel_sizes: Sequence[Sequence[int]],
                 strides: Sequence[Sequence[int]],
                 n_conv_per_stage: Sequence[int],
                 n_conv_per_stage_decoder: Sequence[int],
                 norm_op: Type[nn.Module] = nn.InstanceNorm2d,
                 norm_op_kwargs: dict = None,
                 dropout_op=None,
                 dropout_op_kwargs=None,
                 nonlin: Type[nn.Module] = nn.LeakyReLU,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = True,
                 deep_supervision: bool = True,
                 backbone: str = "resnet18",
                 pretrained: bool = True,
                 freeze_upto: int = 2):
        super().__init__()

        self.key_to_encoder = "encoder"
        self.key_to_stem = "encoder.conv1"
        self.keys_to_in_proj = ("encoder.conv1",)

        self.encoder = PreEncEncoder(
            input_channels=input_channels,
            backbone=backbone,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            pretrained=pretrained,
            freeze_upto=freeze_upto,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def compute_conv_feature_map_size(self, input_size):
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        """Called by get_network_from_plans(allow_init=True).  The pretrained
        trunk must NOT be re-initialised, so this only touches the projections,
        the plain stages 4..7 and the decoder."""
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He
        init = InitWeights_He(1e-2)
        for name, sub in module.named_children():
            if name == "encoder":
                init(sub.proj)
                init(sub.stages_plain)
            else:
                init(sub)

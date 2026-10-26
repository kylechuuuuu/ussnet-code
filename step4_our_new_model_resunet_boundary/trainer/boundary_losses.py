#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/boundary_losses.py

Why this file exists
--------------------
The step-3 failure analysis measured the single worst number of the U-Net
baseline: *boundary agreement*.  Its Dice is 93.0, but the Dice restricted to a
1-pixel band around the ground-truth boundary is only 23.3 % (34.0 % at 2 px),
and the predicted perimeter is 1.81x the ground-truth perimeter (1957 px vs
1079 px).  Dice is dominated by the bulk of the gland, so it hides this
completely.  A plain Dice+CE objective (what both the U-Net and stock nnU-Net
optimise) has almost no gradient pressure exactly where the errors are.

Fix: keep the stock nnU-Net Dice + CE objective, and add a term that gives the
pixels in a thin band around the ground-truth boundary extra weight.  The band
is obtained with a morphological gradient implemented with max-pooling, so it
runs on the GPU with no scipy / CPU round-trip, and it only touches the target.

    L = L_dice_ce  +  lambda(t) * L_boundary

`lambda(t)` is ramped from 0 to `boundary_weight` over the first
`boundary_warmup_epochs` epochs: early in training the network is far from the
target and a boundary-weighted term mostly amplifies noise, which is the
well-known failure mode of boundary losses trained from scratch.

The term is a *weighted mean* ((per_pixel * w).sum() / w.sum()), not the
default reduced mean, so raising the weight on the band does not silently
scale the whole term up.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def boundary_band(mask: torch.Tensor, band: int = 3) -> torch.Tensor:
    """Morphological gradient: 1 where `mask` changes value, 0 in flat areas.

    mask: (B, 1, X, Y) float tensor holding 0/1.  band: odd kernel size, so
    `band=3` gives a 1-pixel-thick boundary ring (3x3 max-pool gradient).
    """
    assert band % 2 == 1, "band must be odd"
    pad = band // 2
    dil = F.max_pool2d(mask, band, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, band, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


class BoundaryWeightedBCE(nn.Module):
    """Weighted BCE on the foreground channel, extra weight near the GT boundary.

    Arguments
    ---------
    beta : extra weight the boundary band gets (a band pixel counts 1 + beta).
    band : odd kernel size of the band (3 -> 1 px thick).

    Inputs follow nnU-Net's convention *at one deep-supervision head*:
      net_output : (B, C, X, Y) logits, C = 2 (background, prostate)
      target     : (B, 1, X, Y) int16/long label map with values 0 / 1
                   (nnU-Net does NOT pass one-hot targets to the loss)
    """

    def __init__(self, beta: float = 4.0, band: int = 3):
        super().__init__()
        self.beta = float(beta)
        self.band = int(band)
        # ramped 0 -> boundary_weight by the trainer; a plain multiplier keeps
        # the ramp visible in the loss logs.
        self.multiplier = 0.0

    def set_multiplier(self, value: float):
        self.multiplier = float(value)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = net_output[:, 1:2]                       # (B, 1, X, Y) fg logits
        if target.dim() == 3:                             # (B, X, Y) -> (B, 1, X, Y)
            target = target[:, None]
        fg = (target[:, 0:1] > 0.5).to(logits.dtype)      # (B, 1, X, Y)
        if fg.shape[-2:] != logits.shape[-2:]:            # deep-supervision safety
            fg = F.interpolate(fg, size=logits.shape[-2:], mode="nearest")
        w = 1.0 + self.beta * boundary_band(fg, self.band)
        per_pixel = F.binary_cross_entropy_with_logits(logits, fg, reduction="none")
        loss = (per_pixel * w).sum() / w.sum().clamp_min(1e-6)
        return self.multiplier * loss


class EdgeAwareCompoundLoss(nn.Module):
    """nnU-Net's loss (already deep-supervision wrapped) + an optional boundary
    term, applied at the full-resolution head only (where the band is meaningful).

    `net_output` / `target` are the lists produced by the deep-supervision
    pipeline, so index 0 is the full-resolution head.
    """

    def __init__(self, base_loss: nn.Module, boundary_loss: BoundaryWeightedBCE,
                 boundary_weight: float, warmup_epochs: int, is_full_resolution: bool = True):
        super().__init__()
        self.base_loss = base_loss
        self.boundary_loss = boundary_loss
        self.boundary_weight = float(boundary_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.is_full_resolution = bool(is_full_resolution)

    def set_epoch(self, epoch: int):
        if self.warmup_epochs <= 0:
            self.boundary_loss.set_multiplier(self.boundary_weight)
        else:
            frac = min(1.0, float(epoch) / float(self.warmup_epochs))
            self.boundary_loss.set_multiplier(self.boundary_weight * frac)

    def forward(self, net_output, target) -> torch.Tensor:
        loss = self.base_loss(net_output, target)
        if self.is_full_resolution and self.boundary_loss.multiplier > 0.0:
            if isinstance(net_output, (list, tuple)):
                out0, tgt0 = net_output[0], target[0]
            else:
                out0, tgt0 = net_output, target
            loss = loss + self.boundary_loss(out0, tgt0)
        return loss

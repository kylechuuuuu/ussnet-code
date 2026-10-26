#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/fp_boundary_losses.py

The redesigned objective, driven by the boundary/geometry fingerprint.

WHY THESE THREE TERMS (each one answers a measurement, not a hunch)
-------------------------------------------------------------------
1. `TolerantBoundaryDice`  -- the *tolerant* boundary term.
   Measurement: the residual contour error of BOTH models is a coherent
   displacement of ~1-3 px, not high-frequency jitter (gaussian-smoothing the
   probability map changes nnU-Net's boundary Dice@1px by only +0.11, while it
   does remove U-Net's jaggies).  A 1 px band therefore punishes a prediction
   that is uniformly 2 px off exactly as hard as one that is a mess.  This term
   instead scores Dice between *bands* of +-k px around GT and around the
   prediction, i.e. it directly optimises boundary Dice at a tolerance k that
   the fingerprint derives from the gland's linear scale (k = 1 % of 281 px = 3).
   Fully GPU-side (max-pool morphological gradient), no distance transform.

2. `SharpBoundaryBCE` -- the sharp 1 px pressure, kept from the v1 design but
   now with its band/beta/weight derived from `boundary_contrast` instead of
   being hard-coded.  It is only meaningful because the fingerprint says the GT
   boundary really is an edge (|gradI| ratio 3.47x, 77 % of boundary pixels above
   the image p75); if `boundary_is_strong_edge` were False this term would be
   pointless.

3. `SoftPerimeterLoss` -- the length penalty.
   Measurement: U-Net's predicted perimeter is 1.81x the ground truth while
   nnU-Net's is 1.01x, and the ground truth is extremely smooth (morphological
   opening with a disk of radius 30 px still reproduces it at Dice >= 0.995).
   Total variation of the probability map is a differentiable length; we
   normalise it by the ground truth's own TV so the term is ~1.0 for a
   GT-length boundary and ~1.8 for U-Net's jaggies, making the weight
   scale-free.

Optional direction weighting (from `angular_profile`): sector weights are
1/relative-edge-strength, so the anterior sectors -- where the fingerprint says
the edge is weakest (rel 0.65-0.67) and where the error analysis measured the
FN rate to be 2.2x the posterior sectors -- get more weight.

Deep supervision: the extra terms are applied at the **full-resolution head
only**, because a +-3 px band is meaningless after 4x downsampling.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- helpers
def boundary_band(mask: torch.Tensor, band: int = 3) -> torch.Tensor:
    """Hard morphological gradient: 1 on the ~band-wide ring around the boundary.

    mask: (B, 1, X, Y) float 0/1. `band` must be odd (max-pool kernel).
    """
    assert band % 2 == 1, "band must be odd"
    pad = band // 2
    dil = F.max_pool2d(mask, band, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, band, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def soft_band(prob: torch.Tensor, band: int) -> torch.Tensor:
    """Differentiable (grey) morphological gradient of a probability map."""
    pad = band // 2
    dil = F.max_pool2d(prob, band, stride=1, padding=pad)
    ero = -F.max_pool2d(-prob, band, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def fg_from_target(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 3:
        target = target[:, None]
    return (target[:, 0:1] > 0.5).to(torch.float32)



def fg_prob(logits: torch.Tensor) -> torch.Tensor:
    """Probability of the FOREGROUND, whatever the number of classes.

    For a binary model this is exactly `softmax(logits)[:, 1:2]`.  For a k-class
    model (k > 2) the foreground we ultimately score is the UNION of the
    foreground classes, i.e. `1 - P(background)`; writing `[:, 1:2]` there would
    silently score only the first zone while the boundary/length terms claim to
    describe the gland outline.  Both branches are the same tensor when k == 2.
    """
    p = torch.softmax(logits, dim=1)
    if p.shape[1] == 2:
        return p[:, 1:2]
    return 1.0 - p[:, 0:1]

def total_variation(prob: torch.Tensor) -> torch.Tensor:
    """Differentiable perimeter proxy: TV of the foreground probability map."""
    gx = (prob[..., :, 1:] - prob[..., :, :-1]).abs().sum(dim=(1, 2, 3))
    gy = (prob[..., 1:, :] - prob[..., :-1, :]).abs().sum(dim=(1, 2, 3))
    return gx + gy


def sector_weight_map(gt: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-pixel weight from the angular sector of the GT centroid.

    gt: (B, 1, X, Y) float 0/1.  weights: (S,) tensor.  Returns (B, 1, X, Y).
    Angle convention matches the fingerprint: +90 deg = image up = anterior.
    """
    b, _, h, w = gt.shape
    dev = gt.device
    yy = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, w)
    m = gt.sum(dim=(2, 3)).clamp_min(1.0)                    # (B,1)
    cy = (gt * yy).sum(dim=(2, 3)) / m                       # (B,1)
    cx = (gt * xx).sum(dim=(2, 3)) / m
    ang = torch.atan2(-(yy - cy.view(b, 1, 1, 1)), xx - cx.view(b, 1, 1, 1))
    s = weights.numel()
    idx = torch.floor((ang + math.pi) / (2 * math.pi) * s).long().clamp_(0, s - 1)
    return weights.to(dev)[idx]                              # (B,1,H,W)


# ----------------------------------------------------------------------- terms
class TolerantBoundaryDice(nn.Module):
    """1 - Dice(band(prediction, k), band(GT, k))  -- a tolerant boundary metric.

    This is the differentiable version of the `boundary Dice@k` that the
    failure analysis reports, which is the number we actually want to move.
    Per-sample sums then mean over the batch: band sizes differ a lot between
    images (foreground fraction 0.031 .. 0.241), so a global sum would let the
    large glands dominate.
    """

    def __init__(self, band: int = 7, smooth: float = 1e-5):
        super().__init__()
        self.band = int(band)
        self.smooth = float(smooth)

    def forward(self, net_output, target):
        prob = fg_prob(net_output)
        gt = fg_from_target(target)
        if gt.shape[-2:] != prob.shape[-2:]:
            gt = F.interpolate(gt, size=prob.shape[-2:], mode="nearest")
        g = boundary_band(gt, self.band)
        p = soft_band(prob, self.band)
        num = 2.0 * (p * g).sum(dim=(1, 2, 3)) + self.smooth
        den = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3)) + self.smooth
        return (1.0 - num / den).mean()


class SharpBoundaryBCE(nn.Module):
    """Boundary-weighted BCE on the foreground channel.

    beta and band come from `boundary_contrast`.  `dir_weights` (optional) is a
    (S,) tensor of angular sector weights; `per_pixel` is multiplied by it.
    The reduction is a *weighted mean* so that raising the weight on the band
    does not silently scale the whole term.
    """

    def __init__(self, beta: float = 4.0, band: int = 3, dir_weights=None):
        super().__init__()
        self.beta = float(beta)
        self.band = int(band)
        if dir_weights is not None:
            self.register_buffer("dir_weights",
                                 torch.as_tensor(dir_weights, dtype=torch.float32))
        else:
            self.dir_weights = None

    def forward(self, net_output, target):
        logits = net_output[:, 1:2]
        gt = fg_from_target(target)
        if gt.shape[-2:] != logits.shape[-2:]:
            gt = F.interpolate(gt, size=logits.shape[-2:], mode="nearest")
        w = 1.0 + self.beta * boundary_band(gt, self.band)
        if self.dir_weights is not None:
            w = w * sector_weight_map(gt, self.dir_weights)
        per_pixel = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
        return (per_pixel * w).sum() / w.sum().clamp_min(1e-6)


class SoftPerimeterLoss(nn.Module):
    """TV(prediction) / TV(GT): a scale-free, differentiable length penalty."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, net_output, target):
        prob = fg_prob(net_output)
        gt = fg_from_target(target)
        if gt.shape[-2:] != prob.shape[-2:]:
            gt = F.interpolate(gt, size=prob.shape[-2:], mode="nearest")
        ptv = total_variation(prob)
        gtv = total_variation(gt)
        return (ptv / gtv.clamp_min(self.smooth)).mean()


# ------------------------------------------------------------------- composite
class FPCompoundLoss(nn.Module):
    """stock nnU-Net loss (deep-supervision wrapped) + the fingerprint terms.

    The extra terms ramp in over `warmup_epochs` (early on the network is far
    from the target and a boundary term mostly amplifies noise) and are applied
    at the full-resolution head only.
    """

    def __init__(self, base_loss, cfg, is_full_resolution: bool = True):
        super().__init__()
        self.base_loss = base_loss
        self.is_full_resolution = bool(is_full_resolution)
        self.disabled = bool(getattr(cfg, "disabled", False))
        d = cfg.loss
        self.w_tol = float(d["weight_tolerant_bdice"])
        self.w_bce = float(d["weight_boundary_bce"])
        self.w_per = float(d["weight_perimeter"])
        self.warmup_epochs = int(d["warmup_epochs"])
        dir_w = d.get("dir_sector_weights") if getattr(cfg, "use_direction", False) else None
        self.tolerant = TolerantBoundaryDice(band=int(d["tolerant_band"]))
        self.sharp = SharpBoundaryBCE(beta=float(d.get("bce_beta", 4.0)),
                                      band=int(d.get("bce_band", 3)),
                                      dir_weights=dir_w)
        self.length = SoftPerimeterLoss()
        self.multiplier = 0.0

    def set_epoch(self, epoch: int):
        if self.disabled or self.warmup_epochs <= 0:
            self.multiplier = 0.0 if self.disabled else 1.0
        else:
            self.multiplier = min(1.0, float(epoch) / float(self.warmup_epochs))

    def extra_terms(self, logits, target):
        return (self.w_tol * self.tolerant(logits, target)
                + self.w_bce * self.sharp(logits, target)
                + self.w_per * self.length(logits, target))

    def forward(self, net_output, target):
        loss = self.base_loss(net_output, target)
        if self.disabled or self.multiplier <= 0.0:
            return loss
        if self.is_full_resolution and isinstance(net_output, (list, tuple)):
            out0, tgt0 = net_output[0], target[0]
        else:
            out0, tgt0 = net_output, target
        return loss + self.multiplier * self.extra_terms(out0, tgt0)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/trainer/sandwich_losses.py

THE THREE-VERSION OBJECTIVE OF THE SANDWICH SQUEEZE MODEL (夹逼定理 v2).

    base      (基础版本) : stock DC+CE against GT                -- the estimate
    cautious  (谨慎版本) : stock DC+CE against erode(GT,  r_c)   -- inner bound
    amplified (放大版本) : stock DC+CE against dilate(GT, r_a)   -- outer bound

Why this is the honest form of the theorem: in step 5 the two bound targets
were annealed onto GT itself (r: 6 -> 1 px), so by the end they differed from
GT by only ~2.8 % of the pixels and carried almost no independent information.
Here the two versions keep a FIXED, error-scale separation (r_c = 3 px inward,
r_a = 4 px outward; the residual contour displacement of every model on this
dataset is 1-3 px, step 4 section 5.2), so the trained bounds really bound.
The nesting

        {p_cautious > 0.5}  ⊆  {p_base > 0.5}  ⊆  {p_amplified > 0.5}

is enforced EXACTLY at inference by construction (tools/62 + tools/63), so --
unlike step 5 -- there is NO ordering hinge and NO gap term in the objective.
The two failure modes of step 5 (bounds collapsing onto GT, the angular band
being erased by the gap term) simply cannot occur: the angular modulation acts
on the TARGETS and nothing overwrites it.

Deep supervision: every term uses the stock nnU-Net deep-supervision weights;
the bound targets follow the deep-supervision downsampling of the GT (radii are
in pixels of that scale); a bound narrower than 1 px at a scale is skipped
there (nothing left to bound).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- morphology
def _erode(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=2 * k + 1, stride=1, padding=k)


def _dilate(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=2 * k + 1, stride=1, padding=k)


def chebyshev_depth_inside(gt: torch.Tensor, rmax: int) -> torch.Tensor:
    """depth_in(p) = largest d <= rmax such that p survives d erosions of gt."""
    d = torch.zeros_like(gt)
    cur = gt
    for _ in range(int(rmax)):
        cur = _erode(cur, 1)
        d = d + cur
    return d


def chebyshev_depth_outside(gt: torch.Tensor, rmax: int) -> torch.Tensor:
    """depth_out(p) = min d <= rmax with p inside dilate^d(gt) (0 for foreground)."""
    d = torch.zeros_like(gt)
    cur = gt
    for _ in range(int(rmax)):
        d = d + (1.0 - cur)
        cur = _dilate(cur, 1)
    return d


def sector_scale_map(gt: torch.Tensor, sector_scale, radius: float) -> torch.Tensor:
    """Per-pixel bound radius  r * s(theta)   (theta from the GT centroid).

    `sector_scale` comes from the fingerprint's dir_sector_weights: 1.0 where
    the edge evidence is average, > 1 where it is weak -> the interval opens.
    Angle convention matches step 4 (+90 deg = image up = anterior).
    """
    if sector_scale is None or len(sector_scale) == 0:
        return torch.full_like(gt, float(radius), dtype=torch.float32)
    b, _, h, w = gt.shape
    dev = gt.device
    yy = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, w)
    m = gt.sum(dim=(2, 3)).clamp_min(1.0)
    cy = (gt * yy).sum(dim=(2, 3)) / m                       # (B,1)
    cx = (gt * xx).sum(dim=(2, 3)) / m
    ang = torch.atan2(-(yy - cy.view(b, 1, 1, 1)), xx - cx.view(b, 1, 1, 1))
    s = len(sector_scale)
    idx = torch.floor((ang + math.pi) / (2 * math.pi) * s).long().clamp_(0, s - 1)
    scale = torch.tensor([float(v) for v in sector_scale], device=dev, dtype=torch.float32)
    return float(radius) * scale[idx].to(gt.dtype)


def fg_from_target(target: torch.Tensor) -> torch.Tensor:
    """Foreground mask from an nnU-Net target ((B,1,H,W) label map, or (B,2,..) one-hot)."""
    if target.dim() == 3:
        target = target[:, None]
    if target.shape[1] == 1:
        return (target[:, 0:1] > 0.5).to(torch.float32)
    return (target[:, 1:2] > 0.5).to(torch.float32)


def as_label_map(mask: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) float 0/1 -> the exact layout DC_and_CE_loss expects."""
    return mask.to(torch.float32)


def softmax_fg(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1:2]


# ------------------------------------------------------------------- loss
class SandwichLoss(nn.Module):
    """L = L_base + w_c * L_cautious + w_a * L_amplified (each: stock DC+CE, DS-weighted)."""

    def __init__(self, dc_ce_factory, ds_weights, cfg, device=None):
        super().__init__()
        self.loss_base = dc_ce_factory()
        self.loss_cautious = dc_ce_factory()
        self.loss_amplified = dc_ce_factory()
        self.ds_weights = [float(w) for w in ds_weights]
        self.cfg = cfg
        self.last_diag = {}
        # Optional additional objective, set by a subclass (step 9).  When it
        # is None the forward pass below is exactly the step-6 loss.
        self.extra = None

    def forward(self, output, target):
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        tgts = list(target) if isinstance(target, (list, tuple)) else [target]
        assert len(outs) == len(tgts), "output/target deep-supervision mismatch"
        assert len(outs) == len(self.ds_weights), "deep-supervision weight mismatch"

        l_base = torch.zeros((), device=outs[0].device)
        l_c = torch.zeros((), device=outs[0].device)
        l_a = torch.zeros((), device=outs[0].device)
        gt0 = caut_t = amp_t = None

        for i, (o, t) in enumerate(zip(outs, tgts)):
            w = self.ds_weights[i]
            if w <= 0.0:
                continue
            t = t.to(o.device)
            # the base version is trained exactly like the stock nnU-Net head
            l_base = l_base + w * self.loss_base(o[:, 0:2], t)

            gt = fg_from_target(t)
            stride = float(2 ** i)
            rc_s = self.cfg.r_c / stride
            ra_s = self.cfg.r_a / stride
            if rc_s < 1.0 and ra_s < 1.0:
                continue
            rmax = max(2, int(math.ceil(max(rc_s, ra_s))) + 1)
            din = chebyshev_depth_inside(gt, rmax)
            dout = chebyshev_depth_outside(gt, rmax)

            if rc_s >= 1.0:
                rc_map = (sector_scale_map(gt, self.cfg.sector_scale, rc_s)
                          if self.cfg.angular else torch.full_like(gt, float(rc_s)))
                cautious = (((din + 1.0) > rc_map) & (gt > 0.5)).to(gt.dtype)
                l_c = l_c + self.cfg.w_c * w * self.loss_cautious(o[:, 2:4],
                                                                  as_label_map(cautious))
                if i == 0:
                    caut_t = cautious
            if ra_s >= 1.0:
                ra_map = (sector_scale_map(gt, self.cfg.sector_scale, ra_s)
                          if self.cfg.angular else torch.full_like(gt, float(ra_s)))
                amplified = (dout <= ra_map).to(gt.dtype)
                l_a = l_a + self.cfg.w_a * w * self.loss_amplified(o[:, 4:6],
                                                                   as_label_map(amplified))
                if i == 0:
                    amp_t = amplified
            if i == 0:
                gt0 = gt

        total = l_base + l_c + l_a
        diag = self._diagnostics(outs[0], gt0, caut_t, amp_t)
        diag["l_base"] = float(l_base.detach())
        diag["l_c"] = float(l_c.detach())
        diag["l_a"] = float(l_a.detach())
        if self.extra is not None:
            extra_loss, extra_diag = self.extra(outs, tgts, self.ds_weights)
            total = total + extra_loss
            diag["l_extra"] = float(extra_loss.detach())
            diag.update(extra_diag)
        self.last_diag = diag
        return total

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def _diagnostics(self, o0, gt, caut_t, amp_t):
        """Cheap per-iteration sanity of the sandwich (logged by the trainer):
        does the cautious version stay inside GT, does amplified cover GT,
        is the true boundary inside the predicted interval, how wide is it."""
        d = {}
        if gt is None:
            return d
        p_b = softmax_fg(o0[:, 0:2])
        p_c = softmax_fg(o0[:, 2:4])
        p_a = softmax_fg(o0[:, 4:6])
        pred_b = (p_b > 0.5).float()
        pred_c = (p_c > 0.5).float()
        pred_a = (p_a > 0.5).float()
        gt_sum = gt.sum().clamp_min(1.0)
        bd_gt = (_dilate(gt, 1) - _erode(gt, 1)).clamp(0.0, 1.0)
        band = pred_a * (1.0 - pred_c)
        d["caut_leak"] = float((pred_c - gt).clamp_min(0.0).sum() / gt_sum)
        d["amp_miss"] = float((gt - pred_a).clamp_min(0.0).sum() / gt_sum)
        d["viol_in"] = float((pred_c - pred_b).clamp_min(0.0).sum() / gt_sum)
        d["viol_out"] = float((pred_b - pred_a).clamp_min(0.0).sum() / gt_sum)
        d["bd_contain"] = float((bd_gt * band).sum() / bd_gt.sum().clamp_min(1.0))
        d["width_px"] = float(band.sum() / bd_gt.sum().clamp_min(1.0))
        if caut_t is not None:
            d["dice_c"] = float(2.0 * (pred_c * caut_t).sum()
                                / (pred_c.sum() + caut_t.sum() + 1e-6))
        if amp_t is not None:
            d["dice_a"] = float(2.0 * (pred_a * amp_t).sum()
                                / (pred_a.sum() + amp_t.sum() + 1e-6))
        return d

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/trainer/boundary_losses.py

MECHANISM 2 of step 9 -- make the OBJECTIVE boundary-metric instead of
area-metric.

WHY (measured in this project, all on the fold-0 val split):

  * The base head is trained with stock Dice+CE.  Both are AREA functionals.
    A 100 px long, 20 px deep excursion of the anterior midline into the
    bladder therefore costs

        2 * 2000 / (2 * 90000)  ~=  2.2 %  Dice

    while it destroys 14 % of all 1 px boundary points (step9/tools/91:
    |t_gt| > 10 px on 14.1 % of contour points, 6.8 % beyond 20 px).  Dice is
    literally blind to WHERE the error sits, as long as the total area is
    small.
  * The failure is not diffuse: 31/31 val cases have a contiguous tail, the
    tail carries essentially all of HD95 (Spearman(HD95, length of the
    |t|>10 run) = +0.756) and step9/tools/96 showed that repairing only the
    |t|>10 px points moves bdice@1px 34.5 -> 45.7 and HD95 28.3 -> 12.6.
  * step6's hard-shell sandwich (erode-3 / dilate-4 targets on SEPARATE heads)
    is the only boundary-aware supervision this project ever ran and it did
    move bdice@1px (stock control 32.36 -> 35.33 at equal Dice) -- but the
    shell is 3-4 px wide, it lives on heads that are not the one scored, and
    step9/tools/92 showed the three heads fail CORRELATEDLY in the anterior.

THE LOSS  (distance-weighted symmetric tolerance-band penalty)

  With q(p) = foreground probability, g(p) = GT mask, and Chebyshev distances
  to the GT region (pooling ops, no scipy, fully vectorised on GPU):

      d_out(p) = distance from p outside to GT   (0 for p inside GT, >=1 out)
      d_in(p)  = depth of p inside GT            (0 for p outside GT, >=1 in)
      b_g(p)   = 1 px boundary of GT
      P        = sum(b_g)          (boundary length in px)
      w(p)     = clip(d(p), 0, C*D)            C = 3  -> "totally wrong" cap

      L_out = (2 / (D*P)) * sum_p w_out(p) * q(p)          # mass leaking OUT
      L_in  = (2 / (D*P)) * sum_p w_in(p)  * (1 - q(p))    # GT mass MISSED
      L_band = w_band * (L_out + L_in)

  Why THIS normalisation and THIS weight (the first version of this file got
  it wrong and the smoke test caught it -- recorded here so it is not retried):

    * w must be a RAMP, not a saturating clip: with w = clip(d/D, 0, 1) every
      pixel further than D px from the GT gets the same weight, so the "band"
      becomes the whole image background and L_out degenerates into a global
      volume penalty (~0.34 for a 20 px inflation of a 50x50 square, versus
      ~0.93 for the symmetric deflation: not a boundary loss at all, and it
      would just shrink the mask).
    * Normalising by the band area has the same defect: the outside band is
      unbounded, the inside band is bounded by the object, so the two terms
      are not comparable.  Normalising both by the GT BOUNDARY LENGTH (P)
      makes them exactly symmetric and gives L_band a clean reading:

          uniformly inflated/deflated by delta  ->  L_band ~= delta^2 / D

      i.e. "mean boundary displacement, measured quadratically, in units of D".
      A 3 px global displacement costs 0.75; a 1 px one costs 0.083.  Small
      errors are almost free -- which is what bdice@1px itself does, and we do
      NOT want to chase sub-pixel contours.
    * The quadratic ramp means the 20-60 px anterior tail is penalised far
      harder than a 3 px wobble, unlike Dice.
    * Symmetric in in/out => cannot be gamed by shrinking or growing.

  D is the only new hyper-parameter (the tolerated displacement scale).
  step9/tools/91 gives p90(|t_gt|) = 8.7 px, so D = 12 px sits just above the
  bulk of the residual; C = 3 puts full weight on anything beyond 36 px.

This module is deliberately kept OUT of step6's trainer directory: the step-6
sandwich results are a fixed reference point and must not be perturbed by a
later edit.  It imports the step-6 helpers read-only.
"""
import math

import torch
import torch.nn as nn

from sandwich_losses import (_dilate, _erode, chebyshev_depth_inside,
                             chebyshev_depth_outside, fg_from_target, softmax_fg)


# ------------------------------------------------------------------ core term
def band_terms(q: torch.Tensor, gt: torch.Tensor, D: float, cap: float = 3.0):
    """L_out, L_in for one scale.  q, gt: (B,1,H,W) in [0,1]."""
    D = float(max(D, 1.0))
    C = float(cap) * D
    R = int(math.ceil(C)) + 1
    d_out = chebyshev_depth_outside(gt, R)        # 0 inside, 1.. outside
    d_in = chebyshev_depth_inside(gt, R)          # 0 outside, 1.. inside
    w_out = torch.clamp(d_out, max=C)
    w_in = torch.clamp(d_in, max=C)
    bd_g = (_dilate(gt, 1) - _erode(gt, 1)).clamp(0.0, 1.0)
    P = bd_g.sum().clamp_min(1.0)
    k = 2.0 / (D * P)
    l_out = k * (w_out * q).sum()
    l_in = k * (w_in * (1.0 - q)).sum()
    return l_out, l_in


@torch.no_grad()
def boundary_diag(q: torch.Tensor, gt: torch.Tensor, tols=(1, 3, 8)):
    """Unweighted boundary recall/precision at a few tolerances (logging only)."""
    pb = (q > 0.5).to(gt.dtype)
    bd_p = (_dilate(pb, 1) - _erode(pb, 1)).clamp(0.0, 1.0)
    bd_g = (_dilate(gt, 1) - _erode(gt, 1)).clamp(0.0, 1.0)
    out = {}
    for t in tols:
        rec = (bd_g * _dilate(bd_p, int(t))).sum() / bd_g.sum().clamp_min(1.0)
        pre = (bd_p * _dilate(bd_g, int(t))).sum() / bd_p.sum().clamp_min(1.0)
        out["brec%d" % t] = float(rec)
        out["bpre%d" % t] = float(pre)
    return out


# ---------------------------------------------------------------------- module
class BoundaryBandLoss(nn.Module):
    """Extra objective for the base head; plug into SandwichLoss.extra.

    forward(outs, tgts, ds_weights) -> (loss, diag)
    """

    def __init__(self, D: float = 12.0, weight: float = 1.0,
                 scales=(0, 1, 2), cap: float = 3.0, min_D: float = 3.0):
        super().__init__()
        self.D = float(D)
        self.weight = float(weight)
        self.scales = tuple(int(s) for s in scales)
        self.cap = float(cap)
        self.min_D = float(min_D)

    def forward(self, outs, tgts, ds_weights):
        dev = outs[0].device
        total = torch.zeros((), device=dev)
        diag = {}
        gt0 = None
        q0 = None
        for i, (o, t) in enumerate(zip(outs, tgts)):
            if i not in self.scales:
                continue
            w = float(ds_weights[i])
            if w <= 0.0:
                continue
            stride = float(2 ** i)
            D_s = self.D / stride
            if D_s < self.min_D:
                continue
            t = t.to(o.device)
            gt = fg_from_target(t)
            q = softmax_fg(o[:, 0:2])
            l_out, l_in = band_terms(q, gt, D_s, cap=self.cap)
            total = total + w * (l_out + l_in)
            diag["bnd_out_s%d" % i] = float(l_out.detach())
            diag["bnd_in_s%d" % i] = float(l_in.detach())
            if i == 0:
                gt0, q0 = gt, q
        if q0 is not None:
            diag.update(boundary_diag(q0, gt0))
        diag["bnd_D"] = self.D
        return self.weight * total, diag

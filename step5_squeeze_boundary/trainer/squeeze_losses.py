#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/trainer/squeeze_losses.py

THE SQUEEZE-THEOREM (夹逼定理) OBJECTIVE FOR ULTRASOUND BOUNDARY DETERMINATION.

The theorem, and what each symbol becomes here
------------------------------------------------
Classical:  if  g(x) <= f(x) <= h(x)  for all x,  and  lim g = lim h = L,
            then  lim f = L   (the object is *pinned* by its two bounds).

Here:
    f  = the boundary of the main head          (target: GT)
    g  = the INNER bound:  erode(GT, r)         (a region that is certainly
                                                 inside the gland; its boundary
                                                 is inside the true boundary)
    h  = the OUTER bound:  dilate(GT, r)        (a region that certainly
                                                 contains the gland)
    r  = the annealing half-width r(t): R0 -> R_min   (the "limit")

Three things make the discrete model obey the theorem:

  1. INTERVAL CONSTRUCTION (hard, exact - not learned)
         Omega_in(r)  subseteq  Omega_gt  subseteq  Omega_out(r)     for all r
     because erode/dilate move monotonically.  This is the premise g <= f <= h.
     It is enforced on the *labels*, so the premise is true by construction and
     the network only has to reproduce it.

  2. ORDERING HINGE (learned, soft)
         L_ord = mean[ relu(p_in - p_m) + relu(p_m - p_out) ]
     -> the predicted regions inherit the same nesting
        {p_in>0.5} subseteq {p_m>0.5} subseteq {p_out>0.5}.

  3. SQUEEZE GAP (the "lim g = lim h" half)
         L_gap = mean[ 4 p_m (1 - p_m) * relu(p_out - p_in) ]
     -> wherever the middle head is UNDECIDED (p_m near 0.5, which is exactly
        where the decision boundary is), the two bounds are pushed onto each
        other.  The weight 4p(1-p) is a built-in window: it is 1 at p_m=0.5 and
        0 in the certain interior/background, so the pressure acts only on the
        undecided boundary layer and no mask has to be hand-made.

  4. RADIUS CURRICULUM  r(t) -> R_min  (see squeeze_config.radius)
     makes the two supervision targets converge onto GT itself.  At the limit
     the interval has zero width, Omega_in = Omega = Omega_out, so every point
     on the boundary is decided *by the collapse of the interval*, not by
     picking a threshold.  g = h = L  =>  f = L.

Consequences that are measurable (and are reported by tools/61_*):
    * ordering violation rate      -> 0
    * |Omega_out \\ Omega_in|        -> small (the interval closes)
    * |Omega_gt \\ Omega_out|        -> 0 (the outer bound really bounds)
    * |Omega_in \\ Omega_gt|         -> 0 (the inner bound really is inside)
    * boundary(GT) inside the predicted interval  -> high (the pinch contains
      the truth), and the residual error must live inside the interval, which
      turns the interval into a free uncertainty band.

Implementation notes
--------------------
* Head layout in every deep-supervision output of the network is
      channels [0:2] = main  (bg, fg)
      channels [2:4] = inner (bg, fg)
      channels [4:6] = outer (bg, fg)
  Each head is therefore a *stock* nnU-Net binary head, and the per-head
  objective is literally the same DC_and_CE_loss the baseline uses -> the main
  head alone is trained exactly like nnU-Net's, which keeps the comparison
  attributable to the squeeze machinery rather than to a loss swap.
* The bound targets are built on the GPU from the (already deep-supervision
  downsampled) ground-truth, so no extra dataset/preprocessing is needed and
  the same pipeline as step 1 is reused.  They are handed to the segmentation
  loss in nnU-Net's own target layout, a (B, 1, H, W) class-index map.
* Morphology is Chebyshev (square structuring element) and is done by iterated
  3x3 max-pooling, which needs no distance transform and no autograd.
* The r=0 limit is honoured: bounds which would be <1 px wide at a given
  deep-supervision scale are skipped there (they would carry no information),
  so the extra supervision fades out automatically as the curriculum closes.
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
    """depth_in(p) = largest d <= rmax such that p survives d erosions of gt.

    This is the L-infinity (Chebyshev) distance from p to the background,
    clamped at rmax: exactly the number that an erosion by radius r thresholds.
    """
    d = torch.zeros_like(gt)
    cur = gt
    for _ in range(int(rmax)):
        cur = _erode(cur, 1)
        d = d + cur
    return d


def chebyshev_depth_outside(gt: torch.Tensor, rmax: int) -> torch.Tensor:
    """depth_out(p) = min d <= rmax with p inside dilate^d(gt)

    (0 for foreground pixels; L-infinity distance to the foreground otherwise).
    """
    d = torch.zeros_like(gt)
    cur = gt                                  # cur = dilate^(k-1)(gt)
    for _ in range(int(rmax)):
        d = d + (1.0 - cur)                   # p not covered yet -> distance grows
        cur = _dilate(cur, 1)
    return d


def squeeze_bounds(gt: torch.Tensor, r_map: torch.Tensor, rmax: int):
    """The two bounds of the squeeze interval.

    gt    : (B, 1, H, W) float 0/1
    r_map : (B, 1, H, W) band half-width in *pixels of this resolution*
            (a constant, or a per-pixel value for the angular band)
    returns (inner, outer) with  inner <= gt <= outer  guaranteed pointwise.

    Exact discrete semantics, with Chebyshev (square) structuring element:

        erode_r(gt)  = { p : dist_cheb(p, background) >  r }     (so r < 1 is a no-op)
        dilate_r(gt) = { p : dist_cheb(p, foreground) <= r }     (likewise)

    `din`/`dout` are those distances minus 1 / unchanged, which is why the inner
    test is `din + 1 > r`.
    """
    din = chebyshev_depth_inside(gt, rmax)
    dout = chebyshev_depth_outside(gt, rmax)
    # `& gt` matters: din+1 > r is also true for *background* pixels when r < 1,
    # and the inner bound must never leave the foreground.
    inner = (((din + 1.0) > r_map) & (gt > 0.5)).to(gt.dtype)
    outer = (dout <= r_map).to(gt.dtype)
    return inner, outer


def sector_scale_map(gt: torch.Tensor, sector_scale, radius: float) -> torch.Tensor:
    """Per-pixel band radius  r * s(theta).

    `sector_scale` (S,) comes from the fingerprint's angular profile: it is
    1.0 where the ultrasound edge is average, >1 where it is weak (more room for
    the interval) and <1 where the edge is strong (the bound can be tighter).
    The angle is measured from the GT centroid in the SAME convention as
    step4's `sector_weight_map` and the fingerprint itself:
    +90 deg = image up = anterior, -90 deg = posterior (rectal wall).
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
    """Foreground mask from an nnU-Net target.

    nnU-Net 2.6.2 hands the trainer a *label map* of shape (B, 1, H, W) with
    values in {0, 1} (see DC_and_CE_loss: "target must be b, c, x, y with c=1"),
    and `DownsampleSegForDSTransform` keeps that layout at every deep-supervision
    scale.  A one-hot (B, 2, ...) target is accepted as well so the loss stays
    correct if the pipeline ever switches.
    """
    if target.dim() == 3:
        target = target[:, None]
    if target.shape[1] == 1:
        return (target[:, 0:1] > 0.5).to(torch.float32)
    return (target[:, 1:2] > 0.5).to(torch.float32)


def as_label_map(mask: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) float 0/1 -> the exact target layout DC_and_CE_loss expects.

    The bound targets are NOT one-hot encoded: the segmentation loss consumes
    (B, 1, ...) class-index maps and does its own one-hot conversion, so handing
    it a 2-channel tensor would silently invert the cross-entropy term.
    """
    return mask.to(torch.float32)


def softmax_fg(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1:2]


# ------------------------------------------------------------------- loss
class SqueezeLoss(nn.Module):
    """See the module docstring: 3 heads + ordering hinge + squeeze gap."""

    def __init__(self, dc_ce_factory, ds_weights, cfg, device=None):
        super().__init__()
        self.dc_ce_factory = dc_ce_factory
        self.loss_main = dc_ce_factory()
        self.loss_in = dc_ce_factory()
        self.loss_out = dc_ce_factory()
        self.ds_weights = [float(w) for w in ds_weights]
        self.cfg = cfg
        self.epoch = 0
        self.r = float(cfg.R0)
        self.last_diag = {}

    # -------------------------------------------------------------- helpers
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.r = float(self.cfg.radius(epoch))

    def _scale_stride(self, i: int) -> float:
        """Downsampling factor of deep-supervision scale i (1, 2, 4, ...)."""
        return float(2 ** i)

    # --------------------------------------------------------------- forward
    def forward(self, output, target):
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        tgts = list(target) if isinstance(target, (list, tuple)) else [target]
        assert len(outs) == len(tgts), "output/target deep-supervision mismatch"
        assert len(outs) == len(self.ds_weights), "deep-supervision weight mismatch"

        l_main = l_in = l_out = 0.0
        l_ord = l_gap = 0.0

        # ---- 1. the three heads, deep-supervision weighted exactly like stock
        for i, (o, t) in enumerate(zip(outs, tgts)):
            w = self.ds_weights[i]
            if w <= 0.0:
                continue
            t = t.to(o.device)
            l_main = l_main + w * self.loss_main(o[:, 0:2], t)

            r_s = self.r / self._scale_stride(i)
            if r_s < 1.0:
                # the interval would be narrower than one pixel at this scale:
                # nothing left to bound (this is the r -> 0 limit of the theorem)
                continue
            gt = fg_from_target(t)
            rmax = max(2, int(math.ceil(r_s)) + 1)
            if self.cfg.angular:
                r_map = sector_scale_map(gt, self.cfg.sector_scale, r_s)
            else:
                r_map = torch.full_like(gt, float(r_s))
            inner, outer = squeeze_bounds(gt, r_map, rmax)
            l_in = l_in + w * self.loss_in(o[:, 2:4], as_label_map(inner))
            l_out = l_out + w * self.loss_out(o[:, 4:6], as_label_map(outer))

        # ---- 2./3. ordering hinge and squeeze gap (full resolution only)
        o0 = outs[0]
        t0 = tgts[0].to(o0.device)
        p_m = softmax_fg(o0[:, 0:2])
        p_in = softmax_fg(o0[:, 2:4])
        p_out = softmax_fg(o0[:, 4:6])

        l_ord = (F.relu(p_in - p_m) + F.relu(p_m - p_out)).mean()
        w_mid = 4.0 * p_m * (1.0 - p_m)                      # 1 at the decision layer
        gap = F.relu(p_out - p_in)
        l_gap = (w_mid * gap).sum() / w_mid.sum().clamp_min(1e-6)

        total = (l_main
                 + self.cfg.w_in * l_in
                 + self.cfg.w_out * l_out
                 + self.cfg.lambda_order * l_ord
                 + self.cfg.lambda_gap * l_gap)

        self.last_diag = self._diagnostics(t0, p_in, p_m, p_out, l_main, l_in, l_out,
                                           l_ord, l_gap)
        return total

    # ---------------------------------------------------------- diagnostics
    @torch.no_grad()
    def _diagnostics(self, t0, p_in, p_m, p_out, l_main, l_in, l_out, l_ord, l_gap):
        """The measurable claims of the squeeze theorem, on the training batch.

        (Cheap enough for every iteration: a handful of 3x3 max-pools at the
        full patch resolution; used for the collapse curves in the report.)
        """
        gt = fg_from_target(t0)
        pred_in = (p_in > 0.5).to(gt.dtype)
        pred_out = (p_out > 0.5).to(gt.dtype)
        pred_m = (p_m > 0.5).to(gt.dtype)

        gt_sum = gt.sum().clamp_min(1.0)
        in_sum = pred_in.sum().clamp_min(1.0)
        # premise of the theorem: does the predicted interval really contain GT?
        outer_miss = ((gt - pred_out).clamp_min(0.0).sum() / gt_sum)      # GT outside outer
        inner_leak = ((pred_in - gt).clamp_min(0.0).sum() / in_sum)       # inner outside GT
        # the nesting of the predicted regions themselves
        nesting = (((pred_m - pred_out).clamp_min(0.0).sum()
                    + (pred_in - pred_m).clamp_min(0.0).sum()) / gt_sum)
        # is the true boundary inside the predicted interval (= the pinch)?
        bd_gt = (_dilate(gt, 1) - _erode(gt, 1)).clamp(0.0, 1.0)
        band = (pred_out * (1.0 - pred_in))
        bd_contain = (bd_gt * band).sum() / bd_gt.sum().clamp_min(1.0)

        return {
            "epoch": self.epoch,
            "r": float(self.r),
            # ordering violations: the *magnitude* is what matters.  A bare pixel
            # fraction is dominated by the far background, where all three heads
            # are ~0 and differ in the 4th decimal - a crossing there means
            # nothing, so the reported fraction only counts crossings > 0.1.
            "viol_mag": float(l_ord),
            "viol_big": float((((p_in - p_m) > 0.1) | ((p_m - p_out) > 0.1)).float().mean()),
            "gap": float(F.relu(p_out - p_in).mean()),
            "gap_mid": float(l_gap),
            "outer_miss": float(outer_miss),
            "inner_leak": float(inner_leak),
            "nesting": float(nesting),
            "bd_contain": float(bd_contain),
            "l_main": float(l_main),
            "l_in": float(l_in),
            "l_out": float(l_out),
            "l_ord": float(l_ord),
            "l_gap": float(l_gap),
        }

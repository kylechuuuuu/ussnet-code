#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/tools/60_selftest_squeeze_loss.py

CPU self-test of the squeeze-theorem machinery.  Runs in a few seconds, needs no
GPU, no dataset and no checkpoint.  Two halves:

  A. TENSOR MATH - the geometric primitives of the squeeze interval
       * Chebyshev depth (inside/outside) against hand-computed distances
       * inner <= GT <= outer for every radius, isotropic and angular
       * the r -> 0 limit: the interval collapses to GT (the theorem's limit)
       * the angular scale map against the fingerprint's convention
       * the loss: shapes, finiteness, gradients on all three heads, the
         ordering hinge is exactly 0 when ordered, and the gap term pushes the
         inner bound up / the outer bound down where the middle head is undecided

  B. AN IN-SILICO SQUEEZE - does the pinch actually determine a boundary?
       A synthetic 1-px-scale disc with a deliberately biased (2 px shifted)
       starting estimate, optimised as three free logit maps:
         arm P : main head only, stock Dice+CE           (the baseline analogue)
         arm S : all three heads, ordering + gap + annealed radius
       and then measures what the *theorem* promises:
         - ordering violation rate -> 0
         - interval width -> small (the bounds converge)
         - the true boundary lies inside the interval (the pinch contains truth)
         - the fused (projected median) boundary is closer to GT than the plain
           single-head decision of arm P
       This is a controlled statement of the mechanism; the real run repeats it
       on the ultrasound data.

usage:  ~/anaconda3/envs/zk/bin/python tools/60_selftest_squeeze_loss.py
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer"))

from squeeze_config import SqueezeConfig                                  # noqa: E402
from squeeze_losses import (as_label_map, chebyshev_depth_inside,        # noqa: E402
                            chebyshev_depth_outside, fg_from_target,
                            sector_scale_map, softmax_fg, squeeze_bounds,
                            SqueezeLoss)
from squeeze_net import SqueezeHeadUNet                                   # noqa: E402

FAIL = []


def check(name, cond, info=""):
    status = "ok  " if cond else "FAIL"
    print("[%s] %s %s" % (status, name, info))
    if not cond:
        FAIL.append(name)


def try_dc_ce():
    try:
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

        def factory():
            return DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False,
                                   'ddp': False}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)
        return factory, "nnU-Net DC_and_CE_loss"
    except Exception as e:                                     # pragma: no cover
        print("[warn] nnU-Net loss unavailable (%s) -> falling back to a small DiceCE" % e)

        class DiceCE(torch.nn.Module):
            def forward(self, x, t):
                p = torch.softmax(x, dim=1)[:, 1:2]
                g = t[:, 1:2].float()
                num = 2 * (p * g).sum() + 1e-5
                den = p.sum() + g.sum() + 1e-5
                return 1 - num / den - (g * torch.log_softmax(x, 1)[:, 1:2]).mean()
        return DiceCE, "fallback DiceCE"


# ======================================================================= A
def test_morphology():
    print("\n=== A1. Chebyshev depth / squeeze interval ===")
    # a 20x20 square, 4 px inside a 28x28 field
    gt = torch.zeros(1, 1, 28, 28)
    gt[0, 0, 4:24, 4:24] = 1.0
    din = chebyshev_depth_inside(gt, rmax=6)
    dout = chebyshev_depth_outside(gt, rmax=6)
    check("depth inside at the square centre == min(depth, rmax)",
          abs(float(din[0, 0, 14, 14]) - 6.0) < 1e-6, "got %.4f" % din[0, 0, 14, 14])
    check("depth inside on the square edge == 0 (an erosion by 1 already removes it)",
          float(din[0, 0, 4, 14]) == 0.0, "got %.4f" % din[0, 0, 4, 14])
    check("depth inside one pixel inside the edge == 1", float(din[0, 0, 5, 14]) == 1.0)
    check("depth outside just outside the square == 1", abs(float(dout[0, 0, 3, 14]) - 1.0) < 1e-6,
          "got %.4f" % dout[0, 0, 3, 14])
    check("depth outside inside the square == 0", float(dout[0, 0, 14, 14]) == 0.0)
    check("depth outside at the corner of the field == 4",
          abs(float(dout[0, 0, 0, 0]) - 4.0) < 1e-6, "got %.4f" % dout[0, 0, 0, 0])

    for r in [1.0, 2.0, 3.5, 6.0]:
        inner, outer = squeeze_bounds(gt, torch.full_like(gt, r), rmax=8)
        ok = bool(((inner <= gt).all()) and ((gt <= outer).all()))
        check("nesting inner <= GT <= outer at r=%.1f" % r, ok)
    inner, outer = squeeze_bounds(gt, torch.full_like(gt, 100.0), rmax=8)
    check("huge radius -> inner empty, outer = everything",
          float(inner.sum()) == 0.0 and float(outer.min()) == 1.0)
    inner, outer = squeeze_bounds(gt, torch.full_like(gt, 0.5), rmax=8)
    check("r < 1 px -> interval collapses onto GT (theorem limit)",
          bool(torch.equal(inner, gt)) and bool(torch.equal(outer, gt)))


def test_angular_map():
    print("\n=== A2. angular band (fingerprint profile) ===")
    gt = torch.zeros(1, 1, 64, 64)
    gt[0, 0, 22:42, 22:42] = 1.0                      # centroid at (32, 32)
    # 12 sectors, weakest at +75/+105 deg (like this dataset's fingerprint).
    # The band multiplier is the INVERSE of the edge strength: where the
    # ultrasound evidence is weak the interval is allowed to open.
    rel = [1.04, 0.91, 1.23, 1.29, 0.99, 1.00, 1.15, 0.87, 0.66, 0.65, 0.93, 1.28]
    mean = sum(rel) / len(rel)
    scale = [min(1.5, max(0.5, mean / r)) for r in rel]
    r = 6.0
    rmap = sector_scale_map(gt, scale, r)
    check("radius map has the shape of the mask", tuple(rmap.shape) == tuple(gt.shape))
    check("radius map is within the clipped multiplier range",
          float(rmap.min()) >= 0.5 * r - 1e-6 and float(rmap.max()) <= 1.5 * r + 1e-6,
          "range %.2f..%.2f" % (float(rmap.min()), float(rmap.max())))
    # an anterior pixel (straight above the centroid) must fall in sector idx 9 (+105 deg),
    # which in this dataset is the WEAKEST edge -> the widest band
    val_up = float(rmap[0, 0, 32 - 12, 32])
    check("weak anterior sector (+105 deg) gets the WIDEST band",
          val_up > 1.2 * r, "r_map(anterior)=%.2f (r=%.1f)" % (val_up, r))
    val_down = float(rmap[0, 0, 32 + 12, 32])
    check("strong sector at -105/+135.. gets a NARROWER band than the weak one",
          val_down < val_up, "posterior=%.2f vs anterior=%.2f" % (val_down, val_up))
    # angular profile orientation must survive the config path (real fingerprint)
    try:
        from squeeze_config import load_squeeze_config, DEFAULT_FP_JSON
        if os.path.isfile(DEFAULT_FP_JSON):
            cfg = load_squeeze_config()
            weakest = sorted(zip(cfg.sector_scale, cfg.sector_deg))[-1]
            check("config: widest multiplier lands on the fingerprint's weakest sector",
                  int(weakest[1]) in (75, 105),
                  "widest %.2f at %+d deg" % (weakest[0], int(weakest[1])))
            check("config: multiplier is inverse to edge strength (weak -> wide)",
                  weakest[0] > 1.2)
            sched = cfg.radius_schedule()
            check("config: radius curriculum is monotone and ends at R_min",
                  all(sched[i] >= sched[i + 1] - 1e-9 for i in range(len(sched) - 1))
                  and abs(sched[-1] - cfg.R_min) < 1e-9,
                  "r(0)=%.2f r(T)=%.2f" % (sched[0], sched[-1]))
        else:
            print("[warn] step-4 fingerprint not found, config orientation not checked")
    except Exception as e:
        check("config loads the fingerprint", False, str(e))
    # widening the band in the weak sectors must keep the containment property
    inner, outer = squeeze_bounds(gt, rmap, rmax=8)
    check("angular interval still nests inner <= GT <= outer",
          bool(((inner <= gt).all()) and ((gt <= outer).all())))
    mean_r = float(rmap.mean())
    check("mean radius is close to r (mean-preserving multiplier)",
          abs(mean_r - r) / r < 0.20, "mean %.2f vs r %.2f" % (mean_r, r))


def test_loss_and_grads():
    print("\n=== A3. loss, shapes, gradients, term behaviour ===")
    factory, src = try_dc_ce()
    print("       (per-head segmentation loss: %s)" % src)
    cfg = SqueezeConfig({"meta": {"n_train": 0},
                         "shape_statistics": {"mean_linear_scale_px": 280.6},
                         "angular_profile": {"edge_strength_rel": [],
                                             "sector_deg_centers": []}},
                        "selftest", angular_default=False)
    cfg.num_epochs = 20
    loss = SqueezeLoss(factory, [1.0, 0.5], cfg)
    b, h, w = 2, 64, 64
    # nnU-Net target layout: (B, 1, H, W) class-index map, NOT one-hot
    gt = torch.zeros(b, 1, h, w)
    gt[:, 0, 16:48, 16:48] = 1.0
    tgts = [gt, F.interpolate(gt, scale_factor=0.5, mode='nearest')]
    check("fg_from_target reads nnU-Net's label-map layout",
          float(fg_from_target(gt).sum()) == float(gt.sum()))
    outs = [torch.randn(b, 6, h, w, requires_grad=True),
            torch.randn(b, 6, h // 2, w // 2, requires_grad=True)]
    loss.set_epoch(6)
    total = loss(outs, tgts)
    check("loss is finite", math.isfinite(float(total)), "value %.4f" % float(total))
    total.backward()
    for name, sl in [("main", slice(0, 2)), ("inner", slice(2, 4)), ("outer", slice(4, 6))]:
        g = outs[0].grad[:, sl]
        check("gradient reaches the %s head" % name, float(g.abs().sum()) > 0)
    g2 = outs[1].grad
    check("gradient reaches the deep-supervision scales", float(g2.abs().sum()) > 0)

    # ordering hinge is exactly zero on a correctly ordered prediction
    o = torch.zeros(1, 6, 32, 32)
    o[0, 1] = 2.0                     # main fg logit
    o[0, 3] = 0.0                     # inner  (p = 0.5)  <= main
    o[0, 5] = 4.0                     # outer  (p ~ 0.98) >= main
    l_ord = (F.relu(softmax_fg(o[:, 2:4]) - softmax_fg(o[:, 0:2]))
             + F.relu(softmax_fg(o[:, 0:2]) - softmax_fg(o[:, 4:6]))).mean()
    check("ordering hinge == 0 when p_in <= p_m <= p_out", abs(float(l_ord)) < 1e-8)
    o2 = o.clone()
    o2[0, 3] = 6.0                    # inner flipped to the outside -> violation
    l_ord2 = (F.relu(softmax_fg(o2[:, 2:4]) - softmax_fg(o2[:, 0:2]))
              + F.relu(softmax_fg(o2[:, 0:2]) - softmax_fg(o2[:, 4:6]))).mean()
    check("ordering hinge > 0 on a violation", float(l_ord2) > 0.1, "%.4f" % float(l_ord2))

    # gap term direction: d/dinner < 0 (push inner up), d/douter > 0 (push outer down)
    # NOTE: start from an OPEN interval - relu(p_out - p_in) has no gradient where
    # the two bounds coincide exactly, which is why the model is initialised with
    # the inner/outer logits offset by head_bias instead of being tied.
    p_m = torch.full((1, 1, 16, 16), 0.5)
    w_mid = 4 * p_m * (1 - p_m)
    inner_logit = torch.full((1, 1, 16, 16), -1.0, requires_grad=True)
    outer_logit = torch.full((1, 1, 16, 16), 1.0, requires_grad=True)
    gap = F.relu(torch.sigmoid(outer_logit) - torch.sigmoid(inner_logit))
    (w_mid * gap).sum().backward()
    check("gap pushes the inner bound up (dinner < 0)",
          float(inner_logit.grad.mean()) < 0, "%.4f" % float(inner_logit.grad.mean()))
    check("gap pushes the outer bound down (douter > 0)",
          float(outer_logit.grad.mean()) > 0, "%.4f" % float(outer_logit.grad.mean()))


def test_fusion():
    print("\n=== A4. inference fusion (squeeze projection) ===")
    # ordered triple -> median == main; violated triple -> the bounds win
    logits = torch.zeros(1, 6, 1, 1)
    logits[0, 1] = 2.0                       # main p = 0.88
    logits[0, 3] = 0.0                       # inner p = 0.50
    logits[0, 5] = 4.0                       # outer p = 0.98
    p_main = float(torch.softmax(SqueezeHeadUNet.fuse(logits, "main")[:, 1:2], 1))
    p_med = float(torch.softmax(SqueezeHeadUNet.fuse(logits, "median")[:, 1:2], 1))
    check("fusion returns 2 channels", SqueezeHeadUNet.fuse(logits, "median").shape[1] == 2)
    check("ordered case: median == main", abs(p_main - p_med) < 1e-4,
          "%.4f vs %.4f" % (p_main, p_med))

    logits2 = torch.zeros(1, 6, 1, 1)
    logits2[0, 1] = -3.0                     # main says background (p = 0.05)
    logits2[0, 3] = 3.0                      # inner says foreground (p = 0.95)
    logits2[0, 5] = 4.0                      # outer says foreground
    med2 = float(torch.softmax(SqueezeHeadUNet.fuse(logits2, "median")[:, 1:2], 1))
    check("violated case: the bounds override the middle head", med2 > 0.9, "%.4f" % med2)


# ======================================================================= B
def disc(h, w, cy, cx, r):
    yy = torch.arange(h).view(h, 1).float()
    xx = torch.arange(w).view(1, w).float()
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).float()


def bdice(pred, gt, band=1):
    dil = F.max_pool2d(pred[None, None], 2 * band + 1, 1, band)
    ero = -F.max_pool2d(-pred[None, None], 2 * band + 1, 1, band)
    pb = (dil - ero).clamp(0, 1)
    dil = F.max_pool2d(gt[None, None], 2 * band + 1, 1, band)
    ero = -F.max_pool2d(-gt[None, None], 2 * band + 1, 1, band)
    gb = (dil - ero).clamp(0, 1)
    return float(2 * (pb * gb).sum() / (pb.sum() + gb.sum() + 1e-6))


def test_in_silico():
    print("\n=== B. in-silico squeeze: does the pinch determine the boundary? ===")
    torch.manual_seed(0)
    h = w = 128
    gt = disc(h, w, 64, 62, 34)                                   # "true" gland
    biased = disc(h, w, 64, 64, 34)                               # 2 px shifted estimate
    tgt = gt[None, None]                                          # (1,1,H,W) nnU-Net layout
    factory, _ = try_dc_ce()
    scales = [1.0, 0.5, 0.25]

    def ds_targets(t, n=3):
        return [F.interpolate(t, scale_factor=s, mode='nearest') for s in scales[:n]]

    def init_logit(mask, shift=0.0):
        p = (mask * 0.9 + 0.05).clamp(1e-4, 1 - 1e-4)
        return torch.logit(p)[None, None] + shift

    def run(arm, outer_iters=60, inner_steps=12):
        md = torch.nn.Parameter(init_logit(biased))
        ii = torch.nn.Parameter(init_logit(biased, -1.0))          # interval starts OPEN
        oo = torch.nn.Parameter(init_logit(biased, +1.0))
        params = [md] if arm == "P" else [md, ii, oo]
        opt = torch.optim.Adam(params, lr=0.05)
        cfg = SqueezeConfig({"meta": {}, "shape_statistics": {}, "angular_profile": {}},
                            "selftest", angular_default=False)
        cfg.num_epochs = outer_iters
        loss = SqueezeLoss(factory, [0.6, 0.3, 0.1], cfg)
        for it in range(outer_iters):
            loss.set_epoch(it)
            for _ in range(inner_steps):
                opt.zero_grad()
                if arm == "P":
                    # baseline analogue: only the middle head exists, stock objective
                    l = factory()(torch.cat([-md, md], 1), tgt)
                else:
                    ds_out = [torch.cat([torch.cat([-md, md], 1),
                                         torch.cat([-ii, ii], 1),
                                         torch.cat([-oo, oo], 1)], 1)]
                    for s in scales[1:]:
                        ds_out.append(F.interpolate(ds_out[0], scale_factor=s, mode='area'))
                    l = loss(ds_out, ds_targets(tgt))
                l.backward()
                opt.step()
        with torch.no_grad():
            p_m = torch.sigmoid(md)[0, 0]
            p_i = torch.sigmoid(ii)[0, 0]
            p_o = torch.sigmoid(oo)[0, 0]
            med = torch.sort(torch.stack([p_i, p_m, p_o]), 0).values[1]
            return dict(p_m=p_m, p_i=p_i, p_o=p_o, med=med)

    resP = run("P")
    resS = run("S")
    gt_b = gt
    stats = {}
    for tag, p in [("P", resP["p_m"]), ("S", resS["med"]), ("S-main", resS["p_m"])]:
        pred = (p > 0.5).float()
        dice = float(2 * (pred * gt_b).sum() / (pred.sum() + gt_b.sum()))
        stats[tag] = dict(dice=dice, bdice=bdice(pred, gt_b),
                          area_px=float((pred - gt_b).abs().sum()))
    viol = float(((resS["p_i"] > resS["p_m"]).float() + (resS["p_m"] > resS["p_o"]).float()).mean())
    gap = float((resS["p_o"] - resS["p_i"]).mean())
    band = ((resS["p_o"] > 0.5) & (resS["p_i"] <= 0.5)).float()
    bd_gt = bdice_mask = ((F.max_pool2d(gt_b[None, None], 3, 1, 1)
                           - (-F.max_pool2d(-gt_b[None, None], 3, 1, 1)))[0, 0])
    contain = float((bd_gt * band).sum() / (bd_gt.sum() + 1e-6))
    outer_miss = float(((gt_b - (resS["p_o"] > 0.5).float()).clamp_min(0)).sum() / gt_b.sum())
    inner_leak = float(((resS["p_i"] > 0.5).float() - gt_b).clamp_min(0).sum()
                       / (resS["p_i"] > 0.5).float().sum())

    print("        arm P (main head only)    Dice %.4f | bDice@1 %.4f | area err %d px"
          % (stats["P"]["dice"], stats["P"]["bdice"], stats["P"]["area_px"]))
    print("        arm S (main head)         Dice %.4f | bDice@1 %.4f | area err %d px"
          % (stats["S-main"]["dice"], stats["S-main"]["bdice"], stats["S-main"]["area_px"]))
    print("        arm S (squeeze median)    Dice %.4f | bDice@1 %.4f | area err %d px"
          % (stats["S"]["dice"], stats["S"]["bdice"], stats["S"]["area_px"]))
    print("        interval: viol %.4f | mean width %.4f | GT outside outer %.4f | "
          "inner outside GT %.4f | true boundary inside interval %.3f"
          % (viol, gap, outer_miss, inner_leak, contain))

    check("in-silico: ordering violations vanish (< 2 %)", viol < 0.02, "%.4f" % viol)
    check("in-silico: interval closes (mean width < 0.15)", gap < 0.15, "%.4f" % gap)
    check("in-silico: outer bound contains GT (> 99 %)", outer_miss < 0.01, "%.4f" % outer_miss)
    check("in-silico: inner bound inside GT (> 99 %)", inner_leak < 0.01, "%.4f" % inner_leak)
    check("in-silico: true boundary inside the predicted interval (> 90 %)",
          contain > 0.90, "%.3f" % contain)
    check("in-silico: the squeeze fixes the 2 px bias (area error < 60 % of arm P)",
          stats["S"]["area_px"] < 0.6 * stats["P"]["area_px"],
          "%d vs %d px" % (stats["S"]["area_px"], stats["P"]["area_px"]))


if __name__ == "__main__":
    test_morphology()
    test_angular_map()
    test_loss_and_grads()
    test_fusion()
    test_in_silico()
    print("\n==================================================")
    if FAIL:
        print("SELFTEST FAILED (%d): %s" % (len(FAIL), ", ".join(FAIL)))
        sys.exit(1)
    print("SELFTEST OK - squeeze machinery behaves as designed")

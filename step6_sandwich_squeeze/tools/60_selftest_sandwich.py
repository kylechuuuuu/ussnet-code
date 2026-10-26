#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/tools/60_selftest_sandwich.py

CPU self-test for the step-6 sandwich squeeze machinery (no GPU, no training):

  1. targets      :  cautious ⊆ GT ⊆ amplified   (fixed radii)
  2. angular map  :  two-level sector scaling behaves as documented
  3. sandwich loss:  forward + backward, all diagnostics keys, finite grads
  4. network      :  6-channel heads, tied heads, forward modes
                     all / base / cautious / amplified
  5. arbitration  :  the interval clamp really bounds the movement (synthetic
                     ridge; needs an f32 CAPABLE interpreter, runs on CPU)
"""
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STEP6 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(STEP6, "trainer"))

from sandwich_config import SandwichConfig                                     # noqa: E402
from sandwich_losses import (chebyshev_depth_inside, chebyshev_depth_outside,  # noqa: E402
                             sector_scale_map, SandwichLoss)
from sandwich_net import SandwichHeadUNet, tie_seg_heads                       # noqa: E402

OK = True


def check(name, cond, extra=""):
    global OK
    tag = "OK " if cond else "FAIL"
    if not bool(cond):
        OK = False
    print("[60] %s %-44s %s" % (tag, name, extra))


def ring(h=96, w=120, r=30, cy=48, cx=60):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)


def main():
    cfg = SandwichConfig({"angular_profile": {}}, "selftest")
    gt = torch.from_numpy(ring())[None, None]

    # ---- 1. bound targets ---------------------------------------------------
    rmax = 10
    din = chebyshev_depth_inside(gt, rmax)
    dout = chebyshev_depth_outside(gt, rmax)
    cautious = (((din + 1.0) > cfg.r_c) & (gt > 0.5)).float()
    amplified = (dout <= cfg.r_a).float()
    check("cautious ⊆ GT", bool((cautious <= gt).all()))
    check("GT ⊆ amplified", bool((gt <= amplified).all()))
    check("cautious non-empty", float(cautious.sum()) > 50,
          "px=%d" % int(cautious.sum()))
    check("amplified non-empty", float(amplified.sum()) > float(gt.sum()))
    ratio_c = float(cautious.sum() / gt.sum())
    ratio_a = float(amplified.sum() / gt.sum())
    check("cautious area ratio sane", 0.65 < ratio_c < 0.95, "%.2f" % ratio_c)
    check("amplified area ratio sane", 1.10 < ratio_a < 1.50, "%.2f" % ratio_a)

    # ---- 2. angular map ------------------------------------------------------
    cfg2 = SandwichConfig({"angular_profile": {
        "dir_sector_weights": [1.0] * 4 + [1.5] * 4,
        "sector_deg_centers": [0, 45, 90, 135, 180, 225, 270, 315]}}, "selftest")
    m = sector_scale_map(gt, cfg2.sector_scale, 3.0)
    vals = sorted({round(float(v), 3) for v in torch.unique(m).tolist()})
    check("angular map two levels", vals == [2.4, 3.6], str(vals))

    # ---- 3. sandwich loss ------------------------------------------------------
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

    def factory():
        return DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False,
                               'ddp': False}, {}, weight_ce=1, weight_dice=1,
                              ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)

    loss = SandwichLoss(factory, [1.0], cfg)
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 96, 120, requires_grad=True)
    target = torch.cat([gt, gt], 0)                            # (2,1,96,120)
    out = loss(logits, target)
    check("loss finite", bool(torch.isfinite(out)), "%.3f" % float(out))
    out.backward()
    check("gradients finite", bool(torch.isfinite(logits.grad).all()))
    d = loss.last_diag
    need = {"caut_leak", "amp_miss", "viol_in", "viol_out", "bd_contain",
            "width_px", "dice_c", "dice_a", "l_base", "l_c", "l_a"}
    check("diagnostics keys", need <= set(d.keys()), str(sorted(d.keys())))
    check("diagnostics finite", all(np.isfinite(float(v)) for v in d.values()))

    # ---- 4. network -------------------------------------------------------------
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    kw = {"n_stages": 4, "features_per_stage": [4, 8, 16, 32],
          "conv_op": "torch.nn.modules.conv.Conv2d",
          "kernel_sizes": [[3, 3]] * 4, "strides": [[1, 1]] + [[2, 2]] * 3,
          "n_conv_per_stage": [1, 1, 1, 1], "n_conv_per_stage_decoder": [1, 1, 1],
          "conv_bias": True, "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
          "norm_op_kwargs": {"eps": 1e-05, "affine": True},
          "dropout_op": None, "dropout_op_kwargs": None,
          "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True}}

    def build(ds):
        net = get_network_from_plans(
            "dynamic_network_architectures.architectures.unet.PlainConvUNet",
            kw, ["conv_op", "norm_op", "dropout_op", "nonlin"], 1, 6,
            deep_supervision=ds)
        tie_seg_heads(net, 1.0)
        return net

    x = torch.zeros(1, 1, 64, 64)
    w1 = SandwichHeadUNet(build(False), forward_mode="base")
    o = w1(x)
    check("mode base -> 2ch", tuple(o.shape) == (1, 2, 64, 64), str(tuple(o.shape)))
    w1.set_forward_mode("cautious")
    check("mode cautious -> 2ch", tuple(w1(x).shape) == (1, 2, 64, 64))
    w1.set_forward_mode("amplified")
    check("mode amplified -> 2ch", tuple(w1(x).shape) == (1, 2, 64, 64))
    w2 = SandwichHeadUNet(build(True), forward_mode="all")
    o2 = w2(x)
    check("mode all -> DS list, 6ch",
          isinstance(o2, (list, tuple)) and o2[0].shape[1] == 6,
          str([tuple(t.shape) for t in o2][:2]))
    conv6 = [mm for mm in w2.modules()
             if isinstance(mm, torch.nn.Conv2d) and mm.out_channels == 6
             and mm.kernel_size == (1, 1)][0]
    with torch.no_grad():
        b0, b1, b3, b5 = (float(conv6.bias[0]), float(conv6.bias[1]),
                          float(conv6.bias[3]), float(conv6.bias[5]))
    check("heads tied with +/- bias",
          abs(b3 - (b1 - 1.0)) < 1e-6 and abs(b5 - (b1 + 1.0)) < 1e-6,
          "b1=%.2f b3=%.2f b5=%.2f" % (b1, b3, b5))

    # ---- 5. arbitration clamp ----------------------------------------------------
    spec = importlib.util.spec_from_file_location(
        "arb63", os.path.join(STEP6, "tools", "63_arbitrate.py"))
    arb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arb)
    from scipy import ndimage
    from skimage import morphology

    m0 = ring().astype(bool)
    c = arb.s57.contour_of(m0)
    pts = arb.s57.resample_closed(c, 200, 0)
    nrm = arb.s58.normals_of(pts, m0)
    Min = ndimage.binary_erosion(m0, morphology.disk(3))
    Mout = ndimage.binary_dilation(m0, morphology.disk(4))
    t_in, t_out = arb.limits(pts, nrm, Min, Mout)
    check("inward room ≈ 3 px", 2.0 <= float(np.mean(t_in)) <= 4.0,
          "mean %.2f" % float(np.mean(t_in)))
    check("outward room ≈ 4 px", 3.0 <= float(np.mean(t_out)) <= 5.0,
          "mean %.2f" % float(np.mean(t_out)))
    G = np.zeros((len(pts), len(arb.TS)))
    G[:, int(np.argmin(np.abs(arb.TS - 3.0)))] = 1.0
    t = arb.pick(G, arb.TS, 0.7, -t_in, t_out)
    check("ridge at +3 is taken", abs(float(np.mean(t)) - 3.0) < 0.3,
          "mean %.2f" % float(np.mean(t)))
    G2 = np.zeros_like(G)
    G2[:, int(np.argmin(np.abs(arb.TS - 7.0)))] = 1.0
    t2 = arb.pick(G2, arb.TS, 0.7, -t_in, t_out)
    check("ridge at +7 is refused (out of interval)",
          float(np.max(np.abs(t2))) <= 1e-9, "max|t| %.2f" % float(np.max(np.abs(t2))))

    print("[60] %s" % ("ALL OK" if OK else "FAILURES PRESENT"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())

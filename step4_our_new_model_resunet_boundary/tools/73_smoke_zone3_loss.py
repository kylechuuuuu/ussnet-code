#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU smoke test for the 3-class path of the fingerprint loss (arm E).

Checks, with no GPU and no data:

  1. `fg_prob` is *identical* to the old binary expression when there are two
     classes -- i.e. the generalisation cannot have changed arms A/B/C/D;
  2. for three classes it really returns the union `1 - P(background)`;
  3. `FPCompoundLoss` (the real, fingerprint-driven objective) runs forward and
     backward on a 3-class batch and produces finite loss and finite grads;
  4. `fg_from_target` on an integer (B,1,X,Y) 3-class target yields the union,
     which is what the boundary terms assume.

Run:  python tools/73_smoke_zone3_loss.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "trainer"))

from fp_boundary_losses import FPCompoundLoss, fg_from_target, fg_prob  # noqa: E402
from fp_config import load_fp_config                                    # noqa: E402

torch.manual_seed(0)


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("ok  " if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    return ok


def main():
    ok = True
    B, H, W = 2, 64, 64

    logits2 = torch.randn(B, 2, H, W)
    old = torch.softmax(logits2, dim=1)[:, 1:2]
    ok &= check("fg_prob == softmax[:,1:2] for 2 classes",
                torch.allclose(fg_prob(logits2), old, atol=0, rtol=0),
                "max|diff| = %.3g" % (fg_prob(logits2) - old).abs().max().item())

    logits3 = torch.randn(B, 3, H, W)
    p3 = torch.softmax(logits3, dim=1)
    ok &= check("fg_prob == 1 - P(bg) for 3 classes",
                torch.allclose(fg_prob(logits3), 1.0 - p3[:, 0:1], atol=1e-6))

    # a 3-class integer label map, like nnU-Net hands the loss
    tgt = torch.zeros(B, 1, H, W, dtype=torch.float32)
    tgt[0, 0, : H // 2, : W // 2] = 1
    tgt[0, 0, H // 2 :, W // 2 :] = 2
    tgt[1, 0, :, : W // 2] = 2
    ok &= check("fg_from_target(int 3-class) == union",
                torch.equal(fg_from_target(tgt),
                            (tgt > 0.5).to(torch.float32)))

    cfg = load_fp_config()
    cfg.use_direction = True                     # exercise the sector weights
    loss = FPCompoundLoss(
        base_loss=torch.nn.CrossEntropyLoss(),
        cfg=cfg, is_full_resolution=True)
    loss.set_epoch(999)                          # multiplier = 1.0

    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

    for ncls, lg in ((2, logits2.clone().requires_grad_(True)),
                     (3, logits3.clone().requires_grad_(True))):
        base = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False,
                               'ddp': False}, {}, weight_ce=1, weight_dice=1,
                              dice_class=MemoryEfficientSoftDiceLoss)
        obj = FPCompoundLoss(base, cfg, is_full_resolution=True)
        obj.set_epoch(999)
        t = (tgt == 1).float() if ncls == 2 else tgt
        val = obj(lg, t)
        val.backward()
        finite_g = bool(torch.isfinite(lg.grad).all())
        finite_l = bool(torch.isfinite(val).all())
        ok &= check("%d-class: finite loss and grads" % ncls,
                    finite_l and finite_g,
                    "loss=%.4f  max|grad|=%.3g" % (val.item(), lg.grad.abs().max()))
        # the union probability must actually receive gradient in the 3-class case
        if ncls == 3:
            union = fg_prob(lg)
            union.sum().backward(retain_graph=True)
            active = int((lg.grad.abs().sum(dim=1) > 0).sum())
            ok &= check("3-class: every pixel receives gradient",
                        active == B * H * W, "%d/%d" % (active, B * H * W))

    print("SMOKE %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

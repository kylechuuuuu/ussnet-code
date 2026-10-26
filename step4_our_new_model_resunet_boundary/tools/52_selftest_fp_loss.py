#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast CPU self-test of the fingerprint loss stack (no nnU-Net, no GPU).

Checks: fingerprint loads, every term is finite, shapes work with and without
deep-supervision lists, gradients reach the logits, and the ramp behaves.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer"))

import torch
import torch.nn.functional as F

from fp_config import load_fp_config
from fp_boundary_losses import (FPCompoundLoss, TolerantBoundaryDice,
                                SharpBoundaryBCE, SoftPerimeterLoss,
                                boundary_band, sector_weight_map)

torch.manual_seed(0)


class DSStub(torch.nn.Module):
    """Stands in for nnU-Net's DeepSupervisionWrapper (accepts lists)."""

    def forward(self, outs, tgts):
        if isinstance(outs, (list, tuple)):
            return sum(F.cross_entropy(o, t.long()) for o, t in zip(outs, tgts))
        return F.cross_entropy(outs, tgts.long())


def main():
    fp = load_fp_config()
    for line in fp.summary_lines():
        print(line)
    print()

    B, C, X, Y = 2, 2, 128, 160
    logits = torch.randn(B, C, X, Y, requires_grad=True)
    target = torch.zeros(B, X, Y, dtype=torch.long)
    target[0, 30:90, 40:110] = 1
    target[1, 40:100, 30:120] = 1

    tl = TolerantBoundaryDice(band=fp.loss["tolerant_band"])(logits, target)
    sb = SharpBoundaryBCE(beta=fp.loss["bce_beta"], band=fp.loss["bce_band"])(logits, target)
    pl = SoftPerimeterLoss()(logits, target)
    print("tolerant bdice %.4f | sharp bce %.4f | length ratio %.4f" % (tl, sb, pl))
    assert torch.isfinite(tl) and torch.isfinite(sb) and torch.isfinite(pl)

    w = torch.tensor(fp.loss["dir_sector_weights"], dtype=torch.float32)
    sb2 = SharpBoundaryBCE(beta=fp.loss["bce_beta"], band=fp.loss["bce_band"],
                           dir_weights=w)(logits, target)
    print("sharp bce + direction weights %.4f" % sb2)
    assert torch.isfinite(sb2)

    gt = target[0:1, None].float()
    per = 2 * (60 + 70)
    print("hard-mask perimeter ~%d px" % per)
    for k in (3, 5, 7, 9):
        a = float(boundary_band(gt, k).sum())
        print("  band k=%d -> %5d px  (~%+.1f px around the boundary)"
              % (k, int(a), (a / per - 1) / 2))
    assert float(boundary_band(gt, 7).sum()) > float(boundary_band(gt, 3).sum())

    # length ratio must be ~1 when the prediction already matches the GT
    gt2 = target[:, None].float()
    logits_gt = torch.cat([(1 - gt2), gt2], dim=1) * 20.0 - 10.0
    lr_perfect = float(SoftPerimeterLoss()(logits_gt, target))
    # and >1 when the boundary is too long (jagged): add noise to the logits
    logits_jag = torch.cat([(1 - gt2), gt2], dim=1) * 20.0 - 10.0
    logits_jag[:, 1:2] += 6.0 * torch.randn_like(gt2)
    lr_jag = float(SoftPerimeterLoss()(logits_jag, target))
    print("length ratio: GT-like %.3f  |  jittered %.3f" % (lr_perfect, lr_jag))
    assert 0.5 < lr_perfect < 2.0, "length ratio should be ~1 for a matching boundary"
    assert lr_jag > lr_perfect

    sm = sector_weight_map(gt, w)
    print("sector weight map: min %.3f mean %.3f max %.3f"
          % (float(sm.min()), float(sm.mean()), float(sm.max())))
    assert sm.shape == gt.shape

    fp.use_direction = True
    comp = FPCompoundLoss(DSStub(), fp, is_full_resolution=True)
    comp.set_epoch(50)
    print("multiplier at epoch 50 / warmup %d = %.3f" % (fp.loss["warmup_epochs"],
                                                         comp.multiplier))
    assert 0.0 < comp.multiplier < 1.0
    comp.set_epoch(fp.loss["warmup_epochs"] + 5)
    assert comp.multiplier == 1.0

    out_list = [logits, torch.randn(B, C, X // 2, Y // 2, requires_grad=True)]
    tgt_list = [target, torch.zeros(B, X // 2, Y // 2, dtype=torch.long)]
    loss = comp(out_list, tgt_list)
    loss.backward()
    g = logits.grad
    print("composite (deep-supervision list) loss %.4f | grad finite %s | grad nonzero %d"
          % (float(loss), bool(torch.isfinite(g).all()), int((g.abs() > 0).sum())))
    assert torch.isfinite(loss) and torch.isfinite(g).all() and (g.abs() > 0).any()

    # plain (non-list) form must work too
    logits.grad = None
    loss_plain = comp(logits, target)
    loss_plain.backward()
    assert torch.isfinite(loss_plain) and (logits.grad.abs() > 0).any()
    print("composite (plain tensor)    loss %.4f  ok" % float(loss_plain))

    # ablation switch
    fp.disabled = True
    comp2 = FPCompoundLoss(DSStub(), fp, is_full_resolution=True)
    comp2.set_epoch(400)
    assert comp2.multiplier == 0.0
    assert torch.allclose(comp2(out_list, tgt_list), DSStub()(out_list, tgt_list))
    print("ablation switch (STEP4_DISABLE_EXTRA) ok")
    print("\nall loss self-tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

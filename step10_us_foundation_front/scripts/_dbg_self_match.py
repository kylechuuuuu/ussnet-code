#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/_dbg_self_match.py

Plumbing sanity check for 10_l0_probe.py.

A query image that is ITSELF in the reference bank must come back with a
near-perfect mask at k=1 (nearest neighbour == itself).  If the self-match is
good but the cross-patient val number is poor, the plumbing is fine and the
number is a property of the frozen features; if the self-match is bad, the
probe itself is broken.
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l0", os.path.join(HERE, "10_l0_probe.py"))
l0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l0)

model = sys.argv[1] if len(sys.argv) > 1 else "ultrafedfm"
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda:0")
cw, ch = l0.CANVAS
stride = 8
gh, gw = ch // stride, cw // stride

enc = l0.build_encoder(model, dev, l0.CANVAS)
train_names, val_names = l0.eval_lib.read_split(l0.USS)
ref_names = train_names[:12]

feats, labs = [], []
for n in ref_names:
    img, gt_c = l0.load_case(n, "in_domain")
    f = enc.encode(img)
    feats.append(F.interpolate(f[None], size=(gh, gw), mode="bilinear", align_corners=False)[0])
    labs.append(torch.from_numpy(l0.grid_labels(gt_c, stride).astype(np.float32)))
feat_ref = torch.stack(feats).to(dev)
lab_ref = torch.stack(labs).to(dev)
print("[dbg] bank %s label %s  fg %.4f" % (tuple(feat_ref.shape), tuple(lab_ref.shape), float(lab_ref.mean())))

print("[dbg] self-match (query is in the bank, k=1):")
for n in ref_names[:3]:
    img, gt_c = l0.load_case(n, "in_domain")
    pred = l0.predict_mask(feat_ref, lab_ref, enc.encode(img), stride, k=1)
    m = l0.eval_lib.all_metrics(pred, gt_c, None)
    print("   %-28s dice %6.2f  bdice@1 %5.2f  pred fg %.4f gt fg %.4f"
          % (n[:26], m["dice"], float(l0.s57.bdice(pred, gt_c, 1)), pred.mean(), gt_c.mean()))

print("[dbg] held-out val queries (k=5):")
for n in val_names[:3]:
    img, gt_c = l0.load_case(n, "in_domain")
    pred = l0.predict_mask(feat_ref, lab_ref, enc.encode(img), stride, k=5)
    m = l0.eval_lib.all_metrics(pred, gt_c, None)
    print("   %-28s dice %6.2f  bdice@1 %5.2f  pred fg %.4f gt fg %.4f"
          % (n[:26], m["dice"], float(l0.s57.bdice(pred, gt_c, 1)), pred.mean(), gt_c.mean()))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the learned normal refiner (tools/60) to the 31 fold-0 validation
images and score it against the raw prediction, the resample round-trip control,
and the §5.2 oracle.

Protocol identical to §5.1 / §5.2 / §5.5: cc1 of the prediction, 400 arc-length
points, outward normals, exact even-odd re-rasterisation, shared boundary-Dice
definition, so the numbers sit directly next to

    U-Net   raw 93.01 / 23.25   nnU-Net raw 96.11 / 35.16
    §5.2 oracle: U-Net 96.02 / 48.95   nnU-Net 98.69 / 84.91
    §5.5 classical snap: U-Net 27.59   nnU-Net 36.59
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS = os.path.dirname(STEP4)

spec = importlib.util.spec_from_file_location(
    "shape57", os.path.join(HERE, "57_shape_prior_ceiling.py"))
s57 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s57)
spec58 = importlib.util.spec_from_file_location(
    "refine58", os.path.join(HERE, "58_normal_refine_oracle.py"))
s58 = importlib.util.module_from_spec(spec58)
spec58.loader.exec_module(s58)
spec60 = importlib.util.spec_from_file_location(
    "refiner60", os.path.join(HERE, "60_train_normal_refiner.py"))
s60 = importlib.util.module_from_spec(spec60)
spec60.loader.exec_module(s60)

NPT = s60.NPT
MAXD = s60.MAXD
VH, VW = s60.VH, s60.VW
NV, NU = s60.NV, s60.NU


def featurize(pts, nrm, z, g):
    n = len(pts)
    tan = np.stack([nrm[:, 1], -nrm[:, 0]], axis=1)
    vs = np.arange(-VH, VH + 1, dtype=float)
    us = np.arange(-VW, VW + 1, dtype=float)
    V, U = np.meshgrid(vs, us, indexing="ij")
    q = (pts[:, None, None, :]
         + V[None, :, :, None] * nrm[:, None, None, :]
         + U[None, :, :, None] * tan[:, None, None, :])
    rr = q[..., 0].reshape(n, -1)
    cc = q[..., 1].reshape(n, -1)
    pz = ndimage.map_coordinates(z, [rr.ravel(), cc.ravel()], order=1,
                                 mode="nearest").reshape(n, NV, NU)
    pg = ndimage.map_coordinates(g, [rr.ravel(), cc.ravel()], order=1,
                                 mode="nearest").reshape(n, NV, NU)
    patches = np.stack([pz, pg], axis=1).astype(np.float32)

    d1 = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    d2 = np.roll(pts, -2, axis=0) - 2 * pts + np.roll(pts, 2, axis=0)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    kappa = np.abs(cross) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-6)
    cen = pts.mean(axis=0)
    R = float(np.sqrt(((pts - cen) ** 2).sum(axis=1).mean()) + 1e-6)
    rnorm = np.linalg.norm(pts - cen, axis=1) / R
    arc = np.arange(n) / float(n) * 2.0 - 1.0
    scal = np.stack([np.clip(kappa * 40.0, 0, 5), rnorm, arc,
                     s60.bil(z, pts), s60.bil(g, pts),
                     np.full(n, R / 280.0)], axis=1).astype(np.float32)
    return patches, scal


def blen(m):
    return float(s57.surface(m).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(STEP4, "results", "normal_refiner.pt"))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--smooth", type=float, default=2.0,
                    help="tangential smoothing of the predicted displacement")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(STEP4, "results",
                                                  "normal_refiner_eval.json"))
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu")
    net = s60.Refiner()
    net.load_state_dict(ck["state"])
    net.eval()
    print("[eval] loaded %s  (%d params)" % (args.model, ck.get("nparam", -1)))

    _, val = s57.read_split(USS, exp="ultrasound_prostate_73")
    if args.max:
        val = val[:args.max]

    models = {
        "unet": os.path.join(USS, "unet", "results", "ultrasound_prostate_73"),
        "nnunet_best": os.path.join(USS, "step1_nnunet_baseline", "predictions",
                                    "nnunet_2d_fold0_checkpoint_best_png"),
    }

    rows = []
    with torch.no_grad():
        for nm in val:
            gt = s57.load_gt(nm)
            img0 = s60.load_image(nm)
            if gt is None or img0 is None:
                continue
            z, g = s60.prep_image(img0)
            sp = s58.sdf_of(gt)
            for tag, d in models.items():
                pred = s57.load_pred(os.path.join(
                    d, nm if nm.lower().endswith(".png") else nm + ".png"))
                if pred is None or pred.shape != gt.shape:
                    continue
                cc1 = s57.largest_cc(pred)
                c = s57.contour_of(cc1)
                if c is None:
                    continue
                rec = dict(name=nm, model=tag,
                           dice_raw=s57.dice(pred, gt), bd1_raw=s57.bdice(pred, gt, 1),
                           bd2_raw=s57.bdice(pred, gt, 2), blen_raw=blen(pred))
                base_area = float(cc1.sum())

                # oracle one-step reference on the same start
                pts0 = s57.resample_closed(c, NPT, 0)
                nrm0 = s58.normals_of(pts0, cc1)
                t_or = np.clip(-s60.bil(sp.astype(np.float32), pts0), -MAXD, MAXD)
                m_or = s57.rasterize(pts0 + t_or[:, None] * nrm0, gt.shape)
                rec["dice_oracle1"] = s57.dice(m_or, gt)
                rec["bd1_oracle1"] = s57.bdice(m_or, gt, 1)
                rec["bd2_oracle1"] = s57.bdice(m_or, gt, 2)
                rec["blen_oracle1"] = blen(m_or)

                pts = pts0.copy()
                mask = cc1
                for it in range(args.iters + 1):
                    key = "ctrl" if it == 0 else "r%d" % it
                    if it > 0:
                        nrm = s58.normals_of(pts, mask)
                        p, s = featurize(pts, nrm, z, g)
                        dv = net(torch.from_numpy(p),
                                 torch.from_numpy(s)).numpy() * args.gain
                        dv = np.clip(dv, -MAXD, MAXD)
                        if args.smooth > 0:
                            dv = ndimage.gaussian_filter1d(dv, args.smooth, mode="wrap")
                        pts = pts + dv[:, None] * nrm
                        m_new = s57.rasterize(pts, gt.shape)
                        if m_new.sum() <= 0 or not (0.5 * base_area <= m_new.sum()
                                                    <= 2.0 * base_area):
                            break
                        mask = m_new
                    else:
                        mask = s57.rasterize(pts, gt.shape)
                    rec["dice_" + key] = s57.dice(mask, gt)
                    rec["bd1_" + key] = s57.bdice(mask, gt, 1)
                    rec["bd2_" + key] = s57.bdice(mask, gt, 2)
                    rec["blen_" + key] = blen(mask)
                rows.append(rec)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)

    keys = ["ctrl"] + ["r%d" % i for i in range(1, args.iters + 1)] + ["oracle1"]
    print("\n[eval] n=%d val images, smooth=%.1f gain=%.2f\n" % (len(val), args.smooth, args.gain))
    for tag in models:
        sel = [r for r in rows if r["model"] == tag]
        if not sel:
            continue
        print("== %s (n=%d) ==" % (tag, len(sel)))
        print("%-9s %8s %8s %8s %9s" % ("variant", "Dice", "bd1px", "bd2px", "bnd_len"))
        print("%-9s %8.2f %8.2f %8.2f %9.1f" %
              ("raw", np.mean([r["dice_raw"] for r in sel]),
               np.mean([r["bd1_raw"] for r in sel]),
               np.mean([r["bd2_raw"] for r in sel]),
               np.mean([r["blen_raw"] for r in sel])))
        for k in keys:
            if not any("dice_" + k in r for r in sel):
                continue
            print("%-9s %8.2f %8.2f %8.2f %9.1f" %
                  (k, np.mean([r["dice_" + k] for r in sel]),
                   np.mean([r["bd1_" + k] for r in sel]),
                   np.mean([r["bd2_" + k] for r in sel]),
                   np.mean([r["blen_" + k] for r in sel])))
        print()
    print("[eval] wrote %s" % args.out)


if __name__ == "__main__":
    main()

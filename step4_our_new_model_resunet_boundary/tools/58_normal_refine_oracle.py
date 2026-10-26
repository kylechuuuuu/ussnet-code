#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DESIGN_NEXT.md section 5.2 -- the ORACLE ceiling for normal-direction
contour refinement (redesign item A).

Item A is "a second stage that regresses a signed normal displacement on
+/-16 px strips around the contour".  Before building any such model, measure
the ceiling: give the refinement a PERFECT teacher -- move every predicted
contour point along its own normal by exactly the amount that lands it on the
GT boundary -- and see how much boundary Dice that buys.

If the oracle cannot gain >= 3 points of boundary Dice@1px without losing Dice,
item A is dead and no amount of architecture work will save it.  If it can, the
gain is the budget available for the learned version.

Method (no training, no network)
--------------------------------
1.  Signed distance field of the GT boundary: d = EDT(not gt_boundary), signed
    negative inside the GT.  Sampling d at a contour point gives how far that
    point must travel along the normal to reach the GT boundary.
2.  Predicted contour points are taken from the LARGEST connected component
    (so the oracle cannot be credited with topology cleanup), normals estimated
    from a lightly smoothed tangent, outward-oriented by testing the mask.
3.  Move p_i -> p_i - sdf(p_i) * n_i, clamp to +/-16 px, re-rasterise, score.

Reported alongside raw/cc1 so the effect is isolated, and the displacement
statistics show whether the residual really is the ~1-3 px the error analysis
claimed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS = os.path.dirname(STEP4)

spec = importlib.util.spec_from_file_location(
    "shape57", os.path.join(HERE, "57_shape_prior_ceiling.py"))
s57 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s57)

MAXD = 16.0


def sdf_of(gt):
    """Signed distance to the GT boundary: <0 inside, >0 outside."""
    gb = s57.surface(gt)
    d = ndimage.distance_transform_edt(~gb)
    return np.where(gt, -d, d)


def normals_of(pts, mask, smooth=5):
    """Outward unit normals at each point of a closed contour."""
    n = len(pts)
    k = np.arange(-(smooth // 2), smooth // 2 + 1)
    idx = (np.arange(n)[:, None] + k[None, :]) % n
    sm = pts[idx].mean(axis=1)                       # gentle tangential smoothing
    t = np.roll(sm, -1, axis=0) - np.roll(sm, 1, axis=0)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.maximum(tn, 1e-9)
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)      # 90-deg rotation of the tangent

    # orient outward: a small step along +n must leave the mask
    rr = np.clip(np.round(pts[:, 0] + 2.0 * nrm[:, 0]).astype(int), 0, mask.shape[0] - 1)
    cc = np.clip(np.round(pts[:, 1] + 2.0 * nrm[:, 1]).astype(int), 0, mask.shape[1] - 1)
    step_outside = ~mask[rr, cc]
    nrm[~step_outside] *= -1.0
    return nrm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(STEP4, "results",
                                                  "normal_refine_oracle.json"))
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train, val = s57.read_split(USS, exp="ultrasound_prostate_73")
    if args.max:
        val = val[:args.max]

    models = {
        "unet": os.path.join(USS, "unet", "results", "ultrasound_prostate_73"),
        "nnunet_best": os.path.join(USS, "step1_nnunet_baseline", "predictions",
                                    "nnunet_2d_fold0_checkpoint_best_png"),
    }

    rows = []
    for nm in val:
        gt = s57.load_gt(nm)
        if gt is None:
            continue
        base = nm if nm.lower().endswith(".png") else nm + ".png"
        sd = sdf_of(gt)
        for tag, d in models.items():
            pred = s57.load_pred(os.path.join(d, base))
            if pred is None:
                continue
            cc1 = s57.largest_cc(pred)
            rec = {"name": nm, "model": tag,
                   "dice_raw": s57.dice(pred, gt), "bd1_raw": s57.bdice(pred, gt, 1),
                   "dice_cc1": s57.dice(cc1, gt), "bd1_cc1": s57.bdice(cc1, gt, 1),
                   "bd2_cc1": s57.bdice(cc1, gt, 2),
                   "blen_cc1": int(s57.surface(cc1).sum())}

            c = s57.contour_of(cc1)
            if c is None:
                rows.append(rec)
                continue
            pts = c[:-1] if np.allclose(c[0], c[-1]) else c
            nrm = normals_of(pts, cc1)

            r = np.clip(np.round(pts[:, 0]).astype(int), 0, gt.shape[0] - 1)
            cc = np.clip(np.round(pts[:, 1]).astype(int), 0, gt.shape[1] - 1)
            disp = -sd[r, cc]                        # + means move outward
            disp = np.clip(disp, -MAXD, MAXD)

            moved = pts + disp[:, None] * nrm
            ref = s57.rasterize(moved, gt.shape)

            ad = np.abs(disp)
            rec.update({
                "dice_ref": s57.dice(ref, gt), "bd1_ref": s57.bdice(ref, gt, 1),
                "bd2_ref": s57.bdice(ref, gt, 2),
                "blen_ref": int(s57.surface(ref).sum()),
                "n_pts": int(len(pts)),
                "disp_absmean": float(ad.mean()),
                "disp_p50": float(np.percentile(ad, 50)),
                "disp_p90": float(np.percentile(ad, 90)),
                "disp_p99": float(np.percentile(ad, 99)),
                "frac_gt1px": float((ad > 1).mean()),
                "frac_clamped": float((ad >= MAXD - 1e-6).mean()),
                "frac_gt3px": float((ad > 3).mean()),
            })
            rows.append(rec)

    keys = ["dice", "bd1", "bd2", "blen"]
    variants = ["raw", "cc1", "ref"]
    stats = ["disp_absmean", "disp_p50", "disp_p90", "disp_p99",
             "frac_gt1px", "frac_gt3px", "frac_clamped"]

    def agg(sel, key):
        v = [r[key] for r in sel if key in r]
        return float(np.mean(v)) if v else float("nan")

    summary = {"n_val": len(val), "max_disp": MAXD, "models": {}}
    for tag in models:
        sel = [r for r in rows if r["model"] == tag]
        summary["models"][tag] = {"%s_%s" % (k, v): agg(sel, "%s_%s" % (k, v))
                                 for v in variants for k in keys}
        summary["models"][tag].update({s: agg(sel, s) for s in stats})
        summary["models"][tag]["n"] = len(sel)

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_image": rows}, f, indent=2)

    for tag in models:
        a = summary["models"][tag]
        print()
        print("=== %s  (n=%d) ===" % (tag, a["n"]))
        print("  %-6s %8s %8s %8s %10s" % ("var", "Dice", "bd1px", "bd2px", "bnd_len"))
        for v in variants:
            print("  %-6s %8.2f %8.2f %8.2f %10.1f"
                  % (v, a["dice_%s" % v], a["bd1_%s" % v], a["bd2_%s" % v], a["blen_%s" % v]))
        print("  ORACLE one-step normal refinement vs cc1:")
        print("      dDice %+0.2f | d_bd1px %+0.2f | d_bd2px %+0.2f | d_blen %+0.1f"
              % (a["dice_ref"] - a["dice_cc1"], a["bd1_ref"] - a["bd1_cc1"],
                 a["bd2_ref"] - a["bd2_cc1"], a["blen_ref"] - a["blen_cc1"]))
        print("  required displacement (px): mean %.2f | p50 %.2f | p90 %.2f | p99 %.2f"
              % (a["disp_absmean"], a["disp_p50"], a["disp_p90"], a["disp_p99"]))
        print("  fraction of contour points: >1px %.1f%% | >3px %.1f%% | clamped@%.0fpx %.1f%%"
              % (100 * a["frac_gt1px"], 100 * a["frac_gt3px"], MAXD, 100 * a["frac_clamped"]))

    print()
    print("[58] wrote %s" % args.out)
    print()
    for tag in models:
        a = summary["models"][tag]
        g = a["bd1_ref"] - a["bd1_cc1"]
        print("VERDICT %-12s oracle bd1px gain %+0.2f  (go for item A if >= +3.0 and Dice not down)"
              % (tag, g))


if __name__ == "__main__":
    main()

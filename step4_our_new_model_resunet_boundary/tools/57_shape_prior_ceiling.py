#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DESIGN_NEXT.md section 5.1 -- the shape-prior projection CEILING test.

Question: how much boundary quality is available from a pure 2D shape prior
(no retraining, no new network)?  If the ceiling is small, redesign item B is
not worth building; if it is large, B(ii) is justified.

Method
------
1.  Build a PCA shape model from the **71 fold-0 TRAINING ground-truth masks**
    only (no validation leakage).  Each contour is normalised for
    centroid / scale / orientation / chirality, resampled by arc length to
    N points, and stacked into a 2N-dim vector.
2.  For each of the 31 validation images and each model, take the prediction
    contour, normalise it the same way, project it onto the K first principal
    components, reconstruct it, and rasterise it back into the image frame.
3.  Report Dice / boundary-Dice@1px / boundary length, before and after.

Three variants are reported so the shape prior is NOT credited with the
topology cleanup it gets for free:
    raw      the prediction as produced
    cc1      only the largest connected component (topology cleanup only)
    proj     cc1 then projected onto the shape subspace (the thing under test)

Control: a GT round-trip (normalise -> reconstruct -> rasterise) is reported
first; if that is not near-lossless the whole measurement is meaningless.

Boundary Dice uses the exact step3_failure_analysis definition so the numbers
sit directly next to the 23.25 / 35.16 in FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS = os.path.dirname(STEP4)
STEP2 = os.path.join(USS, "step2_compare_unet_vs_nnunet")
sys.path.insert(0, os.path.join(STEP2, "scripts"))
from eval_lib import read_split  # noqa: E402

MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
GT_DIR_PNG = MASK_DIR
EPS = 1e-7
NPT = 400


# ---------------------------------------------------------------- geometry --
def surface(m):
    """1-px inner boundary, identical to step3's definition."""
    if not m.any():
        return np.zeros_like(m)
    return m & ~ndimage.binary_erosion(m, ndimage.generate_binary_structure(2, 1),
                                       border_value=0)


def dilate(m, r):
    return ndimage.binary_dilation(m, morphology.disk(r))


def dice(a, b):
    return 2.0 * float(np.logical_and(a, b).sum()) / (float(a.sum()) + float(b.sum()) + EPS) * 100


def bdice(pred, gt, tol=1):
    """Boundary Dice at tolerance, exactly as step3 computes it."""
    gb, pb = surface(gt), surface(pred)
    if gb.sum() == 0 or pb.sum() == 0:
        return 0.0
    prec = float(np.logical_and(pb, dilate(gb, tol)).sum()) / (pb.sum() + EPS)
    rec = float(np.logical_and(gb, dilate(pb, tol)).sum()) / (gb.sum() + EPS)
    return 2 * prec * rec / (prec + rec + EPS) * 100


def largest_cc(m):
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def contour_of(m):
    """Longest closed contour of the mask, in (row, col) image coordinates."""
    if not m.any():
        return None
    cs = measure.find_contours(np.pad(m.astype(np.uint8), 2).astype(float), 0.5)
    if not cs:
        return None
    c = np.asarray(max(cs, key=len), dtype=float) - 2.0
    if not np.allclose(c[0], c[-1]):
        c = np.vstack([c, c[:1]])
    return c


def resample_closed(pts, n, start_idx):
    """Arc-length resample a closed curve to n points, starting at start_idx."""
    p = pts[:-1] if np.allclose(pts[0], pts[-1]) else pts
    p = np.roll(p, -start_idx, axis=0)
    p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def normalize(mask):
    """mask -> (flat vector of n points, params) with centroid/scale/orientation
    /chirality/arc-origin canonicalised."""
    c = contour_of(mask)
    if c is None:
        return None, None
    p = c[:-1] if np.allclose(c[0], c[-1]) else c
    if len(p) < 8:
        return None, None
    cen = p.mean(axis=0)
    q = p - cen

    # principal axis
    cov = np.cov(q.T)
    w, v = np.linalg.eigh(cov)
    ax = v[:, int(np.argmax(w))]           # (dr, dc)
    ang = np.arctan2(ax[0], ax[1])
    ca, sa = np.cos(-ang), np.sin(-ang)
    R = np.array([[ca, -sa], [sa, ca]])
    q = q @ R.T

    # scale = RMS radius
    s = float(np.sqrt((q ** 2).sum(axis=1).mean()))
    if s <= 0:
        return None, None
    q = q / s

    # orientation 180-deg ambiguity: put the farthest point at positive col
    far = int(np.argmax((q ** 2).sum(axis=1)))
    neg180 = False
    if q[far, 1] < 0:
        q = -q                      # this is R180 (det=+1); must be undone
        neg180 = True
        far = int(np.argmax((q ** 2).sum(axis=1)))

    # chirality: enforce positive signed area (allows reflections in the model)
    flip = False
    x, y = q[:, 1], q[:, 0]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area2 < 0:
        q = np.stack([q[:, 0], -q[:, 1]], axis=1)
        flip = True
        far = int(np.argmax((q ** 2).sum(axis=1)))

    # rebuild from the normalised, still ordered curve
    closed = np.vstack([q, q[:1]])
    closed = closed[:-1]
    r = resample_closed(np.vstack([closed, closed[:1]]), NPT, 0)
    # rotate the parameterisation so index 0 is the farthest point
    j = int(np.argmin(np.linalg.norm(r - q[far], axis=1)))
    r = np.roll(r, -j, axis=0)

    params = dict(cen=cen, ang=ang, scale=s, flip=flip, far=int(far),
                  neg180=neg180)
    return r.reshape(-1), params


def denormalize(vec, params, shape):
    """(vector, params) -> binary mask in the original image frame."""
    q = np.asarray(vec, dtype=float).reshape(-1, 2).copy()
    if params["flip"]:
        q = np.stack([q[:, 0], -q[:, 1]], axis=1)
    if params.get("neg180"):
        q = -q
    q = q * params["scale"]
    ca, sa = np.cos(params["ang"]), np.sin(params["ang"])
    R = np.array([[ca, -sa], [sa, ca]])
    q = q @ R.T
    q = q + params["cen"]
    return rasterize(q, shape)


def rasterize(poly_rc, shape):
    """Exact even-odd fill of a (possibly concave) polygon, bounding-box only.

    draw.polygon is a scanline fill that mis-handles concave contours whose two
    sides come close (a narrow neck): the polygon self-intersects and the fill
    lands in the wrong place while keeping the right area.  Path.contains_points
    is exact for concave simple polygons, so the GT round-trip control becomes
    a real control.
    """
    r, c = poly_rc[:, 0], poly_rc[:, 1]
    r0 = max(int(np.floor(r.min())) - 1, 0)
    c0 = max(int(np.floor(c.min())) - 1, 0)
    r1 = min(int(np.ceil(r.max())) + 2, shape[0])
    c1 = min(int(np.ceil(c.max())) + 2, shape[1])
    rr, cc = np.mgrid[r0:r1, c0:c1]
    pts = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(float)
    m = np.zeros(shape, dtype=bool)
    inside = Path(poly_rc).contains_points(pts)
    m[rr.ravel()[inside], cc.ravel()[inside]] = True
    return m


def project(vec, mean, comps, k):
    d = vec - mean
    if k <= 0:
        return mean.copy()
    w = comps[:k] @ d
    return mean + w @ comps[:k]


# ------------------------------------------------------------------- prior --
def build_prior(names, label=""):
    vecs, used = [], []
    for nm in names:
        gt = load_gt(nm)
        if gt is None:
            continue
        v, _ = normalize(gt)
        if v is None:
            continue
        vecs.append(v)
        used.append(nm)
    V = np.asarray(vecs)
    # iterated generalised Procrustes-lite: align everything to the running mean
    mean = V.mean(axis=0)
    for _ in range(3):
        dist = np.linalg.norm(V - mean, axis=1, keepdims=True)
        mean = (V / np.maximum(dist, EPS)).mean(axis=0)
    d = V - mean
    U, S, Vt = np.linalg.svd(d, full_matrices=False)
    ev = (S ** 2) / max(len(V) - 1, 1)
    evr = ev / max(ev.sum(), EPS)
    print("[shape] prior built on %d %s masks; NPT=%d; "
          "explained variance K=8 %.1f%% K=16 %.1f%% K=32 %.1f%%"
          % (len(V), label or "training", NPT,
             100 * evr[:8].sum(), 100 * evr[:16].sum(), 100 * evr[:32].sum()))
    return mean, Vt, evr


def load_gt(name):
    base = name if name.lower().endswith(".png") else name + ".png"
    for d in (GT_DIR_PNG,):
        p = os.path.join(d, base)
        if os.path.exists(p):
            from PIL import Image
            a = np.array(Image.open(p).convert("L"))
            return a > 0
    return None


def load_pred(path):
    if not os.path.exists(path):
        return None
    from PIL import Image
    a = np.array(Image.open(path).convert("L"))
    return a > 0


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--out", default=os.path.join(STEP4, "results",
                                                  "shape_prior_ceiling.json"))
    ap.add_argument("--max", type=int, default=0, help="debug: limit val images")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train_names, val_names = read_split(USS, exp="ultrasound_prostate_73")
    print("[shape] split: %d train / %d val" % (len(train_names), len(val_names)))
    if args.max:
        val_names = val_names[:args.max]

    mean, comps, evr = build_prior(train_names, "fold-0 training")

    models = {
        "unet": os.path.join(USS, "unet", "results", "ultrasound_prostate_73"),
        "nnunet_best": os.path.join(USS, "step1_nnunet_baseline", "predictions",
                                    "nnunet_2d_fold0_checkpoint_best_png"),
    }
    for tag, d in models.items():
        print("[shape] %-12s %s  (%d png)" % (tag, d, len(os.listdir(d)) if os.path.isdir(d) else -1))

    rows = []
    ctrl = []
    for nm in val_names:
        gt = load_gt(nm)
        if gt is None:
            print("[shape] WARN no GT for %s" % nm)
            continue
        base = nm if nm.lower().endswith(".png") else nm + ".png"

        # ---- control: GT round trip -----------------------------------
        v, prm = normalize(gt)
        if v is not None:
            rt = denormalize(v, prm, gt.shape)
            ctrl.append((dice(rt, gt), bdice(rt, gt, 1), bdice(rt, gt, 2)))

        for tag, d in models.items():
            pred = load_pred(os.path.join(d, base))
            if pred is None:
                continue
            rec = {"name": nm, "model": tag,
                   "dice_raw": dice(pred, gt), "bd1_raw": bdice(pred, gt, 1),
                   "bd2_raw": bdice(pred, gt, 2),
                   "blen_raw": int(surface(pred).sum())}
            cc1 = largest_cc(pred)
            rec.update({"dice_cc1": dice(cc1, gt), "bd1_cc1": bdice(cc1, gt, 1),
                        "bd2_cc1": bdice(cc1, gt, 2), "blen_cc1": int(surface(cc1).sum())})
            vp, prmp = normalize(cc1)
            if vp is not None:
                for k in args.ks:
                    proj_vec = project(vp, mean, comps, k)
                    pm = denormalize(proj_vec, prmp, gt.shape)
                    rec["dice_k%d" % k] = dice(pm, gt)
                    rec["bd1_k%d" % k] = bdice(pm, gt, 1)
                    rec["bd2_k%d" % k] = bdice(pm, gt, 2)
                    rec["blen_k%d" % k] = int(surface(pm).sum())
            rows.append(rec)

    keys = ["dice", "bd1", "bd2", "blen"]
    variants = ["raw", "cc1"] + ["k%d" % k for k in args.ks]

    def agg(sel, variant):
        out = {}
        for k in keys:
            vals = [r["%s_%s" % (k, variant)] for r in sel if "%s_%s" % (k, variant) in r]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        out["n"] = len(sel)
        return out

    summary = {"n_train": len(train_names), "n_val": len(val_names),
               "explained_variance": [float(x) for x in evr[:32]],
               "control_gt_roundtrip": {},
               "models": {}}
    if ctrl:
        arr = np.asarray(ctrl)
        summary["control_gt_roundtrip"] = {"dice": float(arr[:, 0].mean()),
                                           "bd1": float(arr[:, 1].mean()),
                                           "bd2": float(arr[:, 2].mean()),
                                           "n": len(ctrl)}

    for tag in models:
        sel = [r for r in rows if r["model"] == tag]
        summary["models"][tag] = {v: agg(sel, v) for v in variants}

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_image": rows}, f, indent=2)

    # ---- report ----------------------------------------------------------
    print()
    if ctrl:
        c = summary["control_gt_roundtrip"]
        print("CONTROL  GT round-trip  : Dice %.2f | bDice@1px %.2f | bDice@2px %.2f  (n=%d)"
              % (c["dice"], c["bd1"], c["bd2"], c["n"]))
    for tag in models:
        print()
        print("=== %s ===" % tag)
        print("  %-6s %8s %8s %8s %10s" % ("var", "Dice", "bd1px", "bd2px", "bnd_len"))
        for v in variants:
            a = summary["models"][tag][v]
            print("  %-6s %8.2f %8.2f %8.2f %10.1f" % (v, a["dice"], a["bd1"], a["bd2"], a["blen"]))
        b0 = summary["models"][tag]["cc1"]
        for k in args.ks:
            a = summary["models"][tag]["k%d" % k]
            print("  -> K=%-2d vs cc1: dDice %+0.2f | d_bd1px %+0.2f | d_bd2px %+0.2f"
                  % (k, a["dice"] - b0["dice"], a["bd1"] - b0["bd1"], a["bd2"] - b0["bd2"]))

    print()
    print("[shape] GT boundary length (mean over val) = %.1f px"
          % np.mean([r["blen_raw"] for r in rows if r["model"] == "nnunet_best"]))
    print("[shape] wrote %s" % args.out)
    print()
    verdict = []
    for tag in models:
        b0 = summary["models"][tag]["cc1"]
        best = max(summary["models"][tag]["k%d" % k]["bd1"] for k in args.ks)
        verdict.append("%s: best bd1px gain %+0.2f" % (tag, best - b0["bd1"]))
    print("VERDICT (go if >= +3.0 bd1px): " + " | ".join(verdict))


if __name__ == "__main__":
    main()

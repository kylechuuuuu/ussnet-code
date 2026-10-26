#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""99e: GT-free boundary-risk score.

step9 established that the val boundary error is invisible in the image and not
fixable by loss / refiner / post-processing.  The practical deliverable is then
triage: can we tell, WITHOUT ground truth, which cases have a bad boundary, so
only those need review?

Scores (all GT-free, from the model's own outputs):
  width_px      mean learned band width  |Mout & ~Min| / perimeter        (step6)
  bnd_entropy   mean binary entropy of base prob in a 3px band on the pred contour
  bnd_std       std of base prob in that band
  p_out         mean base prob in the shell just OUTSIDE the pred contour (2-6px)
  p_in          mean base prob in the shell just INSIDE                       (2-6px)
  laminarity    |base & ~cautious| / per  (how much the cautious head gives up)

Targets: bdice@1px, HD95, and the band loss L_band (needs GT, used only to score).
"""
import csv
import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
GT = os.path.join(USS, "UltrasoundProstate", "masks")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = load_mod("arb", os.path.join(S6, "tools", "63_arbitrate.py"))
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools",
                                    "57_shape_prior_ceiling.py"))

NPZ = os.path.join(S6, "predictions", "sq2resencw25_checkpoint_best_%s_nifti")
train, val = arb.read_split(USS)


def surface(m):
    st = ndi.generate_binary_structure(2, 1)
    return m & ~ndi.binary_erosion(m, st, border_value=0)


def sdf_out(m):
    return ndi.distance_transform_edt(~m)


def l_band(q, gt, D=12.0, cap=3.0):
    """case-level replica of trainer/boundary_losses.band_terms (sum over pixels)."""
    C, R = cap * D, int(np.ceil(cap * D)) + 1
    d_out = ndi.distance_transform_cdt(~gt, metric="chessboard")
    d_in = ndi.distance_transform_cdt(gt, metric="chessboard")
    d_out = np.clip(d_out, 0, R).astype(np.float32)
    d_in = np.clip(d_in, 0, R).astype(np.float32)
    P = max(surface(gt).sum(), 1)
    lo = float((np.clip(d_out, 0, C) * q).sum()) * 2.0 / (D * P)
    li = float((np.clip(d_in, 0, C) * (1.0 - q)).sum()) * 2.0 / (D * P)
    return lo, li


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


rows = []
for name in val:
    stem = arb.stem_of(name)
    gt = s57.load_gt(name)
    pb = arb.align_prob(arb.load_prob(NPZ % "base", stem), gt.shape)
    pc = arb.align_prob(arb.load_prob(NPZ % "cautious", stem), gt.shape)
    pa = arb.align_prob(arb.load_prob(NPZ % "amplified", stem), gt.shape)
    M0 = s57.largest_cc(pb > 0.5)
    Min = M0 & (pc > 0.5)
    Mout = M0 | (pa > 0.5)
    per = max(surface(M0).sum(), 1)
    d = sdf_out(M0)
    band = (d >= 1) & (d <= 3)
    band_in = M0 & (d >= 1) & (d <= 3)
    p = np.clip(pb, 1e-6, 1 - 1e-6)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    lo, li = l_band(pb.astype(np.float32), gt)
    bd = s57.bdice(M0, gt, tol=1)
    rows.append(dict(
        name=name,
        width_px=float((Mout & ~Min).sum()) / per,
        bnd_entropy=float(ent[band].mean()),
        bnd_std=float(pb[band].std()),
        p_out=float(pb[band & ~M0].mean()),
        p_in=float(pb[band_in].mean()),
        laminarity=float((M0 & ~Min).sum()) / per,
        bdice1=float(bd),
        l_band=float(lo + li),
        dice=float(s57.dice(M0, gt)),
    ))

scores = ["width_px", "bnd_entropy", "bnd_std", "p_out", "p_in", "laminarity"]
print("%-22s %7s %7s %7s" % ("score", "rho(bd)", "rho(HD)", "rho(Lb)"))
bd = [r["bdice1"] for r in rows]
lb = [r["l_band"] for r in rows]
for s in scores:
    v = [r[s] for r in rows]
    print("%-22s %+7.3f %7s %+7.3f" % (s, spearman(v, bd), "-", spearman(v, lb)))

# AUROC of each score for "boundary is bad" = bdice1 below the median
def auroc(score, labels):
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


print("\nAUROC for bdice1 in the worst half (n=%d pos=%d)"
      % (len(rows), (np.asarray(bd) < np.median(bd)).sum()))
for s in scores:
    v = np.asarray([r[s] for r in rows])
    for lab, nm in [(np.asarray(bd) < 20.0, "bdice<20"),
                    (np.asarray(bd) < np.median(bd), "bdice<median"),
                    (np.asarray(lb) > np.median(lb), "L_band>median")]:
        print("  %-12s %-14s AUROC %.3f" % (s, nm, auroc(v, lab)))

out = os.path.join(S9, "results", "risk_score_val.json")
with open(out, "w") as f:
    json.dump(dict(rows=rows), f, indent=1)
print("\nwrote", out)

# also dump a csv for eyeballing
with open(os.path.join(S9, "results", "risk_score_val.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("worst by bdice1:")
for r in sorted(rows, key=lambda r: r["bdice1"])[:6]:
    print("  %-28s bdice1 %6.2f  L_band %6.2f  width %5.2f  p_out %.2f  p_in %.2f"
          % (r["name"], r["bdice1"], r["l_band"], r["width_px"], r["p_out"], r["p_in"]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""99f: does the model's own uncertainty LOCALIZE its boundary error?

99e showed case-level triage is possible (rho(bnd_std, bdice1) = -0.68).
The useful question for review / active labelling is spatial: given a contour
point, can a GT-free score tell whether THAT point is off?

Per contour point (400 resampled points on the base contour):
  gt-free : p0        base prob at the point
            ent_loc   mean binary entropy of base prob within |t|<=2 px
            std_loc   std  of base prob within |t|<=4 px
            give      cautious head is 0 inside and base is 1 (head concession)
            over      amplified head is 1 outside (head extension)
  target  : |t_gt|    signed truth displacement along the normal (SDF zero crossing)
"""
import importlib.util
import json
import os

import numpy as np
from scipy import ndimage as ndi

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
NPZ = os.path.join(S6, "predictions", "sq2resencw25_checkpoint_best_%s_nifti")
TS = np.arange(-40.0, 40.0001, 0.5)
NPT = 400


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = load_mod("arb", os.path.join(S6, "tools", "63_arbitrate.py"))
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools",
                                   "57_shape_prior_ceiling.py"))
s58 = load_mod("s58", os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools",
                                   "58_normal_refine_oracle.py"))
train, val = arb.read_split(USS)


def sdf_of(gt):
    st = ndi.generate_binary_structure(2, 1)
    s = gt & ~ndi.binary_erosion(gt, st, border_value=0)
    d = ndi.distance_transform_edt(~s)
    return np.where(gt, -d, d).astype(np.float32)


def t_gt_of(sdf, pts, nrm, ts):
    coords = np.stack([pts[:, 0][None, :] + nrm[:, 0][None, :] * ts[:, None],
                       pts[:, 1][None, :] + nrm[:, 1][None, :] * ts[:, None]])
    v = ndi.map_coordinates(sdf, coords, order=1, mode="nearest")
    nt, npt = v.shape
    sign = np.sign(v)
    out = np.full(npt, np.nan)
    for j in range(npt):
        col = sign[:, j]
        zi = np.nonzero(np.diff(col) != 0)[0]
        if len(zi) == 0:
            continue
        t0 = ts[zi]; t1 = ts[zi + 1]
        v0 = v[zi, j]; v1 = v[zi + 1, j]
        tt = t0 - v0 * (t1 - t0) / (v1 - v0)
        k = np.argmin(np.abs(tt))
        out[j] = tt[k]
    return out


def sample(arr, pts, nrm, ts):
    coords = np.stack([pts[:, 0][None, :] + nrm[:, 0][None, :] * ts[:, None],
                       pts[:, 1][None, :] + nrm[:, 1][None, :] * ts[:, None]])
    return ndi.map_coordinates(arr, coords, order=1, mode="nearest")


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    a, b = a[m], b[m]
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


per_case, pooled = [], {k: [[], []] for k in ("ent_loc", "std_loc", "give", "over", "p0")}
for name in val:
    stem = arb.stem_of(name)
    gt = s57.load_gt(name)
    sdf = sdf_of(gt)
    pb = arb.align_prob(arb.load_prob(NPZ % "base", stem), gt.shape)
    pc = arb.align_prob(arb.load_prob(NPZ % "cautious", stem), gt.shape)
    pa = arb.align_prob(arb.load_prob(NPZ % "amplified", stem), gt.shape)
    M0 = s57.largest_cc(pb > 0.5)
    c = s57.contour_of(M0)
    pts = s57.resample_closed(c, NPT, 0)
    nrm = s58.normals_of(pts, M0)
    tgt = np.abs(t_gt_of(sdf, pts, nrm, TS))
    p = np.clip(pb, 1e-6, 1 - 1e-6)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)).astype(np.float32)
    near = np.abs(TS) <= 2.0
    ent_loc = sample(ent, pts, nrm, TS[near]).mean(axis=0)
    std_loc = sample(pb, pts, nrm, TS[np.abs(TS) <= 4.0]).std(axis=0)
    pc_v = sample(pc, pts, nrm, TS[np.abs(TS) <= 1.0]).mean(axis=0)
    pa_v = sample(pa, pts, nrm, TS[np.abs(TS) <= 1.0]).mean(axis=0)
    pb_v = sample(pb, pts, nrm, np.array([0.0]))[0]
    give = (pb_v > 0.5) & (pc_v < 0.5)
    over = (pb_v < 0.5) & (pa_v > 0.5)
    r = dict(name=name,
             rho_ent=spearman(ent_loc, tgt), rho_std=spearman(std_loc, tgt),
             rho_give=spearman(give.astype(float), tgt),
             rho_over=spearman(over.astype(float), tgt),
             mean_abs_t=float(np.nanmean(tgt)))
    per_case.append(r)
    for k, v in (("ent_loc", ent_loc), ("std_loc", std_loc), ("give", give.astype(float)),
                 ("over", over.astype(float)), ("p0", pb_v)):
        pooled[k][0].append(v); pooled[k][1].append(tgt)

print("%-28s %7s %7s %7s %7s" % ("case", "ent", "std", "give", "over"))
for r in per_case:
    print("%-28s %+7.3f %+7.3f %+7.3f %+7.3f"
          % (r["name"][:28], r["rho_ent"], r["rho_std"], r["rho_give"], r["rho_over"]))
print("\nmean per-case rho:")
for k in ("ent_loc", "std_loc", "give", "over"):
    v = [r["rho_" + k.split("_")[0]] for r in per_case]
    print("  %-8s %+.3f   (positive in %d/%d cases)"
          % (k, np.nanmean(v), int(np.sum(np.array(v) > 0)), len(v)))
print("\npooled (all points, all cases):")
for k in ("ent_loc", "std_loc", "give", "over", "p0"):
    A = np.concatenate(pooled[k][0]); B = np.concatenate(pooled[k][1])
    print("  %-8s rho %+.3f" % (k, spearman(A, B)))

out = os.path.join(S9, "results", "localization_val.json")
with open(out, "w") as f:
    json.dump(dict(per_case=per_case, pooled={
        k: spearman(np.concatenate(v[0]), np.concatenate(v[1])) for k, v in pooled.items()}),
        f, indent=1)
print("\nwrote", out)

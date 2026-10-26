#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""99h: gate the old mechanism (edge snap) with the new one (GT-free risk map).

step9 established two facts that have never been combined:
  * the failure is an OUTWARD spike on the anterior arc (91: 1037 overshoot vs
    718 undershoot), where the truth edge is invisible (94: 68% of tail points
    have no gradient peak at the GT position);
  * the model's own probability field knows WHERE that arc is, GT-free
    (99f: entropy within +-2px, per-case rho +0.45, 31/31 positive).

So: snap to the nearest strong ridge only on the high-entropy points, and try
the shrink-only variant -- the error is an overshoot, so the useful motion is
inward.  Everything here is GT-free at decision time; GT is used only to score.

Variants: gate in {none, all, top50, top25} x mode in {both, shrink, grow}
          x frac in {0.5, 0.7}
"""
import importlib.util
import json
import os

import numpy as np
from PIL import Image
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


def probes(arr, pts, nrm, ts):
    return ndi.map_coordinates(arr, np.stack([
        pts[:, 0][None, :] + nrm[:, 0][None, :] * ts[:, None],
        pts[:, 1][None, :] + nrm[:, 1][None, :] * ts[:, None]]), order=1, mode="nearest")


def prep(name):
    stem = arb.stem_of(name)
    gt = s57.load_gt(name)
    img = arb.load_image(stem)
    pb = arb.align_prob(arb.load_prob(NPZ % "base", stem), gt.shape)
    M0 = s57.largest_cc(pb > 0.5)
    pts = s57.resample_closed(s57.contour_of(M0), NPT, 0)
    nrm = s58.normals_of(pts, M0)
    imgf = ndi.gaussian_filter(img, 1.5)
    G = arb.ridge_profile(imgf, pts, nrm, TS, 1.0)
    p = np.clip(pb, 1e-6, 1 - 1e-6)
    ent = (-(p * np.log(p) + (1 - p) * np.log(1 - p))).astype(np.float32)
    ent_loc = probes(ent, pts, nrm, TS[np.abs(TS) <= 2.0]).mean(axis=0)
    return dict(name=name, stem=stem, gt=gt, pb=pb, M0=M0, pts=pts, nrm=nrm, G=G, ent=ent_loc)


def score(M, gt):
    return dict(bdice=float(s57.bdice(M, gt, tol=1)), dice=float(s57.dice(M, gt)))


VARIANTS = []
for gate in ("none", "all", "top50", "top25"):
    for mode in ("both", "shrink", "grow"):
        for frac in (0.5, 0.7):
            VARIANTS.append((gate, mode, frac))

rows = []
cache = {}
for name in val:
    d = prep(name)
    cache[name] = d
    base = score(d["M0"], d["gt"])
    rows.append(dict(name=name, variant="raw", **base))
    for gate, mode, frac in VARIANTS:
        lo, hi = {"both": (-40.0, 40.0), "shrink": (-40.0, 0.0), "grow": (0.0, 40.0)}[mode]
        loA = np.full(NPT, lo, dtype=np.float64)
        hiA = np.full(NPT, hi, dtype=np.float64)
        t = arb.pick(d["G"], TS, frac, loA, hiA)
        if gate != "none":
            q = {"all": 0.0, "top50": 50.0, "top25": 75.0}[gate]
            thr = np.percentile(d["ent"], q)
            t = np.where(d["ent"] >= thr, t, 0.0)
        M, _tapp = arb.apply_t(d["M0"], d["pts"], d["nrm"], t)
        rows.append(dict(name=name, variant="%s_%s_f%.1f" % (gate, mode, frac),
                         moved=float(np.mean(np.abs(t) > 0.01)), **score(M, d["gt"])))

agg = {}
for r in rows:
    agg.setdefault(r["variant"], []).append(r)
print("%-18s %8s %8s %7s %7s" % ("variant", "bdice1", "dice", "win", "loss"))
ref = np.array([r["bdice"] for r in agg["raw"]])
out = []
for v in ["raw"] + ["%s_%s_f%.1f" % t for t in VARIANTS]:
    a = agg[v]
    bd = np.array([r["bdice"] for r in a])
    dc = np.array([r["dice"] for r in a])
    out.append(dict(variant=v, bdice1=float(bd.mean()), dice=float(dc.mean()),
                    win=int((bd > ref).sum()), loss=int((bd < ref).sum()),
                    delta=float((bd - ref).mean()),
                    moved=float(np.mean([r.get("moved", 0.0) for r in a]))))
for r in sorted(out, key=lambda r: -r["bdice1"]):
    print("%-18s %8.3f %8.4f %7d %7d  %+6.3f  moved %.3f"
          % (r["variant"], r["bdice1"], r["dice"], r["win"], r["loss"], r["delta"], r["moved"]))

p = os.path.join(S9, "results", "gated_snap_val.json")
with open(p, "w") as f:
    json.dump(dict(summary=out, per_case=rows), f, indent=1)
print("wrote", p)

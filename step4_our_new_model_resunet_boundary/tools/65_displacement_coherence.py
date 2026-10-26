#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""65 -- how smooth is the correction the refiner must produce?

Two things are measured on the 31 fold-0 val images, with no training and no GPU:

1.  RECEPTIVE-FIELD REQUIREMENT.  The oracle correction field d(s) (move point s
    along its normal by -sdf_GT) is low-pass filtered along the arc length with
    growing sigma and re-applied.  bd1px(sigma) tells us how much of the oracle
    gain survives when the correction is only allowed to vary slowly along the
    contour -- i.e. what tangential context a network needs in order to express
    the correction at all.  The current refiner (60/64) sees NU = 17 points
    tangentially out of NPT = 400.

2.  DISTRIBUTION MISMATCH.  The autocorrelation half-width of the REAL correction
    field is compared with that of the SYNTHETIC perturbation used to train the
    refiner (s60.perturb_field: gaussian_filter1d(randn, U(0.02,0.20)*400)).
    If the real field is much more coherent than the synthetic one, the refiner
    is being trained to invert the wrong process.

Writes results/displacement_coherence.json.
"""
import os
import csv
import json
import time
import importlib.util

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
USS = os.path.dirname(ROOT)
S1 = os.path.join(USS, "step1_nnunet_baseline")
S1_NPZ = os.path.join(S1, "predictions", "nnunet_2d_fold0_checkpoint_best_nifti")
S1_PNG = os.path.join(S1, "predictions", "nnunet_2d_fold0_checkpoint_best_png")
OUT = os.path.join(ROOT, "results", "displacement_coherence.json")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s57 = _load("s57", os.path.join(TOOLS, "57_shape_prior_ceiling.py"))
s58 = _load("s58", os.path.join(TOOLS, "58_normal_refine_oracle.py"))
s60 = _load("s60", os.path.join(TOOLS, "60_train_normal_refiner.py"))

NPT = s57.NPT
MAXD = s60.MAXD
SIGMAS = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 200.0]


def name2cid():
    out = {}
    with open(os.path.join(S1, "case_id_mapping.csv")) as f:
        for row in csv.DictReader(f):
            out[row["original_filename"]] = row["case_id"]
    return out


def score(m, gt):
    """s57.dice / s57.bdice already return percent."""
    return (s57.dice(m, gt), s57.bdice(m, gt, 1),
            s57.bdice(m, gt, 2), float(s57.surface(m).sum()))


def ac_halfwidth(x):
    """Smallest lag where the autocorrelation drops below 0.5 (periodic)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    F = np.fft.rfft(x, n=n)
    ac = np.fft.irfft(np.abs(F) ** 2, n=n)
    if ac[0] <= 0:
        return float(n)
    ac = ac / ac[0]
    for lag in range(1, n):
        if ac[lag] < 0.5:
            return float(lag)
    return float(n)


def main():
    t0 = time.time()
    n2c = name2cid()
    tr, te = s57.read_split(USS, exp="ultrasound_prostate_73")
    print("[65] val images: %d" % len(te))

    keys = ["ctrl", "oracle"] + ["lp%g" % s for s in SIGMAS]
    rows = []
    for nm in te:
        cid = n2c.get(nm)
        gt = s57.load_gt(nm)
        if cid is None or gt is None:
            continue
        npz = os.path.join(S1_NPZ, cid + ".npz")
        png = os.path.join(S1_PNG, nm)
        if not (os.path.exists(npz) and os.path.exists(png)):
            continue
        m0 = np.array(Image.open(png).convert("L")) > 127
        m = s57.largest_cc(m0)
        c = s57.contour_of(m)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, m)
        sdf = s58.sdf_of(gt)
        d_or = np.clip(-s60.bil(sdf.astype(np.float32), pts), -MAXD, MAXD)

        rec = {"name": nm}
        rec["ctrl"] = score(m, gt)
        rec["oracle"] = score(s57.rasterize(pts + d_or[:, None] * nrm, gt.shape), gt)
        for s in SIGMAS:
            ds = ndimage.gaussian_filter1d(d_or, s, mode="wrap") if s > 0 else d_or
            rr = s57.rasterize(pts + ds[:, None] * nrm, gt.shape)
            rec["lp%g" % s] = score(rr, gt)
        rec["ac_half"] = ac_halfwidth(d_or)
        rec["d_std"] = float(d_or.std())
        rec["d_absmean"] = float(np.abs(d_or).mean())
        rows.append(rec)

    print("\n== n=%d   (bd1px vs low-pass sigma of the correction field) ==" % len(rows))
    print("%-8s %8s %8s %8s %8s %10s" % ("variant", "Dice", "bd1px", "bd2px", "gain", "bnd_len"))
    summ = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        summ[k] = v.mean(axis=0).tolist()
        gain = v[:, 1].mean() - np.array([r["ctrl"] for r in rows])[:, 1].mean()
        print("%-8s %8.2f %8.2f %8.2f %+8.2f %10.1f" %
              (k, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), gain, v[:, 3].mean()))

    bd_ctrl = np.array([r["ctrl"][1] for r in rows]).mean()
    bd_or = np.array([r["oracle"][1] for r in rows]).mean()
    need = {}
    for s in SIGMAS:
        v = np.array([r["lp%g" % s][1] for r in rows]).mean()
        need["lp%g" % s] = (v, (v - bd_ctrl) / max(bd_or - bd_ctrl, 1e-9))
    print("\n== fraction of the oracle boundary gain retained by a sigma-low-passed correction ==")
    for s in SIGMAS:
        v, frac = need["lp%g" % s]
        print("  sigma %6g pt (%7.1f px)  bd1px %6.2f   %5.1f %% of the oracle gain"
              % (s, s * 1094.2 / NPT, v, 100.0 * frac))

    acr = np.array([r["ac_half"] for r in rows], dtype=float)
    print("\n== autocorrelation half-width (arc-length points, 1 pt = %.2f px) ==" %
          (1094.2 / NPT))
    print("  REAL correction   mean %6.1f  median %6.1f  p90 %6.1f  (%.1f px median)" %
          (acr.mean(), np.median(acr), np.percentile(acr, 90),
           np.median(acr) * 1094.2 / NPT))

    rng = np.random.default_rng(0)
    syn, syn_sig = [], []
    n = NPT
    for _ in range(4000):
        sig = rng.uniform(0.02, 0.20) * n
        raw = rng.standard_normal(n)
        d = ndimage.gaussian_filter1d(raw, max(sig, 1.0), mode="wrap")
        d = d / (d.std() + 1e-9)
        amp = float(np.abs(rng.normal(0.0, 1.0)) *
                    rng.choice([0.5, 0.9, 1.5, 2.5, 4.0, 7.0, 11.0]))
        d = np.clip(d * amp, -MAXD, MAXD)
        syn.append(ac_halfwidth(d))
        syn_sig.append(sig)
    syn = np.array(syn)
    print("  SYNTHETIC perturb mean %6.1f  median %6.1f  p90 %6.1f  (sigma draw mean %.1f pt)"
          % (syn.mean(), np.median(syn), np.percentile(syn, 90), float(np.mean(syn_sig))))
    print("\n  model tangential receptive field: NU = %d points (%.1f px) out of %d points"
          % (s60.NU, s60.NU * 1094.2 / NPT, NPT))

    out = {"n": len(rows), "sigmas": SIGMAS, "summary": summ,
           "oracle_gain_fraction": {k: [v[0], v[1]] for k, v in need.items()},
           "ac_half_real": {"mean": float(acr.mean()), "median": float(np.median(acr)),
                            "p90": float(np.percentile(acr, 90))},
           "ac_half_synth": {"mean": float(syn.mean()), "median": float(np.median(syn)),
                             "p90": float(np.percentile(syn, 90))},
           "per_image": rows}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[65] wrote %s   total %.0fs" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""68 -- is the real residual TRANSFERABLE at all?

Tools 64 / 66 / 67 all trained a ~30k-parameter per-point CNN to predict the
normal displacement and all three memorised their training images (train bd1px
+19 to +24) while hurting held-out images (-2.4 to -8.2).  Before concluding
that "the residual cannot be learned", we have to separate two very different
claims:

  (a) the residual is PER-PATIENT IDIOSYNCRATIC -- the required displacement
      differs from patient to patient, so nothing transfers and no amount of
      capacity or data will help;
  (b) the residual is SYSTEMATIC but our model is too high-capacity for 20-71
      images -- a tiny model would transfer.

This tool answers that with essentially no capacity at all: it fits a handful of
global parameters to the real stage-1 residual on the 20 fit images and applies
them unchanged to the 11 held-out ones.

  sector K   : mean displacement per 30-degree sector (K params)
  fourier K  : mean of the first K Fourier coefficients of d(s) (2K+1 params)
  const      : just the global mean displacement (1 param)

If even a 12-parameter model does not transfer, the family is dead for this
dataset.  If it does transfer, the earlier failures were a capacity/data problem
and a small-model variant is worth a GPU.

Writes results/residual_transfer.json.
"""
import os
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
OUT = os.path.join(ROOT, "results", "residual_transfer.json")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s57 = _load("s57", os.path.join(TOOLS, "57_shape_prior_ceiling.py"))
s58 = _load("s58", os.path.join(TOOLS, "58_normal_refine_oracle.py"))
s60 = _load("s60", os.path.join(TOOLS, "60_train_normal_refiner.py"))
s64 = _load("s64", os.path.join(TOOLS, "64_stage2_refiner_context.py"))

NPT = s64.NPT
MAXD = s60.MAXD
NSEC = 12


def score(m, gt):
    return (s57.dice(m, gt), s57.bdice(m, gt, 1),
            s57.bdice(m, gt, 2), float(s57.surface(m).sum()))


def collect(names, pred_dir, tag):
    """-> list of dicts with pts, nrm, d(real residual), sector, shape, mask, gt."""
    out = []
    for nm in names:
        gt = s57.load_gt(nm)
        if gt is None:
            continue
        png = os.path.join(pred_dir, nm)
        if not os.path.exists(png):
            continue
        m0 = np.array(Image.open(png).convert("L")) > 127
        m = s57.largest_cc(m0)
        c = s57.contour_of(m)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, m)
        sdf = s58.sdf_of(gt)
        d = np.clip(-s60.bil(sdf.astype(np.float32), pts), -MAXD, MAXD)
        cen = pts.mean(axis=0)
        ang = np.degrees(np.arctan2(-(pts[:, 0] - cen[0]), pts[:, 1] - cen[1]))
        sec = np.floor((ang + 180.0) / (360.0 / NSEC)).astype(int) % NSEC
        out.append({"name": nm, "pts": pts, "nrm": nrm, "d": d, "sec": sec,
                    "shape": gt.shape, "gt": gt, "m0": m0})
    print("[68] %-10s %d images" % (tag, len(out)))
    return out


def fourier_basis(n, K):
    s = np.arange(n, dtype=float) / n * 2.0 * np.pi
    cols = [np.ones(n)]
    for k in range(1, K + 1):
        cols.append(np.cos(k * s))
        cols.append(np.sin(k * s))
    return np.stack(cols, axis=1)


def apply_field(rec, d):
    return s57.rasterize(rec["pts"] + d[:, None] * rec["nrm"], rec["shape"])


def main():
    t0 = time.time()
    tr, te = s57.read_split(USS, exp="ultrasound_prostate_73")
    fit_names, hold_names = te[:20], te[20:]
    PRED = {
        "nnunet_best": os.path.join(S1, "predictions", "nnunet_2d_fold0_checkpoint_best_png"),
        "unet": os.path.join(USS, "unet", "results", "ultrasound_prostate_73"),
    }

    res = {}
    for tag, pdir in PRED.items():
        fit = collect(fit_names, pdir, tag + "/fit")
        hold = collect(hold_names, pdir, tag + "/hold")
        if not fit or not hold:
            continue
        print("\n########## %s ##########" % tag)

        # ---- reference rows on the held-out set
        rows = {"hold": hold}
        base = {}
        for nm, grp in (("fit", fit), ("hold", hold)):
            base[nm] = {
                "ctrl": np.array([score(r["m0"], r["gt"]) for r in grp]).mean(axis=0),
                "oracle": np.array([score(apply_field(r, r["d"]), r["gt"]) for r in grp]).mean(axis=0),
            }
        print("fit  n=%d  ctrl bd1px %.2f  oracle bd1px %.2f"
              % (len(fit), base["fit"]["ctrl"][1], base["fit"]["oracle"][1]))
        print("hold n=%d  ctrl bd1px %.2f  oracle bd1px %.2f"
              % (len(hold), base["hold"]["ctrl"][1], base["hold"]["oracle"][1]))

        variants = {}

        # ---- (1) sector means: NSEC parameters
        num = np.zeros(NSEC)
        cnt = np.zeros(NSEC)
        for r in fit:
            np.add.at(num, r["sec"], r["d"])
            np.add.at(cnt, r["sec"], 1.0)
        sec_mean = num / np.maximum(cnt, 1.0)
        variants["sector12"] = (lambda r, sm=sec_mean: sm[r["sec"]])

        # ---- (2) Fourier coefficient means, K = 1, 2, 4, 8
        for K in (1, 2, 4, 8):
            B = fourier_basis(NPT, K)
            coef = np.stack([np.linalg.lstsq(B, r["d"], rcond=None)[0] for r in fit])
            cm = coef.mean(axis=0)
            variants["fourier%d" % K] = (lambda r, B=B, cm=cm: B.dot(cm))

        # ---- (3) global constant
        cm0 = np.array([r["d"].mean() for r in fit]).mean()
        variants["const"] = (lambda r, c=cm0: np.full(NPT, c))

        # ---- (4) the mean residual field itself (NPT parameters, upper bound for
        #          "one field for everybody")
        meanfield = np.stack([r["d"] for r in fit]).mean(axis=0)
        variants["meanfield"] = (lambda r, f=meanfield: f)

        # ---- (5) sector means scaled by each held-out image's own residual scale
        #          (uses the test residual amplitude -- an oracle-ish upper bound)
        def scaled(r, sm=sec_mean):
            dm = sm[r["sec"]]
            if dm.std() < 1e-6:
                return dm
            return dm / dm.std() * r["d"].std()
        variants["sector12_scaled"] = scaled

        print("\n%-18s %8s %8s %8s   (fit)  %8s %8s   (held-out)"
              % ("variant", "pars", "bd1px", "dDice", "bd1px", "dDice"))
        npar = {"sector12": NSEC, "fourier1": 3, "fourier2": 5, "fourier4": 9,
                "fourier8": 17, "const": 1, "meanfield": NPT,
                "sector12_scaled": NSEC}
        summary = {}
        for k, fn in variants.items():
            vf = np.array([score(apply_field(r, fn(r)), r["gt"]) for r in fit]).mean(axis=0)
            vh = np.array([score(apply_field(r, fn(r)), r["gt"]) for r in hold]).mean(axis=0)
            print("%-18s %8d %8.2f %+8.3f   %8.2f %+8.3f"
                  % (k, npar[k], vf[1], vf[1] - base["fit"]["ctrl"][1],
                     vh[1], vh[1] - base["hold"]["ctrl"][1]))
            summary[k] = {"pars": npar[k],
                          "fit_bd1px": float(vf[1]), "fit_ddice": float(vf[0] - base["fit"]["ctrl"][0]),
                          "hold_bd1px": float(vh[1]), "hold_ddice": float(vh[0] - base["hold"]["ctrl"][0]),
                          "hold_dbd1px": float(vh[1] - base["hold"]["ctrl"][1])}
        print("reference: ctrl bd1px fit %.2f / hold %.2f | oracle bd1px fit %.2f / hold %.2f"
              % (base["fit"]["ctrl"][1], base["hold"]["ctrl"][1],
                 base["fit"]["oracle"][1], base["hold"]["oracle"][1]))
        res[tag] = {"base": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in base.items()},
                    "variants": summary}

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\n[68] wrote %s  total %.0fs" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()

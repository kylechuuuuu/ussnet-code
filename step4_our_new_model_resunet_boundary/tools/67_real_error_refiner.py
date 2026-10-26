#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""67 -- train the refiner on REAL stage-1 errors instead of synthetic perturbations.

Why this is the decisive experiment
-----------------------------------
§5.6, §5.7 and §5.9 all trained the refiner on `(GT contour + synthetic smooth
perturbation)` and all of them were harmful on held-out images.  Two things were
wrong with that protocol:

  * the synthetic perturbation is 3.6x smoother than the real error (§5.8), and
    even the spectrum-matched generator only moved the held-out bd1px from
    -2.82 to -2.38 -- still negative (§5.9);
  * more fundamentally, a RANDOM perturbation is unpredictable from the input,
    so the best a model can learn is its average.  The REAL stage-1 error is
    systematic (it follows anatomy and the measured angular profile), so it is
    in principle learnable.

This script changes exactly that: the training contour IS the real stage-1
contour and the target IS the real residual `-sdf_GT`.  The protocol is clean
because stage-1 was trained on fold-0 train only, so its predictions on the
fold-0 VAL images are legitimate held-out predictions.  20 of the 31 val images
are used to fit, 11 are held out.

Everything else is inherited from tool 64 (same Refiner2, same 4-channel context
input, same loss and schedule), and tool 64's `build` is monkey-patched.

Usage:
  python tools/67_real_error_refiner.py --inp ctx --ntrain 20 --epochs 40 \
      --out results/stage2_real_eval.json
"""
import os
import sys
import time
import argparse
import importlib.util

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")


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


def real_build(names, prob_of, img_of, pscal_of_map, inp, rng, nrep=4, verbose=True):
    """Same contract as s64.build, but the base contour is the REAL stage-1 one."""
    P, S, T = [], [], []
    for nm in names:
        gt = s57.load_gt(nm)
        img = img_of(nm)
        prob, m0 = prob_of(nm)
        if gt is None or img is None or prob is None:
            continue
        z, g = s60.prep_image(img)
        ps = pscal_of_map(nm)
        geo0 = s57.largest_cc(m0)
        c = s57.contour_of(geo0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm0 = s58.normals_of(pts, geo0)
        for rep in range(nrep):
            off = int(rng.integers(0, NPT))       # rotate the start point
            p0 = np.roll(pts, -off, axis=0)
            n0 = np.roll(nrm0, -off, axis=0)
            if rep == 0:
                p2 = p0
            else:
                # tiny jitter (<= 1.5 px) for regularisation; the real residual
                # stays the dominant signal
                d = ndimage.gaussian_filter1d(rng.standard_normal(NPT), 2.0,
                                              mode="wrap")
                d = d / (d.std() + 1e-9) * float(rng.uniform(0.0, 1.5))
                p2 = p0 + d[:, None] * n0
            geo = s57.rasterize(p2, geo0.shape)
            nrm2 = s58.normals_of(p2, geo)
            sdf = s58.sdf_of(geo)
            pa, sc, tg = s64.extract2(p2, nrm2, sdf, prob, z, g, ps, inp, gt)
            if tg is None:
                continue
            P.append(pa)
            S.append(sc)
            T.append(tg)
        if verbose:
            print("[67] %s ok" % nm)
    if not P:
        raise SystemExit("no training data")
    return np.concatenate(P), np.concatenate(S), np.concatenate(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="ctx", choices=["ctx", "img"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ntrain", type=int, default=20)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "stage2_real_eval.json"))
    a = ap.parse_args()

    s64.build = real_build            # <- the real-error protocol
    print("[67] training on the REAL stage-1 residual (tool 64 protocol otherwise)")
    sys.argv = ["64_stage2_refiner_context.py",
                "--inp", a.inp,
                "--epochs", str(a.epochs),
                "--ntrain", str(a.ntrain),
                "--iters", str(a.iters),
                "--smooth", str(a.smooth),
                "--bs", str(a.bs),
                "--lr", str(a.lr),
                "--split-seed", str(a.seed),
                "--out", a.out]
    t0 = time.time()
    s64.main()
    print("[67] total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

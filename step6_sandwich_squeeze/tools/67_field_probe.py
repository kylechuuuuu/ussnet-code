#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/tools/67_field_probe.py

CPU probe on the three probability fields of a sandwich run (no GPU):

  * base              p_base > 0.5                     (the ordinary decision)
  * median            median(p_cautious, p_base, p_amplified) > 0.5
                      -- step-5's squeeze projection, applied pointwise
  * midpoint          0.5*(p_cautious + p_amplified) > 0.5
  * snap(base)        free-window image-evidence snap of the base mask
  * snap(median)      the same snap applied to the median mask
  * ens(base, <png>)  mask-level combination with another model's PNG
                      (mean > 0.5 / AND / OR) -- e.g. ensemble with step-4 armB

Everything is scored with the step-3/step-4 bdice@1px on the shared fold-0
validation split, so the numbers sit directly next to the arbitration tables.

usage:
  python tools/67_field_probe.py --npz-base-dir <...base_nifti> \
      --npz-cautious-dir <...> --npz-amplified-dir <...> \
      --ens-png-dir <other model png dir> --split val --tag probe
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEP6 = os.path.dirname(HERE)
USS = os.path.dirname(STEP6)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import read_split                                    # noqa: E402

spec = importlib.util.spec_from_file_location(
    "arb63", os.path.join(HERE, "63_arbitrate.py"))
arb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-base-dir", required=True)
    ap.add_argument("--npz-cautious-dir", required=True)
    ap.add_argument("--npz-amplified-dir", required=True)
    ap.add_argument("--ens-png-dir", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--maxstep", type=float, default=4.0)
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--results-dir", default=os.path.join(STEP6, "results"))
    args = ap.parse_args()

    train_names, val_names = read_split(USS)
    names = train_names if args.split == "train" else val_names

    keys = ["base", "median", "midpoint", "snap_base", "snap_median"]
    if args.ens_png_dir:
        keys += ["ens_mean", "ens_and", "ens_or"]
    acc = {k: [] for k in keys}
    n = 0

    for name in names:
        stem = arb.stem_of(name)
        gt = arb.s57.load_gt(stem)
        if gt is None:
            continue
        try:
            pb = arb.align_prob(arb.load_prob(args.npz_base_dir, stem), gt.shape)
            pc = arb.align_prob(arb.load_prob(args.npz_cautious_dir, stem), gt.shape)
            pa = arb.align_prob(arb.load_prob(args.npz_amplified_dir, stem), gt.shape)
            img = arb.load_image(stem)
        except Exception as e:
            print("[67] WARN %s: %s" % (stem, e))
            continue
        if img.shape != gt.shape:
            continue
        M0 = arb.s57.largest_cc(pb > 0.5)
        med = arb.s57.largest_cc(np.median(np.stack([pc, pb, pa]), axis=0) > 0.5)
        mid = arb.s57.largest_cc(0.5 * (pc + pa) > 0.5)

        c = arb.s57.contour_of(M0)
        pts = arb.s57.resample_closed(c, arb.NPT, 0)
        nrm = arb.s58.normals_of(pts, M0)
        imgf = arb.ndimage.gaussian_filter(img, args.sigma)
        G = arb.ridge_profile(imgf, pts, nrm, arb.TS, 1.0)
        t_free = arb.pick(G, arb.TS, args.frac,
                          -np.full(len(pts), args.maxstep),
                          np.full(len(pts), args.maxstep))
        snap_base, _ = arb.apply_t(M0, pts, nrm, t_free)

        c2 = arb.s57.contour_of(med)
        pts2 = arb.s57.resample_closed(c2, arb.NPT, 0)
        nrm2 = arb.s58.normals_of(pts2, med)
        G2 = arb.ridge_profile(imgf, pts2, nrm2, arb.TS, 1.0)
        t2 = arb.pick(G2, arb.TS, args.frac,
                      -np.full(len(pts2), args.maxstep),
                      np.full(len(pts2), args.maxstep))
        snap_med, _ = arb.apply_t(med, pts2, nrm2, t2)

        acc["base"].append(arb.s57.bdice(M0, gt, 1))
        acc["median"].append(arb.s57.bdice(med, gt, 1))
        acc["midpoint"].append(arb.s57.bdice(mid, gt, 1))
        acc["snap_base"].append(arb.s57.bdice(snap_base, gt, 1))
        acc["snap_median"].append(arb.s57.bdice(snap_med, gt, 1))

        if args.ens_png_dir:
            other = arb.load_mask(args.ens_png_dir, stem)
            if other is not None and other.shape == gt.shape:
                o = arb.s57.largest_cc(other)
                acc["ens_mean"].append(arb.s57.bdice((0.5 * (M0.astype(float) + o) > 0.5), gt, 1))
                acc["ens_and"].append(arb.s57.bdice(M0 & o, gt, 1))
                acc["ens_or"].append(arb.s57.bdice(M0 | o, gt, 1))
        n += 1

    print("[67] ==== %s | split %s | n=%d | frac %.2f maxstep %.1f ===="
          % (args.tag, args.split, n, args.frac, args.maxstep))
    out = {"tag": args.tag, "split": args.split, "n": n,
           "params": {"frac": args.frac, "maxstep": args.maxstep, "sigma": args.sigma,
                      "ens_png_dir": args.ens_png_dir}, "mean": {}}
    for k in keys:
        if acc[k]:
            out["mean"][k] = float(np.mean(acc[k]))
            print("[67]   %-12s bdice@1px %6.2f" % (k, out["mean"][k]))
    os.makedirs(args.results_dir, exist_ok=True)
    p = os.path.join(args.results_dir, "field_probe_%s.json" % args.tag)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[67] written %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

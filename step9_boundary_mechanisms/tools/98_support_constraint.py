#!/usr/bin/env python3
"""98_support_constraint.py -- mechanism 2 candidate C: evidential-support constraint.

96 showed the money is not in average boundary accuracy but in the TAIL: repair
only the 14% of points with |t_gt|>10 and bdice goes 34.5 -> 45.7 while HD95 goes
28.3 -> 12.6.  HD95 (and therefore step8's "usable rate") is made entirely of
these coherent 40-70 px excursions, which are predominantly OVERSHOOT into the
bladder / anterior fat, i.e. the base head assigns p>0.5 to a region that is
weakly supported.

Mechanism 1 asked "how far may the boundary move?" via the sandwich interval and
then moved the contour along normals.  This asks the same question from a
different, non-contour direction: how far may the final mask be from the region
the model is CONFIDENT about?  We cap the geodesic distance from the
high-confidence core (p > CORE) and rebuild.

Part A runs a morphology battery (thin protrusions are the visible shape of the
spike) for comparison.  Part B is the support cap.
CPU only.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage
from skimage import morphology as skm

HERE = os.path.dirname(os.path.abspath(__file__))
STEP9 = os.path.dirname(HERE)
USS = os.path.dirname(STEP9)
STEP6 = os.path.join(USS, "step6_sandwich_squeeze")
STEP2 = os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts")
sys.path.insert(0, os.path.join(STEP6, "tools"))
sys.path.insert(0, STEP2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = _load("arb", os.path.join(STEP6, "tools", "63_arbitrate.py"))
ev = _load("eval_lib", os.path.join(STEP2, "eval_lib.py"))
s57 = arb.s57
NPZ = os.path.join(STEP6, "predictions", "sq2resencw25_checkpoint_best_base_nifti")


def usable(dice, bd, hd):
    return (dice >= 90.0) & (bd >= 20.0) & (hd <= 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=NPZ)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cores", default="0.7,0.8,0.9,0.95")
    ap.add_argument("--radii", default="2,4,6,8,12,16,24,40")
    ap.add_argument("--morph", default="1,2,3,5,8,12")
    ap.add_argument("--out-dir", default=os.path.join(STEP9, "results"))
    ap.add_argument("--out", default="support_constraint_sq2resencw25.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cores = [float(x) for x in args.cores.split(",")]
    radii = [float(x) for x in args.radii.split(",")]
    morph = [float(x) for x in args.morph.split(",")]

    train, val = arb.read_split(USS)
    names = train if args.split == "train" else val
    if args.limit:
        names = names[:args.limit]

    per_case = []
    for name in names:
        stem = arb.stem_of(name)
        gt = s57.load_gt(stem)
        arr = arb.load_prob(args.npz_dir, stem)
        if gt is None or arr is None:
            continue
        p = arb.align_prob(arr, gt.shape)
        M0 = s57.largest_cc(p > 0.5)
        cands = {"raw": M0}
        for r in morph:
            g = skm.disk(int(r))
            cands["open%g" % r] = s57.largest_cc(skm.binary_opening(M0, g))
            cands["close%g" % r] = s57.largest_cc(skm.binary_closing(M0, g))
            # opening by reconstruction: drop components with no disk of radius r
            cands["oprec%g" % r] = s57.largest_cc(
                skm.reconstruction(skm.binary_erosion(M0, g), M0, method="dilation"))
        for c in cores:
            core = s57.largest_cc(p > c)
            if core.sum() < 16:
                continue
            d = ndimage.distance_transform_edt(~core)
            for r in radii:
                M = s57.largest_cc((p > 0.5) & (d <= r))
                if M.sum() > 0:
                    cands["cap_c%g_r%g" % (c, r)] = M
        row = {"name": stem, "p_max": float(p.max()), "area0": int(M0.sum())}
        for k, m in cands.items():
            m = np.asarray(m, dtype=bool)
            hd, assd, _ = ev.surface_metrics(m, gt)
            bd = s57.bdice(m, gt, tol=1)
            dc = s57.dice(m, gt)
            row["dice_" + k], row["bdice_" + k] = dc, bd
            row["hd95_" + k], row["assd_" + k] = hd, assd
            row["usable_" + k] = float(usable(dc, bd, hd))
            row["area_" + k] = int(m.sum())
        per_case.append(row)
        print("%-22s raw d %5.2f b %5.2f hd %6.2f | cap_c0.8_r6 b %5.2f hd %6.2f | oprec3 b %5.2f hd %6.2f" %
              (stem, row["dice_raw"], row["bdice_raw"], row["hd95_raw"],
               row.get("bdice_cap_c0.8_r6", np.nan), row.get("hd95_cap_c0.8_r6", np.nan),
               row.get("bdice_oprec3", np.nan), row.get("hd95_oprec3", np.nan)))
        sys.stdout.flush()

    keys = [k[6:] for k in per_case[0] if k.startswith("bdice_")]
    acc = {}
    for k in keys:
        dd = np.array([r["dice_" + k] for r in per_case], float)
        bb = np.array([r["bdice_" + k] for r in per_case], float)
        hh = np.array([r["hd95_" + k] for r in per_case], float)
        aa = np.array([r["assd_" + k] for r in per_case], float)
        uu = np.array([r["usable_" + k] for r in per_case], float)
        acc[k] = {"dice_mean": float(np.nanmean(dd)), "bdice1_mean": float(np.nanmean(bb)),
                  "hd95_mean": float(np.nanmean(hh)), "hd95_median": float(np.nanmedian(hh)),
                  "assd_mean": float(np.nanmean(aa)), "usable_rate": float(np.nanmean(uu)),
                  "delta_bdice": float(np.nanmean(bb) - np.nanmean([r["bdice_raw"] for r in per_case])),
                  "win": int(np.nansum(bb > np.array([r["bdice_raw"] for r in per_case]))),
                  "loss": int(np.nansum(bb < np.array([r["bdice_raw"] for r in per_case])))}
    out = {"split": args.split, "n_cases": len(per_case), "cores": cores,
           "radii": radii, "morph": morph, "per_case": per_case, "summary": acc}
    with open(os.path.join(args.out_dir, args.out), "w") as fh:
        json.dump(out, fh, indent=1)

    print("=" * 104)
    print("SUPPORT CONSTRAINT  split=%s n=%d" % (args.split, len(per_case)))
    print("  %-18s %7s %8s %8s %8s %7s  %8s" %
          ("variant", "Dice", "bdice1", "dBD", "HD95", "HDmed", "usable"))
    for k, d in sorted(acc.items(), key=lambda kv: (-kv[1]["usable_rate"], -kv[1]["bdice1_mean"]))[:26]:
        print("  %-18s %7.3f %8.3f %+8.3f %8.2f %7.2f  %8.3f" %
              (k, d["dice_mean"], d["bdice1_mean"], d["delta_bdice"], d["hd95_mean"],
               d["hd95_median"], d["usable_rate"]))


if __name__ == "__main__":
    main()

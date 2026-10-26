#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""69 -- probability-level ensemble of the fingerprint arms and the baselines.

Motivation (measured, not assumed): the fingerprint arms and the nnU-Net baseline
do NOT make the same mistake.  From comparison_fp.md, per image:

    nnU-Net best   FN 2799 / FP 3844   (over-segments)
    arm A          FN 2750 / FP 3777   (over-segments, slightly less)
    arm B          FN 3038 / FP 3365   (under-segments relative to nnU-Net)

and Dice is a statistical tie (96.106 / 96.112 / 96.159, paired Wilcoxon
p = 0.92 / 0.72).  A tie in the mean with decorrelated errors is exactly the
regime where averaging probabilities helps.

This script averages the saved foreground probabilities (npz, original image
grid, channel 1) of any subset of members, thresholds at 0.5 and scores every
image with the shared eval_lib, so the numbers are directly comparable with
comparison_fp.md.

Selection caveat: the table reports many combinations.  Only the pre-specified
uniform averages ("all arms", "arms + baseline") are honest headline numbers;
the per-combination best is reported for completeness and flagged as selected
on these 31 images.

Writes results/ensemble_eval.json and prints a markdown table.
"""
import os
import sys
import json
import time
import importlib.util

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
USS = os.path.dirname(ROOT)
OUT = os.environ.get("ENSEMBLE_OUT",
                    os.path.join(ROOT, "results", "ensemble_eval.json"))

S2 = os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts")
sys.path.insert(0, S2)
import eval_lib  # noqa: E402

spec = importlib.util.spec_from_file_location("s57", os.path.join(TOOLS, "57_shape_prior_ceiling.py"))
s57 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s57)

P = os.path.join(ROOT, "predictions")
S1P = os.path.join(USS, "step1_nnunet_baseline", "predictions")

MEMBERS = {
    "nnB": os.path.join(S1P, "nnunet_2d_fold0_checkpoint_best_nifti"),
    "nnF": os.path.join(S1P, "nnunet_2d_fold0_checkpoint_final_nifti"),
    "Ab": os.path.join(P, "fpA_stock_fploss_checkpoint_best_nifti"),
    "Af": os.path.join(P, "fpA_stock_fploss_checkpoint_final_nifti"),
    "Bb": os.path.join(P, "fpB_resenc_fploss_dir_checkpoint_best_nifti"),
    "Bf": os.path.join(P, "fpB_resenc_fploss_dir_checkpoint_final_nifti"),
    "Cb": os.path.join(P, "fp_armC_preenc_checkpoint_best_nifti"),
    "Cf": os.path.join(P, "fp_armC_preenc_checkpoint_final_nifti"),
    "Db": os.path.join(P, "fp_armD_preenc_r34_checkpoint_best_nifti"),
    "Df": os.path.join(P, "fp_armD_preenc_r34_checkpoint_final_nifti"),
    "Eb": os.path.join(P, "fp_armE_zone3_checkpoint_best_nifti"),
    "Ef": os.path.join(P, "fp_armE_zone3_checkpoint_final_nifti"),
}

COMBOS = [
    ("nnB", ["nnB"]),
    ("nnF", ["nnF"]),
    ("Ab", ["Ab"]),
    ("Af", ["Af"]),
    ("Bb", ["Bb"]),
    ("Bf", ["Bf"]),
    ("Ab+Bb", ["Ab", "Bb"]),
    ("Af+Bf", ["Af", "Bf"]),
    ("Ab+Af+Bb+Bf", ["Ab", "Af", "Bb", "Bf"]),
    ("Ab+Bb+nnB", ["Ab", "Bb", "nnB"]),
    ("Ab+Af+Bb+Bf+nnB", ["Ab", "Af", "Bb", "Bf", "nnB"]),
    ("Cb", ["Cb"]),
    ("Cf", ["Cf"]),
    ("Ab+Bb+Cb", ["Ab", "Bb", "Cb"]),
    ("Ab+Af+Bb+Bf+Cb+Cf", ["Ab", "Af", "Bb", "Bf", "Cb", "Cf"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+nnB", ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "nnB"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "nnB", "nnF"]),
    ("Cb+Cf+Db+Df", ["Cb", "Cf", "Db", "Df"]),
    ("Cb+Cf+Db+Df+nnB", ["Cb", "Cf", "Db", "Df", "nnB"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+Db+Df", ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+Db+Df+nnB",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "nnB"]),
    ("Ab+Af+Bb+Bf+Cb+Cf+Db+Df+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "nnB", "nnF"]),
    ("Eb", ["Eb"]),
    ("Ef", ["Ef"]),
    ("Eb+nnB", ["Eb", "nnB"]),
    ("Cb+Cf+Db+Df+Eb+Ef", ["Cb", "Cf", "Db", "Df", "Eb", "Ef"]),
    ("AB+CDE", ["Ab", "Bb", "Cb", "Db", "Eb"]),
    ("AB+CDE+nnB", ["Ab", "Bb", "Cb", "Db", "Eb", "nnB"]),
    ("all-arms-best+nnB",
     ["Ab", "Bb", "Cb", "Db", "Eb", "nnB"]),
    ("all-arms-both+nnB+nnF",
     ["Ab", "Af", "Bb", "Bf", "Cb", "Cf", "Db", "Df", "Eb", "Ef",
      "nnB", "nnF"]),
]


def case_map():
    import csv
    m = {}
    with open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")) as f:
        for r in csv.DictReader(f):
            m[r["original_filename"]] = r["case_id"]
    return m


def main():
    t0 = time.time()
    tr, te = eval_lib.read_split(USS, exp="ultrasound_prostate_73")
    cmap = case_map()
    print("[69] val images: %d" % len(te))

    missing = [k for k, d in MEMBERS.items() if not os.path.isdir(d)]
    if missing:
        print("[69] MISSING member dirs: %s" % missing)

    # ---- load every member's foreground probability once
    prob = {k: {} for k in MEMBERS}
    for nm in te:
        cid = cmap.get(nm)
        if cid is None:
            print("[69] no case id for %s" % nm)
            continue
        for k, d in MEMBERS.items():
            p = os.path.join(d, cid + ".npz")
            if os.path.exists(p):
                z = np.load(p)
                pr = z["probabilities"]
                if pr.ndim == 4 and pr.shape[0] >= 3:
                    # 3-class zone model: foreground = union = 1 - P(background)
                    pa = 1.0 - np.asarray(pr[0], dtype=np.float32)
                else:
                    pa = np.asarray(pr[1], dtype=np.float32)
                pa = np.squeeze(pa).astype(np.float32)
                assert pa.ndim == 2, (k, nm, pa.shape)
                prob[k][nm] = pa
            else:
                prob[k][nm] = None
    for k in MEMBERS:
        got = sum(1 for v in prob[k].values() if v is not None)
        print("[69] member %-4s %d/%d" % (k, got, len(te)))

    gt = {nm: (s57.load_gt(nm) > 0) for nm in te}

    res = {}
    per_image = {}
    for label, mem in COMBOS:
        rows = []
        dice_list, bd1_list = [], []
        for nm in te:
            arrs = [prob[k][nm] for k in mem]
            arrs = [a for a in arrs if a is not None]
            if not arrs:
                continue
            m = (np.mean(np.stack(arrs, axis=0), axis=0) > 0.5)
            g = gt[nm]
            met = eval_lib.all_metrics(m, g)
            b1 = s57.bdice(m, g, 1)
            b2 = s57.bdice(m, g, 2)
            rows.append({"name": nm, "dice": met["dice"], "bd1": b1, "bd2": b2,
                         "hd95": met["hd95"], "assd": met["assd"],
                         "vol_diff_pct": met["vol_diff_pct"],
                         "precision": met["precision"], "recall": met["recall"]})
            dice_list.append(met["dice"])
            bd1_list.append(b1)
        res[label] = {
            "members": mem, "n": len(rows),
            "dice": float(np.mean(dice_list)),
            "dice_std": float(np.std(dice_list)),
            "bd1px": float(np.mean(bd1_list)),
            "bd2px": float(np.mean([r["bd2"] for r in rows])),
            "hd95": float(np.mean([r["hd95"] for r in rows])),
            "assd": float(np.mean([r["assd"] for r in rows])),
            "vol_diff_pct": float(np.mean([np.abs(r["vol_diff_pct"]) for r in rows])),
            "precision": float(np.mean([r["precision"] for r in rows])),
            "recall": float(np.mean([r["recall"] for r in rows])),
        }
        per_image[label] = {r["name"]: r["dice"] for r in rows}

    # ---- paired Wilcoxon of every combo against nnB
    try:
        from scipy.stats import wilcoxon
        have_w = True
    except Exception:
        have_w = False

    ref = per_image.get("nnB", {})
    for label in res:
        if label == "nnB" or not have_w:
            continue
        common = [nm for nm in te if nm in ref and nm in per_image[label]]
        if len(common) < 6:
            continue
        a = np.array([per_image[label][nm] for nm in common])
        b = np.array([ref[nm] for nm in common])
        try:
            res[label]["p_vs_nnB_dice"] = float(wilcoxon(a, b).pvalue)
        except Exception:
            res[label]["p_vs_nnB_dice"] = None
        res[label]["wins_vs_nnB"] = int((a > b + 1e-9).sum())
        res[label]["losses_vs_nnB"] = int((a < b - 1e-9).sum())

    print("\n%-24s %7s %7s %7s %7s %7s %8s %8s" %
          ("combo", "Dice", "bd1px", "bd2px", "HD95", "vol%", "p_vs_nnB", "W/L"))
    for label, _ in COMBOS:
        r = res[label]
        p = r.get("p_vs_nnB_dice")
        wl = "%s/%s" % (r.get("wins_vs_nnB", "-"), r.get("losses_vs_nnB", "-"))
        print("%-24s %7.3f %7.2f %7.2f %7.2f %7.2f %8s %8s" %
              (label, r["dice"], r["bd1px"], r["bd2px"],
               r["hd95"], r["vol_diff_pct"], ("%.4f" % p) if p is not None else "-", wl))

    with open(OUT, "w") as f:
        json.dump({"res": res, "per_image_dice": per_image}, f, indent=2, ensure_ascii=False)
    print("\n[69] wrote %s  total %.0fs" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()

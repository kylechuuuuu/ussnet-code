#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99ae_domain_shift_attribution.py

Two questions about the acquisition-shift collapse found in step8:

  A. ATTRIBUTION.  Under compound_B the shipped model (SQ2res-w25, the sandwich)
     falls from 96.4 to 79.0 mean Dice and 5/31 cases lose the organ completely
     (Dice < 10, prediction centroid 200-320 px from the truth), while plain
     nnU-Net 1000ep only falls to 93.2 (worst case 46.8).  Is that the
     sandwich's three-head objective, the ResEncUNet backbone, or the 500-epoch
     budget?  Non-sandwich same-architecture / same-budget arms are run through
     the identical predictor by tools/99ab_attribution_runs.sh.

  B. ABLATION.  Every ingredient of compound_B is nearly harmless alone
     (tgc_m60 96.35, shadow 96.13, blur_35 94.38, speckle_35 96.04,
     gamma_060 96.35) yet the chain costs 17 Dice points.  Leave-one-out
     variants (tools/99ac + 99ad) say which ingredient carries the damage.

Metrics are computed with step8's own helpers (82_clinical_metrics.py) so the
numbers sit in the same table as results/shift_metrics.csv.

usage:
  python tools/99ae_domain_shift_attribution.py [--out results/attribution_shift.json]
"""
import argparse
import csv
import importlib
import json
import os
import sys

import numpy as np

USS = "/media/72TB_disk/personal/zk/USSnet"
S8 = os.path.join(USS, "step8_domain_shift")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
sys.path.insert(0, os.path.join(S8, "tools"))
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))

import eval_lib                                                       # noqa: E402
m82 = importlib.import_module("82_clinical_metrics")                  # noqa: E402
s57 = importlib.import_module("57_shape_prior_ceiling")               # noqa: E402

MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
PREDS = os.path.join(S8, "predictions")
CSV = os.path.join(S8, "results", "shift_metrics.csv")

# the five cases whose prediction leaves the organ under compound_B
COLLAPSE = ["丁全土_001_161201", "李明_001_160415", "何朝良_002_173039",
            "陈靖平_001_110332", "崔志和_001_162508"]

# recognised (tag, condition, head) -> human label.
# non-sandwich arms were produced by tools/99ab_attribution_runs.sh, the
# ablation by tools/99ac + 99ad; everything else is step8's own table.
NEW_TAGS = {"fpB": "ResEnc+fploss 500ep (NO sandwich)",
            "fpA": "stockUNet+fploss 500ep (NO sandwich)",
            "sq2a": "stockUNet+SANDWICH 500ep",
            "s200": "stockUNet 200ep",
            "ab": "ABLATION (SQ2res-w25 base)"}
REF_TAGS = {"sq2shift": "SQ2res-w25 SANDWICH 500ep (shipped)",
            "sq23h": "SQ2res-w25 SANDWICH 500ep (shipped)",
            "stockshift": "stockUNet 1000ep (nnU-Net)",
            "stockttn": "stockUNet 1000ep (nnU-Net, no TTA)",
            "sq2ttn": "SQ2res-w25 SANDWICH (no TTA)"}


def runs_from_json():
    out = {}
    for f in sorted(os.listdir(PREDS)):
        if not f.endswith("_shiftrun.json"):
            continue
        with open(os.path.join(PREDS, f)) as fh:
            r = json.load(fh)
        out[r["tag"]] = r
    return out


def centroid(mask):
    if mask.sum() == 0:
        return None
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def measure_dir(d, names, stems, gt):
    """Dice / bdice@1px / centroid displacement for one prediction directory."""
    rows = []
    for n, stem in zip(names, stems):
        fp = os.path.join(d, stem + ".nii.gz")
        if not os.path.exists(fp):
            continue
        pred = m82.read_nii(fp) > 0.5
        g = gt[n]
        if pred.shape != g.shape:
            continue
        pl, gl = s57.largest_cc(pred), s57.largest_cc(g)
        cp, cg = centroid(pl), centroid(gl)
        disp = float(np.hypot(cp[0] - cg[0], cp[1] - cg[1])) if (cp and cg) else float("nan")
        mm = eval_lib.all_metrics(pred, g, g)
        rows.append({"name": n, "dice": float(mm["dice"]),
                     "bdice1": float(s57.bdice(pred, g, 1)),
                     "centroid_disp": disp, "iou": float(mm["iou"])})
    return rows


def summarise(rows):
    d = np.array([r["dice"] for r in rows], float)
    b = np.array([r["bdice1"] for r in rows], float)
    c = np.array([r["centroid_disp"] for r in rows], float)
    coll = [r for r in rows if r["name"].split(".")[0] in COLLAPSE]
    return {"n": len(rows), "dice": float(d.mean()), "bdice1": float(b.mean()),
            "min_dice": float(d.min()), "n_below80": int((d < 80).sum()),
            "n_below50": int((d < 50).sum()),
            "centroid_disp_mean": float(np.nanmean(c)),
            "centroid_disp_p90": float(np.nanpercentile(c, 90)),
            "collapsed5_dice": [round(float(r["dice"]), 2) for r in coll],
            "collapsed5_disp": [round(float(r["centroid_disp"]), 1) for r in coll],
            "per_case": {r["name"].split(".")[0]: round(r["dice"], 2) for r in rows}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(S9, "results", "attribution_shift.json"))
    args = ap.parse_args()

    train_names, val_names = eval_lib.read_split(USS)
    names, stems = val_names, [os.path.splitext(n)[0] for n in val_names]
    gt = {n: eval_lib.load_binary(os.path.join(MASK_DIR, n)) for n in names}

    res = {"reference_from_csv": {}, "new": {}, "ablation": {}}

    # ---------------------------------------------------------- reference rows
    if os.path.exists(CSV):
        with open(CSV) as fh:
            for r in csv.DictReader(fh):
                if r["model"] not in REF_TAGS:
                    continue
                key = "%s:%s" % (r["model"], r["head"] if r["head"] != "-" else "base")
                d = res["reference_from_csv"].setdefault(key, {}).setdefault(
                    r["condition"], {"dice": [], "bdice1": []})
                d["dice"].append(float(r["dice"]))
                d["bdice1"].append(float(r["bdice1"]))
        for key, conds in res["reference_from_csv"].items():
            for c, v in conds.items():
                v["dice_mean"] = float(np.mean(v["dice"]))
                v["bdice1_mean"] = float(np.mean(v["bdice1"]))
                v["min_dice"] = float(np.min(v["dice"]))
                v["n_below80"] = int((np.array(v["dice"]) < 80).sum())
                v["n_below50"] = int((np.array(v["dice"]) < 50).sum())
                del v["dice"], v["bdice1"]

    # ------------------------------------------------------- new runs on disk
    runs = runs_from_json()
    for tag, label in NEW_TAGS.items():
        run = runs.get(tag)
        if run is None:
            print("[99ae] tag %s not found yet" % tag)
            continue
        tgt = res["ablation"] if tag == "ab" else res["new"]
        for key, d in sorted(run["outputs"].items()):
            if not os.path.isdir(d):
                continue
            rows = measure_dir(d, names, stems, gt)
            if rows:
                tgt["%s|%s" % (tag, key)] = summarise(rows)

    # the compound_B row for the shipped model, same settings, from step8's own run
    ref_dir = os.path.join(PREDS, "sq2shift_compound_B_base_nifti")
    if os.path.isdir(ref_dir):
        res["new"]["sq2shift|compound_B"] = summarise(measure_dir(ref_dir, names, stems, gt))
    ind_dir = os.path.join(PREDS, "sq2shift_in_domain_base_nifti")
    if os.path.isdir(ind_dir):
        res["new"]["sq2shift|in_domain"] = summarise(measure_dir(ind_dir, names, stems, gt))

    # ------------------------------------------------------------------ print
    def line(k, v):
        print("  %-40s Dice %6.2f  bdice1 %6.2f  min %6.2f  <80 %2d  <50 %2d  "
              "disp %6.1f (p90 %6.1f)"
              % (k, v["dice"], v["bdice1"], v["min_dice"], v["n_below80"],
                 v["n_below50"], v["centroid_disp_mean"], v["centroid_disp_p90"]))

    print("\n[99ae] === A. ATTRIBUTION: who collapses under compound_B? ===")
    for k in sorted(res["new"]):
        line(k, res["new"][k])
    print("\n[99ae] reference models (from step8 results/shift_metrics.csv):")
    for k in sorted(res["reference_from_csv"]):
        for c in ("in_domain", "compound_A", "compound_B"):
            v = res["reference_from_csv"][k].get(c)
            if v:
                print("  %-28s %-12s Dice %6.2f  min %6.2f  <80 %2d  <50 %2d"
                      % (k, c, v["dice_mean"], v["min_dice"], v["n_below80"], v["n_below50"]))
    if res["ablation"]:
        print("\n[99ae] === B. ABLATION of compound_B (shipped model, base head) ===")
        for k in sorted(res["ablation"]):
            line(k, res["ablation"][k])
        ref = res["new"].get("sq2shift|compound_B")
        if ref:
            print("\n[99ae] reference: full compound_B  Dice %.2f  bdice1 %.2f  min %.2f"
                  % (ref["dice"], ref["bdice1"], ref["min_dice"]))
            print("[99ae] collapsed-5 Dice under full compound_B: %s"
                  % ref["collapsed5_dice"])
            for k in sorted(res["ablation"]):
                v = res["ablation"][k]
                print("[99ae] %-22s collapsed-5 Dice %s  disp %s"
                      % (k.split("|")[1], v["collapsed5_dice"], v["collapsed5_disp"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n[99ae] wrote %s" % args.out)


if __name__ == "__main__":
    main()

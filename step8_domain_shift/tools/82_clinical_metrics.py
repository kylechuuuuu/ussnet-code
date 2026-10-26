#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_domain_shift/tools/82_clinical_metrics.py

Turn the shifted-domain predictions into the table the report is built on.

Two questions are answered separately, on purpose:

  Q1 (robustness)  how much does the segmentation degrade when the acquisition
                   changes, and is the degradation uniform across cases?
  Q2 (usability)   a mean Dice is not a clinical guarantee.  For every case we
                   also record whether it is USABLE (dice >= 90, boundary dice
                   >= 20, HD95 <= 10 px) and an uncertainty score that can be
                   computed WITHOUT the ground truth, so the numbers can be
                   turned into a referral policy.

Reads every `<pred_root>/*_shiftrun.json` written by 81_predict_shifts.py, so
the (model, head, condition) -> directory mapping is never guessed from names.

usage:
  python tools/82_clinical_metrics.py [--pred-root predictions] [--out results/shift_metrics.csv]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))

import eval_lib                                                       # noqa: E402
import importlib                                                      # noqa: E402
s57 = importlib.import_module("57_shape_prior_ceiling")               # noqa: E402

MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")

# a case is called USABLE only when all three agree -- a good average Dice can
# hide a boundary that is nowhere near the capsule, which is what a clinician
# would actually act on
USABLE_DICE = 90.0
USABLE_BDICE = 20.0
USABLE_HD95 = 10.0
GROSS_DICE = 80.0


def read_nii(path):
    import SimpleITK as sitk
    a = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return np.squeeze(a)


def read_npz_prob(path):
    d = np.load(path)
    for k in ("probabilities", "softmax", "prob"):
        if k in d:
            a = d[k]
            break
    else:
        return None
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = a[1] if a.shape[0] == 2 else a[-1]
    return a


def unc_scores(p):
    """Uncertainty scores that need NO ground truth."""
    out = {}
    band = (p > 0.05) & (p < 0.95)
    out["unc_area_px"] = int(band.sum())
    out["unc_mean"] = float((p * (1.0 - p))[band].mean()) if band.any() else 0.0
    m = (p > 0.5)
    if m.any():
        surf = m & ~eval_lib.ndimage.binary_erosion(m, np.ones((3, 3), bool), border_value=0)
        out["conf_margin"] = float(np.abs(2.0 * p[surf] - 1.0).mean())
    else:
        out["conf_margin"] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default=os.path.join(ROOT, "predictions"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "shift_metrics.csv"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    train_names, val_names = eval_lib.read_split(USS)
    names = train_names if args.split == "train" else val_names
    if args.limit:
        names = names[:args.limit]
    stems = [os.path.splitext(n)[0] for n in names]

    gt, gt_gray = {}, {}
    for n in names:
        g = eval_lib.load_binary(os.path.join(MASK_DIR, n))
        gt[n] = g
        gt_gray[n] = eval_lib.load_gray(os.path.join(MASK_DIR, n))

    runs = []
    for f in sorted(os.listdir(args.pred_root)):
        if f.endswith("_shiftrun.json"):
            with open(os.path.join(args.pred_root, f)) as fh:
                runs.append(json.load(fh))
    if not runs:
        raise SystemExit("no *_shiftrun.json in %s" % args.pred_root)
    print("[82] %d inference runs found" % len(runs))

    rows = []
    for run in runs:
        tag = run["tag"]
        for key, d in run["outputs"].items():
            if not os.path.isdir(d):
                continue
            if "_" in key and key.rsplit("_", 1)[-1] in ("base", "cautious", "amplified"):
                cond, head = key.rsplit("_", 1)
            else:
                cond, head = key, ""
            for n, stem in zip(names, stems):
                fp = os.path.join(d, stem + ".nii.gz")
                if not os.path.exists(fp):
                    continue
                pred = read_nii(fp) > 0.5
                g = gt[n]
                if pred.shape != g.shape:
                    raise SystemExit("shape mismatch %s %s %s" % (fp, pred.shape, g.shape))
                m = eval_lib.all_metrics(pred, g, gt_gray[n])
                row = {"model": tag, "head": head or "-", "condition": cond,
                       "name": n, "split": args.split,
                       "bdice1": float(s57.bdice(pred, g, 1)),
                       "bdice2": float(s57.bdice(pred, g, 2))}
                for k in ("dice", "precision", "recall", "iou", "vol_diff_pct",
                          "hd95", "assd", "hd100", "pred_pixels", "gt_pixels",
                          "n_pred_cc", "perimeter_pred", "perimeter_gt"):
                    row[k] = float(m[k])
                npz = os.path.join(d, stem + ".npz")
                if os.path.exists(npz):
                    p = read_npz_prob(npz)
                    if p is not None and p.shape == g.shape:
                        row.update(unc_scores(p))
                # ------- clinical usability flags (thresholds fixed a priori) --
                row["usable"] = int(row["dice"] >= USABLE_DICE
                                    and row["bdice1"] >= USABLE_BDICE
                                    and row["hd95"] <= USABLE_HD95)
                row["fail_gross"] = int(row["dice"] < GROSS_DICE)
                row["fail_boundary"] = int(row["bdice1"] < USABLE_BDICE)
                row["fail_vol"] = int(abs(row["vol_diff_pct"]) > 25.0)
                rows.append(row)

    # ---------------------------------------------- sandwich band, where present
    band_by = {}
    for run in runs:
        for key, d in run["outputs"].items():
            if not key.endswith(("_base", "_cautious", "_amplified")):
                continue
            cond, head = key.rsplit("_", 1)
            for stem in stems:
                if os.path.exists(os.path.join(d, stem + ".npz")):
                    band_by.setdefault((run["tag"], cond, stem), {})[head] = d
    n_band = 0
    for row in rows:
        key = (row["model"], row["condition"], os.path.splitext(row["name"])[0])
        trio = band_by.get(key)
        if not trio or len(trio) < 3:
            continue
        g = gt[row["name"]]
        stem = os.path.splitext(row["name"])[0]
        pb = read_npz_prob(os.path.join(trio["base"], stem + ".npz"))
        pc = read_npz_prob(os.path.join(trio["cautious"], stem + ".npz"))
        pa = read_npz_prob(os.path.join(trio["amplified"], stem + ".npz"))
        if pb is None or pc is None or pa is None:
            continue
        M0 = s57.largest_cc(pb > 0.5)
        Min = M0 & (pc > 0.5)
        Mout = M0 | (pa > 0.5)
        band = Mout & ~Min
        per = max(1, int(s57.surface(M0).sum()))
        row["band_px"] = float(band.sum()) / per
        row["band_area_frac"] = float(band.sum()) / max(1, int(M0.sum()))
        gb = s57.surface(g)
        row["contain"] = float((gb & band).sum()) / max(1, int(gb.sum()))
        n_band += 1
    print("[82] band statistics attached to %d rows" % n_band)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    cols = ["model", "head", "condition", "name", "split"] + \
           [c for c in cols if c not in ("model", "head", "condition", "name", "split")]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[82] wrote %s  (%d rows, %d cols)" % (args.out, len(rows), len(cols)))

    # ------------------------------------------------------------- quick view
    models = sorted({r["model"] for r in rows})
    conds = sorted({r["condition"] for r in rows})
    print("\n[82] mean dice / mean bdice@1px / usable-rate by model x condition")
    hdr = "%-16s" % "condition" + "".join("%22s" % m for m in models)
    print(hdr)
    for c in conds:
        line = "%-16s" % c
        for m in models:
            sub = [r for r in rows if r["model"] == m and r["condition"] == c]
            if not sub:
                line += "%22s" % "-"
            else:
                line += "%22s" % ("%.2f/%.1f/%.0f%%" % (
                    np.mean([r["dice"] for r in sub]),
                    np.mean([r["bdice1"] for r in sub]),
                    100.0 * np.mean([r["usable"] for r in sub])))
        print(line)


if __name__ == "__main__":
    main()

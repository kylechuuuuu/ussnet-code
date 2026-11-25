#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step15_teacher_distill_sandwich/tools/eval_pred_dir.py

Score a directory of predicted masks (.nii.gz) against the validation GT of a
fold, using the repository's evaluator (step2 eval_lib + step4 bdice).

usage:
    python tools/eval_pred_dir.py --dataset Dataset001_UltrasoundProstate \
        --pred <validation dir or heads/base> --tag my_run
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STEP2 = os.path.join(ROOT, "step2_compare_unet_vs_nnunet", "scripts")
STEP4 = os.path.join(ROOT, "step4_our_new_model_resunet_boundary", "tools")
sys.path.insert(0, STEP2)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "metrics"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eval_lib = load_mod("eval_lib", os.path.join(STEP2, "eval_lib.py"))
    s57 = load_mod("s57", os.path.join(STEP4, "57_shape_prior_ceiling.py"))
    raw = os.environ["nnUNet_raw"]
    pre = os.environ["nnUNet_preprocessed"]
    splits = json.load(open(os.path.join(pre, args.dataset, "splits_final.json")))
    cases = sorted(splits[args.fold]["val"])

    rows = []
    for c in cases:
        pp = os.path.join(args.pred, c + ".nii.gz")
        gp = os.path.join(raw, args.dataset, "labelsTr", c + ".nii.gz")
        if not (os.path.isfile(pp) and os.path.isfile(gp)):
            continue
        pred = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(pp))) > 0.5
        gt = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(gp))) > 0.5
        if pred.shape != gt.shape:
            continue
        m = eval_lib.all_metrics(pred, gt, None)
        rows.append(dict(case=c, dice=float(m["dice"]), precision=float(m["precision"]),
                         recall=float(m["recall"]), iou=float(m["iou"]),
                         hd95=float(m["hd95"]), assd=float(m["assd"]),
                         vol_diff_pct=float(m["vol_diff_pct"]),
                         bdice1=float(s57.bdice(pred, gt, 1)),
                         bdice4=float(s57.bdice(pred, gt, 4))))

    summ = {"tag": args.tag, "dataset": args.dataset, "n": len(rows),
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "case"}}
    with open(os.path.join(args.out, "eval_%s.json" % args.tag), "w") as fh:
        json.dump(summ, fh, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/scripts/43_ensemble_with_nnunet.py

Probability-space ensemble of the redesigned model and the stock nnU-Net
baseline, then a 0.5 threshold -> PNG masks named after the original files.

Why this is FAIR (and not leakage)
----------------------------------
Both models were trained on fold 0 only, i.e. on the very same 71 cases, and both
were run through the identical nnU-Net plans / preprocessing / sliding-window TTA,
so their .npz probability maps live in the same geometry and the ensemble never
sees a single validation case during training.  (Ensembling across *folds* would
leak here, because every other fold's training list contains fold-0 validation
images - that is why no cross-fold ensemble is used anywhere in this project.)

This is reported ALONGSIDE the single redesigned model, never instead of it: the
headline claim of step 4 is about our model on its own, the ensemble is an extra
row that shows what the pipeline reaches when both models are combined.

usage:
  python 43_ensemble_with_nnunet.py \
      --ours   ../predictions/our_resunet_boundary_2d_fold0_checkpoint_best_nifti \
      --nnunet ../../step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_best_nifti \
      --out    ../predictions/ensemble_ours_nnunet_png \
      --split  val
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
USS_ROOT = os.path.dirname(STEP4)
sys.path.insert(0, os.path.join(USS_ROOT, "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import read_split  # noqa: E402


def load_probs(npz_path):
    with np.load(npz_path) as z:
        key = "probabilities" if "probabilities" in z.files else z.files[0]
        p = np.asarray(z[key])
    p = np.squeeze(p)
    if p.ndim != 3 or p.shape[0] != 2:
        raise RuntimeError(f"{npz_path}: expected 2-channel probabilities, got {p.shape}")
    return p.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="nifti dir of our model (with .npz)")
    ap.add_argument("--nnunet", required=True, help="nifti dir of the nnU-Net baseline (with .npz)")
    ap.add_argument("--out", required=True, help="output PNG folder")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # case_id -> original png file name
    mapping = {}
    with open(os.path.join(USS_ROOT, "step1_nnunet_baseline", "case_id_mapping.csv"),
              encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        i_id, i_name = header.index("case_id"), header.index("original_filename")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) > i_name:
                mapping[parts[i_id]] = parts[i_name]

    train_names, val_names = read_split(USS_ROOT, "ultrasound_prostate_73")
    if args.split == "val":
        wanted = set(val_names)
    elif args.split == "train":
        wanted = set(train_names)
    else:
        wanted = set(train_names) | set(val_names)

    rows = []
    for case_id in sorted(mapping):
        name = mapping[case_id]
        if name not in wanted:
            continue
        f_ours = os.path.join(args.ours, case_id + ".npz")
        f_nn = os.path.join(args.nnunet, case_id + ".npz")
        if not (os.path.isfile(f_ours) and os.path.isfile(f_nn)):
            print(f"[warn] missing npz for {case_id}, skipped")
            continue
        p_ours, p_nn = load_probs(f_ours), load_probs(f_nn)
        if p_ours.shape != p_nn.shape:
            raise RuntimeError(f"{case_id}: geometry mismatch {p_ours.shape} vs {p_nn.shape}")

        fg_ours, fg_nn = p_ours[1], p_nn[1]
        fg = 0.5 * (fg_ours + fg_nn)
        bin_ours, bin_nn, bin_ens = fg_ours > args.threshold, fg_nn > args.threshold, fg > args.threshold
        rows.append(dict(
            case_id=case_id,
            flip_vs_ours=float((bin_ens != bin_ours).mean()) * 100.0,
            flip_vs_nnunet=float((bin_ens != bin_nn).mean()) * 100.0,
            mean_prob_ours=float(fg_ours.mean()),
            mean_prob_nnunet=float(fg_nn.mean()),
            area_ours=int(bin_ours.sum()),
            area_nnunet=int(bin_nn.sum()),
            area_ensemble=int(bin_ens.sum()),
        ))
        Image.fromarray((bin_ens.astype(np.uint8) * 255), mode="L").save(os.path.join(args.out, name))

    n = len(rows)
    if n == 0:
        raise RuntimeError("no cases ensembled")
    print(f"[ensemble] {n} cases -> {args.out}")
    print(f"[ensemble] mean pixels changed vs our model : {np.mean([r['flip_vs_ours'] for r in rows]):.3f} %")
    print(f"[ensemble] mean pixels changed vs nnU-Net   : {np.mean([r['flip_vs_nnunet'] for r in rows]):.3f} %")
    print(f"[ensemble] mean foreground area: ours {np.mean([r['area_ours'] for r in rows]):.0f} px, "
          f"nnU-Net {np.mean([r['area_nnunet'] for r in rows]):.0f} px, "
          f"ensemble {np.mean([r['area_ensemble'] for r in rows]):.0f} px")
    out_csv = os.path.join(STEP4, "metrics", "ensemble_vs_members.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[ensemble] per-case csv -> {out_csv}")


if __name__ == "__main__":
    main()

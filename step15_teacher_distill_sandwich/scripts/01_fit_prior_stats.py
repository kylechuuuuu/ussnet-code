#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step15_teacher_distill_sandwich/scripts/01_fit_prior_stats.py

Fit the frozen teacher statistics on the TRAIN split of one fold:

    mean / PCA basis (k comps) / per-component std   of UltraFedFM features
    mu_fg / mu_bg                                    foreground/background prototypes

They are frozen afterwards: the distillation target is PCA-16 of the teacher
features, not a learnable projection.

usage:
    python scripts/01_fit_prior_stats.py \
        --dataset Dataset001_UltrasoundProstate --fold 0 --k 16
"""
import argparse
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "trainer"))

from us_distill import FrozenTeacher  # noqa: E402


def load2d(path):
    a = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if a.ndim == 3:                      # 2D dataset stored as one-slice volumes
        a = a[0]
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Dataset001_UltrasoundProstate")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2,
                    help="subsample stride over the patch grid used to fit the PCA")
    ap.add_argument("--out", default=os.environ.get("US_DISTILL_STATS", "prior_stats.npz"))
    args = ap.parse_args()

    raw = os.environ.get("nnUNet_raw")
    pre = os.environ.get("nnUNet_preprocessed")
    if not raw or not pre:
        raise SystemExit("set nnUNet_raw and nnUNet_preprocessed first")

    ds_raw = os.path.join(raw, args.dataset)
    splits = json.load(open(os.path.join(pre, args.dataset, "splits_final.json")))
    cases = splits[args.fold]["train"]
    print("dataset %s | fold %d | train cases %d" % (args.dataset, args.fold, len(cases)))

    teacher = FrozenTeacher().cuda().eval()
    vecs, fg_sum, bg_sum, n_fg, n_bg = [], None, None, 0, 0

    with torch.no_grad():
        for i, c in enumerate(cases):
            img = load2d(os.path.join(ds_raw, "imagesTr", c + "_0000.nii.gz"))
            gt = load2d(os.path.join(ds_raw, "labelsTr", c + ".nii.gz"))
            f = teacher(torch.from_numpy(img)[None, None].cuda())      # 1,768,gh,gw
            gh, gw = f.shape[-2:]
            m = torch.from_numpy((gt > 0).astype(np.float32))[None, None].cuda()
            m = F.interpolate(m, size=(gh, gw), mode="nearest")

            fv = f[0].permute(1, 2, 0).reshape(-1, f.shape[1])        # gh*gw,768
            vecs.append(fv[::args.stride].cpu())
            mf = m[0, 0].reshape(-1).bool()
            fg_sum = fv[mf].sum(0) if fg_sum is None else fg_sum + fv[mf].sum(0)
            bg_sum = fv[~mf].sum(0) if bg_sum is None else bg_sum + fv[~mf].sum(0)
            n_fg += int(mf.sum())
            n_bg += int((~mf).sum())
            if (i + 1) % 10 == 0:
                print("  %d/%d" % (i + 1, len(cases)), flush=True)

    A = torch.cat(vecs, 0).float()
    print("feature vectors:", tuple(A.shape))
    mean = A.mean(0)
    Ac = A - mean
    _, S, V = torch.pca_lowrank(Ac, q=args.k, center=False)
    comp = V.t().contiguous()                                        # k,768
    std = (Ac @ comp.t()).std(0).clamp_min(1e-6)

    np.savez(args.out,
             mean=mean.cpu().numpy().astype(np.float32),
             components=comp.cpu().numpy().astype(np.float32),
             std=std.cpu().numpy().astype(np.float32),
             mu_fg=(fg_sum / max(1, n_fg)).cpu().numpy().astype(np.float32),
             mu_bg=(bg_sum / max(1, n_bg)).cpu().numpy().astype(np.float32),
             n_fg=n_fg, n_bg=n_bg,
             train_cases=np.array(cases))
    print("saved", args.out)


if __name__ == "__main__":
    main()

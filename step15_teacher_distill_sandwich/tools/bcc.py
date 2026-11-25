#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step15_teacher_distill_sandwich/tools/bcc.py

OPTIONAL band-consistent calibration (BCC), OFF by default.

    p' = p_base + alpha * (p_amplified - p_cautious)          alpha = 0.25

The three probability maps come from the sandwich heads (base / cautious /
amplified). The correction is positive where the amplified head is more
confident, i.e. it pushes the boundary outwards.

Evidence (see docs/RESULTS.md) says this is not a reliable global gain: it
helps when the model massively over-segments, and costs bDice/volume elsewhere.
Use it only as an ablation, or gate it by an abnormal predicted volume.

usage (batch):
    python tools/bcc.py \
        --base-dir heads/base --cautious-dir heads/cautious \
        --amplified-dir heads/amplified --out-dir heads/bcc \
        --images $nnUNet_raw/DatasetXXX/imagesTr --alpha 0.25
"""
import argparse
import os

import numpy as np
import SimpleITK as sitk


def band_consistent_calibration(p_base, p_cautious, p_amplified, alpha=0.25):
    return np.clip(p_base + alpha * (p_amplified - p_cautious), 0.0, 1.0)


def read_prob(path):
    """Read a nnU-Net .npz softmax map and return the foreground channel."""
    z = np.squeeze(np.load(path)["probabilities"])
    if z.ndim == 3:                     # (2, H, W) -> foreground
        z = z[1] if z.shape[0] == 2 else z[0]
    return np.squeeze(z).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--cautious-dir", required=True)
    ap.add_argument("--amplified-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--images", default=None,
                    help="optional imagesTr dir; if given, write .nii.gz masks "
                         "with the same geometry")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.base_dir) if f.endswith(".npz"))
    if not files:
        raise SystemExit("no .npz files in %s" % args.base_dir)

    for fn in files:
        case = fn[:-4]
        pb = read_prob(os.path.join(args.base_dir, fn))
        pc = read_prob(os.path.join(args.cautious_dir, fn))
        pa = read_prob(os.path.join(args.amplified_dir, fn))
        p = band_consistent_calibration(pb, pc, pa, args.alpha)
        mask = (p > args.threshold).astype(np.uint8)

        if args.images:
            ref = sitk.ReadImage(os.path.join(args.images, case + "_0000.nii.gz"))
            out = sitk.GetImageFromArray(mask)
            out.CopyInformation(ref)
            sitk.WriteImage(out, os.path.join(args.out_dir, case + ".nii.gz"))
        else:
            np.savez_compressed(os.path.join(args.out_dir, case + ".npz"),
                                probabilities=np.stack([1.0 - p, p], 0))
        print(case, "fg %.2f%%" % (100.0 * mask.mean()))


if __name__ == "__main__":
    main()

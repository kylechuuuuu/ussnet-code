#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_domain_shift/tools/81_predict_shifts.py

One in-process nnU-Net inference pass over a whole set of shift conditions.
Model is loaded ONCE; each condition is staged as a plain nnU-Net raw folder
(<stem>_0000.nii.gz) and pushed through the SAME predictor settings as the
step-1 baseline / step-6 sandwich inference (sliding window 0.5, Gaussian
weighting, mirroring TTA, everything on device).

Works for both the sandwich trainers (pass --heads base,cautious,amplified) and
the stock nnU-Net trainer (omit --heads).

usage:
  python tools/81_predict_shifts.py \
      --model-folder <results>/Dataset001_UltrasoundProstate/<Trainer>__nnUNetPlans__2d \
      --checkpoint checkpoint_best.pth [--heads base,cautious,amplified] \
      --image-root data/shift --conditions all --split val \
      --tag sq2 --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
sys.path.insert(0, os.path.join(USS, "step6_sandwich_squeeze", "trainer"))

from shift_lib import SHIFTS, GROUPS  # noqa: E402
import eval_lib  # noqa: E402


def find_wrapper(net):
    from sandwich_net import SandwichHeadUNet
    for _ in range(3):
        if isinstance(net, SandwichHeadUNet):
            return net
        nxt = getattr(net, "_orig_mod", None) or getattr(net, "module", None)
        if nxt is None:
            return None
        net = nxt
    return None


def png_to_nifti_dir(names, img_dir, out_dir):
    """Stage PNGs as <stem>_0000.nii.gz (identity 2-D geometry, 1 mm spacing)."""
    import SimpleITK as sitk
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for name in names:
        stem = os.path.splitext(name)[0]
        dst = os.path.join(out_dir, stem + "_0000.nii.gz")
        if not os.path.exists(dst):
            arr = np.array(Image.open(os.path.join(img_dir, name)).convert("L"), dtype=np.uint8)
            im = sitk.GetImageFromArray(arr)
            im.SetSpacing((1.0, 1.0))
            sitk.WriteImage(im, dst)
        todo.append(dst)
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-folder", required=True)
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--heads", default="",
                    help="comma list of sandwich heads; empty = stock network")
    ap.add_argument("--image-root", default=os.path.join(ROOT, "data", "shift"))
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw-root", default=os.path.join(ROOT, "data", "raw_shift"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "predictions"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--step-size", type=float, default=0.5)
    ap.add_argument("--no-probabilities", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("nnUNet_compile", "False")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if args.conditions == "all":
        conds = list(SHIFTS.keys())
    elif args.conditions in GROUPS:
        conds = list(GROUPS[args.conditions])
    else:
        conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    heads = [h.strip() for h in args.heads.split(",") if h.strip()] or [""]

    train_names, val_names = eval_lib.read_split(USS)
    names = train_names if args.split == "train" else val_names
    if args.limit:
        names = names[:args.limit]

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.no_tta,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder, use_folds=(int(args.fold),), checkpoint_name=args.checkpoint)
    wrapper = find_wrapper(predictor.network)
    if heads != [""] and wrapper is None:
        raise RuntimeError("--heads given but the checkpoint is not a SandwichHeadUNet")
    print("[81] tag=%s heads=%s TTA=%s n=%d conditions=%d device=%s"
          % (args.tag, ",".join(h or "-" for h in heads), not args.no_tta,
             len(names), len(conds), args.device), flush=True)

    done = {}
    t0 = time.time()
    for cond in conds:
        img_dir = os.path.join(args.image_root, cond, "images")
        if not os.path.isdir(img_dir):
            print("[81] SKIP %s (no %s)" % (cond, img_dir), flush=True)
            continue
        raw_dir = os.path.join(args.raw_root, os.path.basename(args.image_root), cond)
        png_to_nifti_dir(names, img_dir, raw_dir)
        for head in heads:
            if head:
                wrapper.set_forward_mode(head)
            suffix = ("_" + head) if head else ""
            out_dir = os.path.join(args.out_root, "%s_%s%s_nifti" % (args.tag, cond, suffix))
            os.makedirs(out_dir, exist_ok=True)
            predictor.predict_from_files(
                raw_dir, out_dir,
                save_probabilities=not args.no_probabilities,
                overwrite=True,
                num_processes_preprocessing=3,
                num_processes_segmentation_export=3,
                folder_with_segs_from_prev_stage=None,
            )
            done["%s%s" % (cond, suffix)] = out_dir
            print("[81] %-16s %-10s -> %s   (%.0fs elapsed)"
                  % (cond, head or "-", out_dir, time.time() - t0), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    meta = {"tag": args.tag, "heads": heads, "model_folder": args.model_folder,
            "checkpoint": args.checkpoint, "split": args.split, "n": len(names),
            "image_root": args.image_root, "outputs": done,
            "seconds": time.time() - t0}
    with open(os.path.join(args.out_root, "%s_shiftrun.json" % args.tag), "w") as f:
        json.dump(meta, f, indent=2)
    print("[81] done: %d passes in %.0fs" % (len(done), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

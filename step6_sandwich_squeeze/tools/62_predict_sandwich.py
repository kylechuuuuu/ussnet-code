#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/tools/62_predict_sandwich.py

Inference for the sandwich squeeze model -- one stock predictor pass PER
VERSION of the segmentation.

The three-version network (trainer/sandwich_net.py) has a `forward_mode`; in
"base"/"cautious"/"amplified" the network returns a normal 2-channel
[background, foreground] logit map of THAT version, so nnU-Net's own predictor
is used completely unchanged: same sliding window, same Gaussian weighting,
same mirroring TTA, same crop-back/resample and the same NIfTI export as the
step-1 baseline.  Run this script once per version:

    base        -> the ordinary segmentation        (基础版本)
    cautious    -> the erosion-supervised bound     (谨慎版本, inside)
    amplified   -> the dilation-supervised bound    (放大版本, outside)

With --save-probabilities each pass also writes <case>.npz (nnU-Net's own
softmax export); tools/63_arbitrate.py turns the three probability fields into
the boundary determination (nesting by construction + interval-clamped image
evidence snapping).

usage:
  python tools/62_predict_sandwich.py \
      --model-folder <results>/Dataset001_UltrasoundProstate/nnUNetTrainerSandwich__nnUNetPlans__2d \
      --checkpoint checkpoint_best.pth --head base|cautious|amplified \
      --input <nnUNet_raw>/Dataset001_UltrasoundProstate/imagesTr \
      --output predictions/sq2_checkpoint_best_base_nifti [--no-tta] [--save-probabilities]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer"))

from sandwich_net import SandwichHeadUNet                                     # noqa: E402


def find_wrapper(net):
    for _ in range(3):
        if isinstance(net, SandwichHeadUNet):
            return net
        nxt = getattr(net, "_orig_mod", None) or getattr(net, "module", None)
        if nxt is None:
            return None
        net = nxt
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-folder", required=True)
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--head", default="base",
                    choices=["base", "cautious", "amplified", "all"])
    ap.add_argument("--input", required=True, help="folder with the *_0000.nii.gz inputs")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-tta", action="store_true", help="disable mirroring TTA")
    ap.add_argument("--save-probabilities", action="store_true")
    ap.add_argument("--step-size", type=float, default=0.5)
    args = ap.parse_args()

    # inference does not need torch.compile; keep it off so the first forward
    # pass is not spent compiling
    os.environ.setdefault("nnUNet_compile", "False")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    os.makedirs(args.output, exist_ok=True)
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.no_tta,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    folds = tuple(int(f) for f in str(args.fold).split(","))
    predictor.initialize_from_trained_model_folder(args.model_folder, use_folds=folds,
                                                  checkpoint_name=args.checkpoint)
    wrapper = find_wrapper(predictor.network)
    if wrapper is None:
        raise RuntimeError("the checkpoint does not build a SandwichHeadUNet - wrong trainer?")
    wrapper.set_forward_mode(args.head)
    print("[62] head = %s | TTA(mirroring) = %s | step size = %.2f"
          % (args.head, not args.no_tta, args.step_size))

    predictor.predict_from_files(
        args.input,
        args.output,
        save_probabilities=args.save_probabilities,
        overwrite=True,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
        folder_with_segs_from_prev_stage=None,
    )
    print("[62] predictions -> %s" % args.output)


if __name__ == "__main__":
    main()

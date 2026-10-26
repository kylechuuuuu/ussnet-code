#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/tools/62_predict_squeeze.py

Inference for the squeeze-theorem model.

The trick that keeps this simple AND stock: the three-head network has a
`fusion` mode (see trainer/squeeze_net.py).  In the modes used here the network
returns a normal 2-channel [background, foreground] logit map that already
contains the fused decision

    main    : p_m                      -- the plain single-head decision
    median  : the projected median of (p_in, p_m, p_out)   <- 夹逼 decision:
                                          the squeeze interval overrides the
                                          middle head wherever it contradicts
                                          both bounds
    inner   : p_in                     -- guaranteed-inside bound
    outer   : p_out                    -- guaranteed-outside bound

so nnU-Net's own predictor can be used unchanged: the same sliding window, the
same Gaussian weighting, the same mirroring TTA, the same crop-back/resample
and the same NIfTI export as the step-1 baseline.  Only the *decision* differs.

usage:
  python tools/62_predict_squeeze.py \
      --model-folder <results>/Dataset001_UltrasoundProstate/nnUNetTrainerProstateSqueeze__nnUNetPlans__2d \
      --checkpoint checkpoint_best.pth --fusion median \
      --input <nnUNet_raw>/Dataset001_UltrasoundProstate/imagesTr \
      --output predictions/sqA_angular_median_best_nifti [--no-tta] [--save-probabilities]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer"))

from squeeze_net import SqueezeHeadUNet                                # noqa: E402


def find_wrapper(net):
    for _ in range(3):
        if isinstance(net, SqueezeHeadUNet):
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
    ap.add_argument("--fusion", default="median",
                    choices=["main", "median", "inner", "outer"])
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
        raise RuntimeError("the checkpoint does not build a SqueezeHeadUNet - wrong trainer?")
    wrapper.set_fusion(args.fusion)
    print("[62] fusion = %s | TTA(mirroring) = %s | step size = %.2f"
          % (args.fusion, not args.no_tta, args.step_size))

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

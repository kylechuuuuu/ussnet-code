#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step15_teacher_distill_sandwich/scripts/72_predict_three_heads.py

Run the trained sandwich model once per head (base / cautious / amplified) on
the validation fold, saving both masks and softmax probabilities (.npz).

usage:
    python scripts/72_predict_three_heads.py \
        --dataset Dataset001_UltrasoundProstate \
        --trainer nnUNetTrainerSandwichDistill --fold 0 --out heads/run

Requires nnUNet_raw / nnUNet_preprocessed / nnUNet_results, and (for the
distillation trainer) US_FM_CHECKPOINT + US_DISTILL_STATS.
"""
import argparse
import json
import os

import torch


def unwrap(net):
    for _ in range(3):
        if hasattr(net, "set_forward_mode"):
            return net
        net = (getattr(net, "_orig_mod", None) or getattr(net, "module", None)
               or getattr(net, "net", None))
        if net is None:
            break
    raise RuntimeError("cannot find SandwichHeadUNet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="base,cautious,amplified")
    args = ap.parse_args()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    raw = os.environ["nnUNet_raw"]
    pre = os.environ["nnUNet_preprocessed"]
    res = os.environ["nnUNet_results"]
    splits = json.load(open(os.path.join(pre, args.dataset, "splits_final.json")))
    val = sorted(splits[args.fold]["val"])
    files = [os.path.join(raw, args.dataset, "imagesTr", c + "_0000.nii.gz")
             for c in val]

    model_dir = os.path.join(res, args.dataset, args.trainer + "__nnUNetPlans__2d")
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True,
                                device=torch.device("cuda:0"),
                                allow_tqdm=False, verbose=False)
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir, use_folds=(args.fold,),
        checkpoint_name=args.checkpoint)
    net = unwrap(predictor.network)
    print("network", type(net).__name__, flush=True)
    for mode in args.modes.split(","):
        net.set_forward_mode(mode)
        outdir = os.path.join(os.path.expanduser(args.out), mode)
        os.makedirs(outdir, exist_ok=True)
        predictor.predict_from_files(
            [[f] for f in files], outdir, save_probabilities=True, overwrite=True,
            num_processes_preprocessing=1, num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        print("wrote", outdir, flush=True)
    net.set_forward_mode("base")


if __name__ == "__main__":
    main()

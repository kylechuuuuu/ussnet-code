#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/28_export_student_masks.py

Export the narrow-ViT student's THREE versions (base / cautious / amplified) as
PNG masks at the ORIGINAL image resolution, in the layout the step-6 boundary
arbitration (step6_sandwich_squeeze/tools/63_arbitrate.py) expects:

    predictions/l2vit_<tag>/{base,cautious,amplified}/<original name>.png

That closes the loop with the project's core innovation: the arbitration takes
the learned interval [base & cautious, base | amplified] and lets image evidence
move the boundary inside it.

The student runs on the 1008x1008 SAM3 canvas (image at top-left, padded), so the
prediction is cropped to the valid region and resampled back to the original
size before saving.

usage: ~/anaconda3/envs/zk/bin/python scripts/28_export_student_masks.py --tag l2vit_200ep [--split val|train|both]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
RUNS = os.path.join(ROOT, "runs")
PRED = os.path.join(ROOT, "predictions")
sys.path.insert(0, HERE)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


l0 = _load("l0e", "10_l0_probe.py")
v27 = _load("v27e", "27_student_narrow_vit.py")

IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")


@torch.no_grad()
def predict_three(net, img_pil, dev):
    W, H = img_pil.size
    raw = np.asarray(img_pil.convert("L"), dtype=np.float32) / 255.0
    x = v27.canvas_input(raw).to(dev)
    logits = net(x)[0]
    lg = F.interpolate(logits.float(), size=v27.TC, mode="bilinear", align_corners=False)
    nw = v27.TC[0]
    nh = min(v27.TC[1], max(1, int(round(H * nw / float(W)))))
    out = {}
    for key, sl in (("base", slice(0, 2)), ("cautious", slice(2, 4)), ("amplified", slice(4, 6))):
        p = torch.softmax(lg[:, sl], 1)[:, 1][0].cpu().numpy()[:nh, :nw]
        m = (p > 0.5).astype(np.uint8) * 255
        out[key] = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST)) > 127
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="both")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--image-root", default=IMG_DIR, help="dir with the images to predict (condition-wise E1)")
    ap.add_argument("--out-dir", default="", help="override output dir (default predictions/l2vit_<tag>)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(os.path.join(RUNS, args.tag, "ckpt.pt"), map_location="cpu", weights_only=False)
    net = v27.ViTStudent(embed=ck["embed"], depth=ck["depth"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    train_names, val_names = v27.fs.read_split(args.fold)
    splits = {"val": val_names, "train": train_names}
    if args.split in splits:
        splits = {args.split: splits[args.split]}

    out_root = args.out_dir or os.path.join(PRED, "l2vit_%s" % args.tag)
    for sp, names in splits.items():
        for key in ("base", "cautious", "amplified"):
            os.makedirs(os.path.join(out_root, key), exist_ok=True)
        for i, n in enumerate(names):
            img = Image.open(os.path.join(args.image_root, n)).convert("L")
            m = predict_three(net, img, dev)
            for key, arr in m.items():
                Image.fromarray((arr * 255).astype(np.uint8)).save(
                    os.path.join(out_root, key, n))
            if (i + 1) % 20 == 0:
                print("  [%s] %d/%d" % (sp, i + 1, len(names)), flush=True)
        # interval diagnostics on this split
        gts = [l0.eval_lib.load_binary(os.path.join(USS, "UltrasoundProstate", "masks", n)) > 0
               for n in names]
        cont, width = [], []
        for n, gt in zip(names, gts):
            b = np.array(Image.open(os.path.join(out_root, "base", n))) > 127
            c = np.array(Image.open(os.path.join(out_root, "cautious", n))) > 127
            a = np.array(Image.open(os.path.join(out_root, "amplified", n))) > 127
            gt_r = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize(
                b.shape[::-1], Image.NEAREST)) > 127
            from scipy import ndimage as ndi
            edge = gt_r & ~ndi.binary_erosion(gt_r, iterations=1, border_value=0)
            inner = b & c
            outer = b | a
            cont.append(float((edge & ~inner & outer).sum() / max(1, edge.sum())))
            width.append(float((outer & ~inner).sum() / max(1, edge.sum())))
        print("[export] %s: n=%d | base in [cautious,amplified] containment %.3f | interval px/edge-px %.2f"
              % (sp, len(names), np.mean(cont), np.mean(width)), flush=True)
    print("[export] wrote %s" % out_root)


if __name__ == "__main__":
    sys.exit(main())

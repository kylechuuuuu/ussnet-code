#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/22_eval_student.py

Evaluate the L2 student on the fold-0 val split: in_domain plus the 17
pre-registered shift conditions of step8, with the SAME metric code as the L0
probe and the step6/step8 tables (eval_lib + 57_shape_prior_ceiling.bdice).

Reported per condition: mean Dice, mean bdice@1px, min Dice, #cases < 80,
centroid shift, and the interval diagnostics of the sandwich part
(fraction of GT boundary pixels inside the [cautious, amplified] band).

usage:
  ~/anaconda3/envs/zk/bin/python scripts/22_eval_student.py --tag l2_student_200ep [--device cuda:0]
"""
import argparse
import csv
import importlib.util
import json
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
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location("l0", os.path.join(HERE, "10_l0_probe.py"))
l0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l0)
spec2 = importlib.util.spec_from_file_location("tr", os.path.join(HERE, "21_train_student.py"))
tr = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(tr)

SHIFT_DIR = os.path.join(USS, "step8_domain_shift", "data", "shift")
CONDITIONS = ["in_domain", "affine_gain_m25", "affine_gain_p25", "blur_20", "blur_35",
              "compound_A", "compound_B", "gamma_060", "gamma_080", "gamma_125",
              "jpeg_15", "jpeg_30", "shadow", "speckle_15", "speckle_35", "tgc_m60", "tgc_p60"]


@torch.no_grad()
def predict(net, img_pil, dev, tta=False):
    img = img_pil.resize((tr.CW, tr.CH), Image.BILINEAR)   # 21_train_student defines CW/CH
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - a.mean()) / (a.std() + 1e-6)
    x = torch.from_numpy(a)[None, None].float().to(dev)
    logits, _ = net(x)
    if tta:                                    # mirror TTA: average the probability fields
        logits2, _ = net(torch.flip(x, dims=[3]))
        logits = 0.5 * (logits + torch.flip(logits2, dims=[3]))
    lg = F.interpolate(logits, size=(tr.CH, tr.CW), mode="bilinear", align_corners=False)
    base = (torch.softmax(lg[:, 0:2], 1)[:, 1] > 0.5)[0].cpu().numpy()
    caut = (torch.softmax(lg[:, 2:4], 1)[:, 1] > 0.5)[0].cpu().numpy()
    amp = (torch.softmax(lg[:, 4:6], 1)[:, 1] > 0.5)[0].cpu().numpy()
    return base, caut, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cond", default="all")
    ap.add_argument("--image-root", default=SHIFT_DIR,
                    help="directory holding <cond>/images/<name>.png (default: step8 shift tree)")
    ap.add_argument("--tag-out", default="")
    ap.add_argument("--tta", action="store_true", help="mirror TTA (average with the flipped view)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(os.path.join(RUNS, args.tag, "ckpt.pt"), map_location="cpu", weights_only=False)
    widths = tuple(ck["widths"]) if ck.get("widths") else None
    net = tr.StudentNet(widths=widths).to(dev).eval()
    net.load_state_dict(ck["net"])
    n_param = sum(p.numel() for p in net.parameters())
    conds = CONDITIONS if args.cond == "all" else [c.strip() for c in args.cond.split(",") if c.strip()]
    image_root = args.image_root
    train_names, val_names = l0.eval_lib.read_split(USS)

    rows = []
    for cond in conds:
        for n in val_names:
            ip = os.path.join(image_root, cond, "images", n)
            if not os.path.exists(ip):
                ip = os.path.join(USS, "UltrasoundProstate", "images", n)
            img = Image.open(ip).convert("L")
            gt = l0.eval_lib.load_binary(os.path.join(USS, "UltrasoundProstate", "masks", n)) > 0
            gt_c = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize(
                (tr.CW, tr.CH), Image.NEAREST)) > 127
            base, caut, amp = predict(net, img, dev, args.tta)
            m = l0.eval_lib.all_metrics(base, gt_c, None)
            # interval diagnostics: is the GT boundary inside [cautious, amplified]?
            from scipy import ndimage as ndi
            b_gt = gt_c & ~ndi.binary_erosion(gt_c, iterations=1, border_value=0)
            rows.append({"condition": cond, "name": n,
                         "dice": float(m["dice"]), "iou": float(m["iou"]),
                         "bdice1": float(l0.s57.bdice(base, gt_c, 1)),
                         "gt_in_interval": float((b_gt & ~caut & amp).sum() /
                                                 max(1, b_gt.sum()))})
        sub = [r for r in rows if r["condition"] == cond]
        d = [r["dice"] for r in sub]
        print("   %-16s Dice %6.2f (min %6.2f) | bdice@1 %5.2f | <80 %2d/%d | gt-in-interval %.3f"
              % (cond, np.mean(d), np.min(d), np.mean([r["bdice1"] for r in sub]),
                 sum(x < 80 for x in d), len(d),
                 np.mean([r["gt_in_interval"] for r in sub])), flush=True)

    base_row = [r for r in rows if r["condition"] == "in_domain"]
    print("\n[l2] tag=%s | student params %.2f M | in_domain Dice %.2f bdice@1 %.2f"
          % (args.tag, n_param / 1e6, np.mean([r["dice"] for r in base_row]),
             np.mean([r["bdice1"] for r in base_row])))
    worst = min(conds, key=lambda c: np.mean([r["dice"] for r in rows if r["condition"] == c])) \
        if conds else "in_domain"
    print("[l2] worst condition: %s  Dice %.2f"
          % (worst, np.mean([r["dice"] for r in rows if r["condition"] == worst])))

    tag_out = args.tag_out or args.tag
    out = os.path.join(ROOT, "results", "l2_eval_%s.csv" % tag_out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {c: {"dice": float(np.mean([r["dice"] for r in rows if r["condition"] == c])),
                   "dice_min": float(np.min([r["dice"] for r in rows if r["condition"] == c])),
                   "bdice1": float(np.mean([r["bdice1"] for r in rows if r["condition"] == c])),
                   "n_lt80": int(sum(r["dice"] < 80 for r in rows if r["condition"] == c))}
               for c in conds}
    with open(os.path.join(ROOT, "results", "l2_eval_%s.json" % tag_out), "w") as fh:
        json.dump({"tag": args.tag, "params": n_param, "summary": summary}, fh, indent=2)
    print("[l2] wrote %s" % out)


if __name__ == "__main__":
    sys.exit(main())

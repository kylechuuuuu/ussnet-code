#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/tools/61_squeeze_diagnostics.py

Does the squeeze actually pinch?  This tool answers that on the validation split
with the trained weights, in the network's own coordinate frame.

For every validation case it runs the three-head network (mirroring TTA, same
sliding window as every other prediction in this project) and measures the
claims the squeeze theorem makes:

  CERTIFICATES (the premise  Omega_in <= Omega_gt <= Omega_out)
    outer_miss   |Omega_gt \\ Omega_out| / |Omega_gt|   -> 0 means the outer bound
                 really contains the gland
    inner_leak   |Omega_in \\ Omega_gt| / |Omega_in|     -> 0 means the inner bound
                 really stays inside it
    bd_contain   fraction of the TRUE boundary that lies inside the predicted
                 interval (Omega_out \\ Omega_in): the truth is pinched
    order_mag    mean(relu(p_in-p_m)) + mean(relu(p_m-p_out))  (violation size)
    order_big    fraction of pixels where a violation exceeds 0.1

  THE PINCH
    width        mean |Omega_out \\ Omega_in| / perimeter  = interval width in px
    width@p50/p95

  ERROR BRACKETING (the useful part)
    for the fused decision rules (main / median / inner / outer):
      err_total          symmetric difference with GT (px)
      err_in_band        part of that error that lies inside the predicted
                         interval --> the interval KNOWS about it
      err_out_band       part that is outside the interval --> the model claimed
                         certainty there and was wrong
    The squeeze claim is err_out_band -> small: the model should never be
    confidently wrong outside its own interval.

  ANGULAR BREAKDOWN (12 sectors from the GT centroid, same convention as the
  fingerprint and as trainer/squeeze_losses.sector_scale_map): band width and
  boundary containment per sector, i.e. whether the fingerprint-driven angular
  band really is wider where the ultrasound evidence is weaker.

usage:
  python tools/61_squeeze_diagnostics.py \
      --model-folder <results>/Dataset001_UltrasoundProstate/nnUNetTrainerProstateSqueeze__nnUNetPlans__2d \
      --checkpoint checkpoint_best.pth --tag sqA_angular_best [--no-tta] [--examples 4]
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(STEP5, "trainer"))

from squeeze_net import SqueezeHeadUNet                                # noqa: E402


class _SixHeadLabelManager(object):
    """Duck-type shim: the sliding window preallocates
       `label_manager.num_segmentation_heads` output maps, and here we want all
       six raw head channels (3 heads x 2 classes) instead of the fused 2."""

    num_segmentation_heads = 6
    has_regions = False
    ignore_label = None
    foreground_labels = [1]


def find_wrapper(net):
    for _ in range(3):
        if isinstance(net, SqueezeHeadUNet):
            return net
        nxt = getattr(net, "_orig_mod", None) or getattr(net, "module", None)
        if nxt is None:
            return None
        net = nxt
    return None


# ------------------------------------------------------------------ morphology
def _ero(x, k=1):
    return -F.max_pool2d(-x, 2 * k + 1, 1, k)


def _dil(x, k=1):
    return F.max_pool2d(x, 2 * k + 1, 1, k)


def boundary(mask):
    return (_dil(mask, 1) - _ero(mask, 1)).clamp(0, 1)


def perimeter_est(mask):
    return float(boundary(mask).sum())


def sector_index_map(h, w, cy, cx, n_sector=12):
    """Sector id per pixel, same convention as the fingerprint:
    +90 deg = image up = anterior, sector k covers [-180 + 30k, -150 + 30k)."""
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    ang = torch.atan2(-(yy - cy), xx - cx)
    return torch.floor((ang + math.pi) / (2 * math.pi) * n_sector).long().clamp_(0, n_sector - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-folder", required=True)
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--examples", type=int, default=0,
                    help="also store an overlay PNG for this many cases (needs matplotlib)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    os.environ.setdefault("nnUNet_compile", "False")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.utilities.file_path_utilities import get_output_folder  # noqa: F401
    from batchgenerators.utilities.file_and_folder_operations import load_json

    out_dir = args.out_dir or os.path.join(STEP5, "results")
    os.makedirs(out_dir, exist_ok=True)

    dataset_json = load_json(os.path.join(args.model_folder, "dataset.json"))
    plans = load_json(os.path.join(args.model_folder, "plans.json"))
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(
        dataset_json.get("_configuration", "2d") if "_configuration" in dataset_json else "2d")
    preprocessed = os.path.join(os.environ["nnUNet_preprocessed"], plans_manager.dataset_name,
                                configuration_manager.data_identifier)

    # ---- the validation keys of fold 0 (identical split to every model here)
    splits = load_json(os.path.join(os.path.dirname(preprocessed), "splits_final.json"))
    val_keys = splits[int(args.fold)]["val"]

    # ---- model
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                use_mirroring=not args.no_tta,
                                perform_everything_on_device=True,
                                device=torch.device(args.device), verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        args.model_folder, use_folds=(int(args.fold),), checkpoint_name=args.checkpoint)
    wrapper = find_wrapper(predictor.network)
    if wrapper is None:
        raise RuntimeError("checkpoint does not build a SqueezeHeadUNet")
    wrapper.set_fusion("train")                       # raw heads
    predictor.label_manager = _SixHeadLabelManager()  # preallocate 6 maps
    print("[61] tag=%s | cases=%d | TTA(mirroring)=%s | checkpoint=%s"
          % (args.tag, len(val_keys), not args.no_tta, args.checkpoint))

    ds = infer_dataset_class(preprocessed)(preprocessed, val_keys)

    rows = []
    overlays = []
    for i, key in enumerate(sorted(val_keys)):
        data, seg, _, _ = ds.load_case(key)
        # blosc2 arrays are lazy: materialise explicitly ([:]) before converting
        d = torch.from_numpy(np.ascontiguousarray(np.asarray(data[:]))).float()
        gt = torch.from_numpy(np.ascontiguousarray(np.asarray(seg[:]))).float()
        if gt.ndim == 4:
            gt = gt[0]                       # (1, H, W)
        elif gt.ndim == 2:
            gt = gt[None]
        gt = (gt > 0.5).float()
        h, w = gt.shape[-2:]

        logits = predictor.predict_sliding_window_return_logits(d).float().cpu()
        # the sliding window returns (num_heads, *data.shape[1:]); for this 2-D
        # dataset data.shape[1:] is (1, H, W), so flatten the singleton away
        logits = logits.reshape(logits.shape[0], -1, *logits.shape[-2:]).squeeze(1)   # (6, H, W)
        # heads: [0:2] main, [2:4] inner, [4:6] outer
        p_m = torch.softmax(logits[0:2], 0)[1:2]
        p_i = torch.softmax(logits[2:4], 0)[1:2]
        p_o = torch.softmax(logits[4:6], 0)[1:2]
        if p_m.shape[-2:] != gt.shape[-2:]:
            p_m = F.interpolate(p_m[None], size=gt.shape[-2:], mode='bilinear')[0]
            p_i = F.interpolate(p_i[None], size=gt.shape[-2:], mode='bilinear')[0]
            p_o = F.interpolate(p_o[None], size=gt.shape[-2:], mode='bilinear')[0]

        m_in = (p_i > 0.5).float()
        m_out = (p_o > 0.5).float()
        m_main = (p_m > 0.5).float()
        m_med = (torch.sort(torch.cat([p_i, p_m, p_o], 0), 0).values[1:2] > 0.5).float()

        band = (m_out * (1 - m_in)).clamp(0, 1)
        bd_gt = boundary(gt)
        perimeter = perimeter_est(gt)
        gt_sum = float(gt.sum())
        in_sum = float(m_in.sum())

        gap = F.relu(p_o - p_i)
        w_mid = 4 * p_m * (1 - p_m)
        order_mag = float((F.relu(p_i - p_m) + F.relu(p_m - p_o)).mean())
        order_big = float((((p_i - p_m) > 0.1) | ((p_m - p_o) > 0.1)).float().mean())

        row = {
            "case": key,
            "gt_area": gt_sum,
            "perimeter": perimeter,
            "outer_miss": float(((gt - m_out).clamp_min(0).sum()) / max(1.0, gt_sum)),
            "inner_leak": float(((m_in - gt).clamp_min(0).sum()) / max(1.0, in_sum)),
            "bd_contain": float((bd_gt * band).sum() / max(1.0, float(bd_gt.sum()))),
            "order_mag": order_mag,
            "order_big": order_big,
            "gap_mean": float(gap.mean()),
            "gap_mid": float((w_mid * gap).sum() / max(1e-6, float(w_mid.sum()))),
            "band_area_px": float(band.sum()),
            "band_width_px": float(band.sum()) / max(1.0, perimeter),
        }
        for name, pred in [("main", m_main), ("median", m_med), ("inner", m_in), ("outer", m_out)]:
            err = (pred - gt).abs()
            row["%s_dice" % name] = float(2 * (pred * gt).sum()
                                          / max(1e-6, float(pred.sum() + gt.sum())))
            row["%s_err_px" % name] = float(err.sum())
            row["%s_err_in_band" % name] = float((err * band).sum())
            row["%s_err_out_band" % name] = float((err * (1 - band)).sum())
        # the middle head's "certain" region should never contain gross errors
        rows.append(row)

        # ---------------- angular breakdown (12 sectors around the GT centroid)
        ys, xs = torch.nonzero(gt[0] > 0.5, as_tuple=True)
        cy, cx = (float(ys.float().mean()), float(xs.float().mean())) if ys.numel() else (h / 2, w / 2)
        sec = sector_index_map(h, w, cy, cx)
        gt_frac = gt[0] > 0.5
        band_2d = band[0]
        bd_gt_2d = bd_gt[0]
        for s in range(12):
            sel = (sec == s)
            if float((sel & gt_frac).sum()) < 50:
                continue
            bd_s = bd_gt_2d * sel.float()
            rows.append({
                "case": key, "sector": s,
                "sector_band_width_px": float((band_2d * (sel & gt_frac).float()).sum()
                                              / max(1.0, float(bd_s.sum()))),
                "sector_bd_contain": float((bd_s * band_2d).sum() / max(1e-6, float(bd_s.sum()))),
            })

        if args.examples and i < args.examples:
            overlays.append((key, gt[0, 0].cpu(), m_med[0, 0].cpu(), m_in[0, 0].cpu(), m_out[0, 0].cpu()))
        print("[61] %-8s dice main %.4f median %.4f | width %.2f px | contain %.3f | "
              "outer_miss %.4f | inner_leak %.4f | order %.4f"
              % (key, row["main_dice"], row["median_dice"], row["band_width_px"],
                 row["bd_contain"], row["outer_miss"], row["inner_leak"], order_mag))

    # ------------------------------------------------------------------ summary
    per_case = [r for r in rows if "main_dice" in r]
    sec_rows = [r for r in rows if "sector" in r]
    keys = ["main_dice", "median_dice", "inner_dice", "outer_dice",
            "main_err_px", "median_err_px", "inner_err_px", "outer_err_px",
            "main_err_in_band", "main_err_out_band", "median_err_in_band", "median_err_out_band",
            "outer_miss", "inner_leak", "bd_contain", "order_mag", "order_big",
            "gap_mean", "gap_mid", "band_width_px", "band_area_px", "perimeter"]
    summary = {"tag": args.tag, "checkpoint": args.checkpoint, "n_cases": len(per_case),
               "tta": not args.no_tta}
    for k in keys:
        vals = np.array([r[k] for r in per_case], dtype=float)
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "median": float(np.median(vals))}
    # error bracketing ratios (how much of the error the interval admits to)
    for name in ["main", "median"]:
        num = np.array([r["%s_err_out_band" % name] for r in per_case], float)
        den = np.array([r["%s_err_px" % name] for r in per_case], float)
        summary["%s_err_out_band_frac" % name] = {
            "mean": float(np.mean(num / np.maximum(den, 1e-9))),
            "pooled": float(num.sum() / max(1.0, den.sum()))}
    # sector table (per-sector means over cases)
    sector_table = []
    for s in range(12):
        vals = [r for r in sec_rows if r["sector"] == s]
        if not vals:
            continue
        sector_table.append({
            "sector": s,
            "deg_center": -165 + 30 * s,
            "band_width_px": float(np.mean([v["sector_band_width_px"] for v in vals])),
            "bd_contain": float(np.mean([v["sector_bd_contain"] for v in vals])),
        })
    summary["sectors"] = sector_table

    csv_path = os.path.join(out_dir, "%s_per_case.csv" % args.tag)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=sorted({k for r in per_case for k in r}))
        wr.writeheader()
        wr.writerows(per_case)
    json_path = os.path.join(out_dir, "%s_summary.json" % args.tag)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[61] === %s (n=%d, %s) ===" % (args.tag, len(per_case),
                                            "mirroring TTA" if not args.no_tta else "no TTA"))
    print("[61] certificates : GT outside outer %.4f | inner outside GT %.4f | "
          "true boundary inside interval %.3f"
          % (summary["outer_miss"]["mean"], summary["inner_leak"]["mean"],
             summary["bd_contain"]["mean"]))
    print("[61] pinch        : interval width %.2f px (p95 %.2f) | gap at decision layer %.4f | "
          "order violation %.4f (big %.4f)"
          % (summary["band_width_px"]["mean"], np.percentile(
              [r["band_width_px"] for r in per_case], 95),
             summary["gap_mid"]["mean"], summary["order_mag"]["mean"], summary["order_big"]["mean"]))
    print("[61] decision     : dice main %.4f | median %.4f | inner %.4f | outer %.4f"
          % (summary["main_dice"]["mean"], summary["median_dice"]["mean"],
             summary["inner_dice"]["mean"], summary["outer_dice"]["mean"]))
    print("[61] bracketing  : median error outside the interval = %.3f (pooled %.3f) of its total error"
          % (summary["median_err_out_band_frac"]["mean"],
             summary["median_err_out_band_frac"]["pooled"]))
    print("[61] written      : %s | %s" % (csv_path, json_path))

    if overlays:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(overlays)
            fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
            axes = np.atleast_1d(axes)
            for ax, (key, gt_, med, mi, mo) in zip(axes, overlays):
                band = (mo * (1 - mi)).clamp(0, 1)
                rgb = np.zeros((*gt_.shape, 3))
                rgb[..., 0] = gt_.numpy() * 0.85
                rgb[..., 1] = med.numpy() * 0.55
                rgb[..., 2] = band.numpy() * 0.95
                ax.imshow(rgb)
                ax.set_title("%s\nred=GT green=fused blue=squeeze interval" % key, fontsize=8)
                ax.axis("off")
            fig.tight_layout()
            fp = os.path.join(STEP5, "figures", "%s_overlays.png" % args.tag)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            fig.savefig(fp, dpi=140)
            print("[61] overlays     : %s" % fp)
        except Exception as e:
            print("[61] overlay figures skipped: %s" % e)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/tools/63_arbitrate.py

BOUNDARY DETERMINATION (然后确定分割的边界).

The sandwich model delivers three versions of the same segmentation:

    cautious   (谨慎版本)  the region the model is sure about       (inside)
    base       (基础版本)  the ordinary segmentation
    amplified  (放大版本)  the region the model cannot exclude      (outside)

with the nesting enforced by construction at prediction time:

    Min = base ∩ cautious        M0 = base        Mout = base ∪ amplified
    (Min ⊆ M0 ⊆ Mout always)

The final boundary is DETERMINED not by a vote of the heads but by
ARBITRATION inside the interval: every contour point of the base version may
slide along its normal toward the strongest image-edge response, but it may
never leave [Min, Mout].  Wide interval -> the image decides; narrow interval
-> the annotation convention is kept.  This is the only route that ever
improved boundary Dice on this dataset (step-4 section 5.5: +1.3 boundary
Dice@1px): here the squeeze interval ARBITRATES how far the classical snap is
allowed to move.

Variants computed per case (shared cc1(base) so topology cleanup is never
credited to the arbitration):

    raw        the base mask itself
    free       classical snapping, window +-maxstep               (step-4 §5.5)
    band_lrn   the same, clamped to the LEARNED interval [Min, Mout]
    band_thr   the same, clamped to a THRESHOLD interval of the base version
               (Min = p_base > thr_hi, Mout = p_base > thr_lo) -- the same
               cautious/amplified idea derived from ONE model field
    veto_lrn   the FREE move executed only where it stays inside the LEARNED
               interval: image evidence proposes, the squeeze disposes
    veto_thr   the same veto with the THRESHOLD interval

Parameters (frac=0.7, maxstep=4, sigma=1.5, ridge smoothing 1.0) follow step
4's train-split-tuned probe f70_s04 and are NOT re-tuned on the validation
split.

usage:
  # probability mode (the SQ2 model; trained + threshold bands):
  python tools/63_arbitrate.py --npz-base-dir predictions/sq2_checkpoint_best_base_nifti \
      --npz-cautious-dir predictions/sq2_checkpoint_best_cautious_nifti \
      --npz-amplified-dir predictions/sq2_checkpoint_best_amplified_nifti \
      --split val --out-dir predictions/sq2_checkpoint_best_arb_val --tag sq2_val
  # mask mode (existing predictions, e.g. the step-5 arms):
  python tools/63_arbitrate.py --base-png-dir <png> --inner-png-dir <png> \
      --outer-png-dir <png> --split train --out-dir <dir> --tag sqA_train
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
STEP6 = os.path.dirname(HERE)
USS = os.path.dirname(STEP6)
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))
from eval_lib import read_split                                    # noqa: E402

STEP4_TOOLS = os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")
IMG_ROOT = IMG_DIR          # evidence image dir; --image-root overrides for condition-wise E1
GT_DIR = os.path.join(USS, "UltrasoundProstate", "masks")

TS = np.arange(-16.0, 16.0001, 0.5)      # search grid along the normal (px)
OFFS = np.arange(0.0, 8.0001, 0.5)       # interval probe grid (px)
NPT = 400                                # contour resampling (same as step 4)


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s57 = _load_mod("s57", os.path.join(STEP4_TOOLS, "57_shape_prior_ceiling.py"))
s58 = _load_mod("s58", os.path.join(STEP4_TOOLS, "58_normal_refine_oracle.py"))

# nnU-Net runs on stable ASCII ids (USP001 ... USP102); the original Chinese
# file names are used by the PNG/mask side of the pipeline.  The mapping was
# written by step1's converter and is reused here to find the prediction npz.
MAPPING_CSV = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")


def _load_name2cid():
    m = {}
    if os.path.exists(MAPPING_CSV):
        with open(MAPPING_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                stem = str(r.get("original_filename", ""))
                if stem.lower().endswith(".png"):
                    stem = stem[:-4]
                if stem:
                    m[stem] = r["case_id"]
    return m


NAME2CID = _load_name2cid()


# --------------------------------------------------------------------- io
def stem_of(name):
    return name[:-4] if name.lower().endswith(".png") else name


def find_file(d, stem, ext):
    p = os.path.join(d, stem + ext)
    if os.path.isfile(p):
        return p
    hits = sorted(glob.glob(os.path.join(d, stem + "*")))
    return hits[0] if hits else None


def load_image(stem):
    p = find_file(IMG_ROOT, stem, ".png")
    return np.array(Image.open(p).convert("L"), dtype=np.float32)


def load_prob(d, stem):
    p = find_file(d, stem, ".npz")
    if p is None and stem in NAME2CID:
        p = find_file(d, NAME2CID[stem], ".npz")
    if p is None:
        return None
    z = np.load(p)
    key = None
    for k in ("probabilities", "softmax", "prob"):
        if k in z.files:
            key = k
            break
    if key is None:
        cands = [k for k in z.files if z[k].size > 16]
        key = max(cands, key=lambda k: z[k].size)
    return np.asarray(z[key], dtype=np.float32)


def align_prob(arr, shape):
    """Return the (H, W) positive-class probability aligned with the mask shape."""
    a = np.squeeze(arr)
    if a.ndim == 3:
        if a.shape[0] == 2:
            a = a[1]
        elif a.shape[-1] == 2:
            a = a[..., 1]
        else:
            raise ValueError("cannot find the positive class in %s" % (arr.shape,))
    if a.ndim != 2:
        raise ValueError("unexpected probability array shape %s" % (arr.shape,))
    if a.shape == shape:
        return a
    if a.shape == shape[::-1]:
        return a.T
    raise ValueError("probability shape %s incompatible with %s" % (a.shape, shape))


def load_mask(d, stem):
    p = find_file(d, stem, ".png")
    if p is None:
        return None
    return np.array(Image.open(p).convert("L")) > 127


def write_png(path, mask):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


# ----------------------------------------------------------- arbitration core
def sample_mask(mask, q):
    v = ndimage.map_coordinates(mask.astype(np.float32),
                                [q[..., 0].ravel(), q[..., 1].ravel()],
                                order=0, mode="nearest").reshape(q.shape[:-1])
    return v > 0.5


def limits(pts, nrm, Min, Mout):
    """Per-point legal room along the normal: (inward to Min, outward to Mout)."""
    qo = pts[:, None, :] + OFFS[None, :, None] * nrm[:, None, :]
    vo = sample_mask(Mout, qo)
    k_out = np.argmin(vo, axis=1)                    # first sample outside Mout
    t_out = np.where(vo.all(axis=1), OFFS[-1], np.maximum(OFFS[k_out] - 0.5, 0.0))
    qi = pts[:, None, :] - OFFS[None, :, None] * nrm[:, None, :]
    vi = sample_mask(Min, qi)
    # A boundary point may slide inward until it would ENTER the cautious
    # region: an allowed boundary position must stay outside Min's interior.
    # If the inward ray never enters Min the constraint is locally vacuous
    # (the caller still caps the window at +-maxstep).  If the point already
    # sits inside Min, any inward move is forbidden (t_in = 0).
    first = np.argmax(vi, axis=1)                    # first offset inside Min
    t_in = np.where(vi.any(axis=1), np.maximum(OFFS[first] - 0.5, 0.0), OFFS[-1])
    return t_in, t_out


def ridge_profile(img, pts, nrm, ts, smooth_t=1.0):
    q = pts[:, None, :] + ts[None, :, None] * nrm[:, None, :]
    v = ndimage.map_coordinates(img, [q[..., 0].ravel(), q[..., 1].ravel()],
                                order=1, mode="nearest").reshape(len(pts), len(ts))
    g = np.abs(np.gradient(v, ts, axis=1))
    if smooth_t > 0:
        g = ndimage.gaussian_filter1d(g, smooth_t, axis=1, mode="nearest")
    return g


def pick(G, ts, frac, lo, hi):
    """Nearest-to-zero ridge above frac*max inside the per-point window; else stay."""
    thr = frac * G.max(axis=1, keepdims=True)
    ok = (G >= thr) & (ts[None, :] >= lo[:, None]) & (ts[None, :] <= hi[:, None])
    cost = np.where(ok, np.abs(ts)[None, :], 1e18)
    j = np.argmin(cost, axis=1)
    t = ts[j].astype(np.float64).copy()
    t[~ok.any(axis=1)] = 0.0
    return t


def apply_t(M0, pts, nrm, t):
    new = pts + t[:, None] * nrm
    m = s57.rasterize(new, M0.shape)
    a0, a1 = float(M0.sum()), float(m.sum())
    if a1 <= 0 or not (0.5 * a0 <= a1 <= 2.0 * a0):
        return M0.copy(), np.zeros_like(t)
    return s57.largest_cc(m), t


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-base-dir", default=None)
    ap.add_argument("--npz-cautious-dir", default=None)
    ap.add_argument("--npz-amplified-dir", default=None)
    ap.add_argument("--base-png-dir", default=None)
    ap.add_argument("--inner-png-dir", default=None)
    ap.add_argument("--outer-png-dir", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--names-file", default=None,
                    help="override the case list: one filename or stem per line (fold-aware arbitration)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="arb")
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--maxstep", type=float, default=4.0)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--thr-hi", type=float, default=0.85)
    ap.add_argument("--thr-lo", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image-root", default=None,
                    help="evidence image dir override (condition-wise E1; default: in-domain images)")
    ap.add_argument("--results-dir", default=os.path.join(STEP6, "results"))
    args = ap.parse_args()

    global IMG_ROOT
    if args.image_root:
        IMG_ROOT = args.image_root

    if args.names_file:
        with open(args.names_file, encoding="utf-8") as fh:
            names = [l.strip() for l in fh if l.strip()]
    else:
        train_names, val_names = read_split(USS)
        names = train_names if args.split == "train" else val_names
    if args.limit:
        names = names[:args.limit]

    rows = []
    for name in names:
        stem = stem_of(name)
        gt = s57.load_gt(stem)
        if gt is None:
            print("[63] WARN no GT for %s" % stem)
            continue
        try:
            img = load_image(stem)
        except Exception as e:
            print("[63] WARN no image for %s (%s)" % (stem, e))
            continue
        if img.shape != gt.shape:
            print("[63] WARN shape mismatch for %s" % stem)
            continue

        # ---- base version + the two intervals --------------------------------
        p_b = None
        if args.npz_base_dir:
            arr = load_prob(args.npz_base_dir, stem)
            if arr is not None:
                try:
                    p_b = align_prob(arr, gt.shape)
                except ValueError as e:
                    print("[63] WARN %s: %s" % (stem, e))
        if p_b is not None:
            M0 = s57.largest_cc(p_b > 0.5)
        else:
            m = load_mask(args.base_png_dir, stem) if args.base_png_dir else None
            if m is None:
                print("[63] WARN no base prediction for %s" % stem)
                continue
            M0 = s57.largest_cc(m)
        if not M0.any():
            print("[63] WARN empty base prediction for %s" % stem)
            continue

        Min = Mout = None
        if args.npz_cautious_dir and args.npz_amplified_dir:
            pc = load_prob(args.npz_cautious_dir, stem)
            pa = load_prob(args.npz_amplified_dir, stem)
            if pc is not None and pa is not None:
                try:
                    Min = M0 & (align_prob(pc, gt.shape) > 0.5)
                    Mout = M0 | (align_prob(pa, gt.shape) > 0.5)
                except ValueError as e:
                    print("[63] WARN %s: %s" % (stem, e))
        if Min is None and args.inner_png_dir and args.outer_png_dir:
            mi = load_mask(args.inner_png_dir, stem)
            mo = load_mask(args.outer_png_dir, stem)
            if mi is not None and mo is not None:
                Min = M0 & mi
                Mout = M0 | mo

        Min_t = Mout_t = None
        if p_b is not None:
            Min_t = M0 & (p_b > args.thr_hi)
            Mout_t = M0 | (p_b > args.thr_lo)

        # ---- contour + evidence profile --------------------------------------
        c = s57.contour_of(M0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, M0)
        imgf = ndimage.gaussian_filter(img, args.sigma)
        G = ridge_profile(imgf, pts, nrm, TS, 1.0)

        row = {"name": stem}
        win = {}
        gb = s57.surface(gt)
        if Min is not None:
            t_in, t_out = limits(pts, nrm, Min, Mout)
            win["band_lrn"] = (-np.clip(t_in, 0, args.maxstep),
                               np.clip(t_out, 0, args.maxstep))
            band = Mout & ~Min
            row["contain_lrn"] = float((gb & band).sum() / max(1, gb.sum()))
            row["band_lrn_px"] = float(np.mean(t_in + t_out))
        if Min_t is not None:
            t_in_t, t_out_t = limits(pts, nrm, Min_t, Mout_t)
            win["band_thr"] = (-np.clip(t_in_t, 0, args.maxstep),
                               np.clip(t_out_t, 0, args.maxstep))
            band = Mout_t & ~Min_t
            row["contain_thr"] = float((gb & band).sum() / max(1, gb.sum()))
            row["band_thr_px"] = float(np.mean(t_in_t + t_out_t))

        res = {"raw": (M0, None)}
        t_free = pick(G, TS, args.frac,
                      -np.full(len(pts), args.maxstep), np.full(len(pts), args.maxstep))
        res["free"] = apply_t(M0, pts, nrm, t_free)
        for v, (lo, hi) in win.items():
            t = pick(G, TS, args.frac, lo, hi)
            res[v] = apply_t(M0, pts, nrm, t)
        # veto semantics: image evidence proposes, the squeeze disposes -- the
        # free move is executed only where it stays inside the band.
        for v, (lo, hi) in win.items():
            ok = (t_free <= hi + 1e-9) & (t_free >= lo - 1e-9)
            res["veto_" + v.split("_", 1)[1]] = apply_t(M0, pts, nrm,
                                                        np.where(ok, t_free, 0.0))

        for v, (m, t) in res.items():
            row["bdice_" + v] = s57.bdice(m, gt, 1)
            row["dice_" + v] = s57.dice(m, gt)
            if t is not None:
                row["move_" + v] = float(np.mean(np.abs(t)))
            write_png(os.path.join(args.out_dir, v, stem + ".png"), m)
        rows.append(row)

    # ---- summary -------------------------------------------------------------
    print("\n[63] ==== %s | split %s | %d images | frac %.2f maxstep %.1f sigma %.1f ===="
          % (args.tag, args.split, len(rows), args.frac, args.maxstep, args.sigma))
    summary = {"tag": args.tag, "split": args.split, "n": len(rows),
               "params": {"frac": args.frac, "maxstep": args.maxstep,
                          "sigma": args.sigma, "thr_hi": args.thr_hi,
                          "thr_lo": args.thr_lo},
               "mean": {}, "n_variant": {}}
    for v in ("raw", "free", "band_lrn", "band_thr", "veto_lrn", "veto_thr"):
        vals = [r["bdice_" + v] for r in rows if ("bdice_" + v) in r]
        dvals = [r["dice_" + v] for r in rows if ("dice_" + v) in r]
        if not vals:
            continue
        summary["mean"][v] = float(np.mean(vals))
        summary["n_variant"][v] = len(vals)
        extra = ""
        if v != "raw":
            mv = [r.get("move_" + v, np.nan) for r in rows]
            extra = "| mean|t| %.2f px" % float(np.nanmean(mv))
        print("[63]   %-9s bdice@1px %6.2f  dice %6.2f  %s"
              % (v, float(np.mean(vals)), float(np.mean(dvals)), extra))
    for k in ("contain_lrn", "contain_thr", "band_lrn_px", "band_thr_px"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            summary["mean"][k] = float(np.mean(vals))
            print("[63]   %-10s %.3f" % (k, float(np.mean(vals))))

    os.makedirs(args.results_dir, exist_ok=True)
    jp = os.path.join(args.results_dir, "arbitrate_%s.json" % args.tag)
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(args.results_dir, "arbitrate_%s.csv" % args.tag),
                  "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print("[63] written %s" % jp)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

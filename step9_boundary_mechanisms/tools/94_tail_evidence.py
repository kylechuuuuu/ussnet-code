#!/usr/bin/env python3
"""94_tail_evidence.py -- is the GT boundary at the tail points VISIBLE?

The choice of mechanism 2 hinges on one question: at the points where the base
model is >10 px wrong (nearly all of them anterior), is the true boundary marked
by an image edge that the model could have found?

For every resampled contour point we take the intensity profile along the normal
(+/-40 px, image smoothed sigma=1.5, |dI/dt| smoothed in t by 1.0 -- exactly the
step6 'ridge' evidence) and ask:
   * G(t_gt)/Gmax  : how strong is the image edge AT the true boundary
   * G(0)/Gmax     : ... and at the model's own boundary
   * is the nearest 'crest' (G >= 0.7*max) the true one or the model's?
   * |I_in - I_out| across the true boundary: is it a real intensity step?
   * how many competing crests are in the window (single edge vs double edge)

If G(t_gt) is weak -> the anterior boundary is an annotation convention the image
does not show, and only a shape/context prior can recover it.
If G(t_gt) is strong AND G(0) is strong -> the image has two edges and the model
picks the wrong one: that is a SELECTION problem, solvable by a global /
shape-constrained arbitration over a wide range.

CPU only.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
STEP9 = os.path.dirname(HERE)
USS = os.path.dirname(STEP9)
STEP6 = os.path.join(USS, "step6_sandwich_squeeze")
STEP6_TOOLS = os.path.join(STEP6, "tools")
sys.path.insert(0, STEP6_TOOLS)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = _load("arb", os.path.join(STEP6_TOOLS, "63_arbitrate.py"))
s57, s58 = arb.s57, arb.s58
NPZ = os.path.join(STEP6, "predictions", "sq2resencw25_checkpoint_best_base_nifti")
NPT = 400
TS = np.arange(-40.0, 40.0001, 0.5)
TG = np.arange(-90.0, 90.0001, 0.5)
SIGMA, FRAC = 1.5, 0.7
TAIL = 10.0


def gt_pos(sdf, pts, nrm, tg=TG):
    q = pts[:, None, :] + tg[None, :, None] * nrm[:, None, :]
    v = ndimage.map_coordinates(sdf, [q[..., 0].ravel(), q[..., 1].ravel()],
                                order=1, mode="nearest").reshape(len(pts), len(tg))
    out = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        s = np.sign(v[i])
        idx = np.flatnonzero(s[:-1] * s[1:] < 0)
        if idx.size == 0:
            continue
        j = idx[np.argmin(np.abs(tg[idx]))]
        v0, v1 = v[i, j], v[i, j + 1]
        out[i] = tg[j] + (tg[j + 1] - tg[j]) * (0.0 - v0) / (v1 - v0) if v1 != v0 else tg[j]
    return out


def sample(img, pts, nrm, ts):
    q = pts[:, None, :] + ts[None, :, None] * nrm[:, None, :]
    return ndimage.map_coordinates(img, [q[..., 0].ravel(), q[..., 1].ravel()],
                                   order=1, mode="nearest").reshape(len(pts), len(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=NPZ)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(STEP9, "results"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train, val = arb.read_split(USS)
    names = train if args.split == "train" else val
    if args.limit:
        names = names[:args.limit]

    rows = []
    for name in names:
        stem = arb.stem_of(name)
        gt = s57.load_gt(stem)
        arr = arb.load_prob(args.npz_dir, stem)
        if gt is None or arr is None:
            continue
        img = arb.load_image(stem)
        p = arb.align_prob(arr, gt.shape)
        M0 = s57.largest_cc(p > 0.5)
        c = s57.contour_of(M0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, M0)
        t_gt = gt_pos(s58.sdf_of(gt), pts, nrm)
        ok = np.isfinite(t_gt)

        imgf = ndimage.gaussian_filter(img, SIGMA)
        G = arb.ridge_profile(imgf, pts, nrm, TS, 1.0)
        prof = sample(imgf, pts, nrm, TS)

        gmax = G.max(axis=1)
        g0 = G[:, int(np.argmin(np.abs(TS)))]
        # value of G at t_gt (linear interp), and the local max within +-8 px
        ti = np.clip((t_gt + 40.0) / 0.5, 0, len(TS) - 1)
        ti = np.where(np.isfinite(ti), ti, 0.0)
        i0 = np.floor(np.clip(ti, 0, len(TS) - 1)).astype(int)
        i0 = np.where(np.isfinite(t_gt), i0, 0)
        i1 = np.clip(i0 + 1, 0, len(TS) - 1)
        w = np.where(np.isfinite(t_gt), ti - i0, 0.0)
        g_at = G[np.arange(len(pts)), i0] * (1 - w) + G[np.arange(len(pts)), i1] * w
        loc = np.abs(TS[None, :] - np.where(np.isfinite(t_gt), t_gt, 0.0)[:, None]) <= 8.0
        gmax_loc = np.where(loc, G, 0.0).max(axis=1)
        # nearest crest at >= FRAC * global max
        thr = FRAC * gmax[:, None]
        okc = G >= thr
        cost = np.where(okc, np.abs(TS)[None, :], 1e18)
        jc = np.argmin(cost, axis=1)
        has_crest = okc.any(axis=1)
        crest_t = TS[jc]
        # number of local maxima above 0.5 * max
        mx = (G[:, 1:-1] > G[:, :-2]) & (G[:, 1:-1] >= G[:, 2:]) & (G[:, 1:-1] > 0.5 * gmax[:, None])
        n_peak = mx.sum(axis=1)
        # --- is the TRUE boundary itself a gradient peak? +-1.5px vs +-20px ---
        tt = np.where(np.isfinite(t_gt), t_gt, 1e6)
        ggt = np.where(np.abs(TS[None, :] - tt[:, None]) <= 1.5, G, 0.0).max(axis=1)
        g20 = np.where(np.abs(TS[None, :] - tt[:, None]) <= 20.0, G, 0.0).max(axis=1)
        ratio20 = ggt / np.maximum(g20, 1e-9)
        # intensity step across the true boundary
        ip = sample(imgf, pts, nrm, np.array([2.0]))[:, 0]
        im = sample(imgf, pts, nrm, np.array([-2.0]))[:, 0]
        tp = sample(imgf, pts, nrm, np.where(np.isfinite(t_gt), t_gt, 0.0) + 2.0)[:, 0]
        tm = sample(imgf, pts, nrm, np.where(np.isfinite(t_gt), t_gt, 0.0) - 2.0)[:, 0]

        rows.append(np.stack([
            np.where(ok, t_gt, np.nan), g_at, g0, gmax, gmax_loc,
            has_crest.astype(float), crest_t, n_peak.astype(float),
            np.abs(ip - im), np.abs(tp - tm), t_gt - crest_t,
            np.where(ok, 1.0, 0.0), ggt, ratio20], axis=1))

    P = np.concatenate(rows, axis=0)
    (t, g_at, g0, gmax, gmax_loc, has_crest, crest_t, n_peak, step0, stepgt,
     dcrest, okv, ggt, ratio20) = [P[:, i] for i in range(14)]
    fin = okv > 0.5
    at = np.abs(t)
    tail = fin & (at > TAIL)
    good = fin & (at <= 2.0)

    def stats(mask, nm):
        if not mask.any():
            return None
        f_gt = g_at[mask] / np.maximum(gmax[mask], 1e-9)
        f_gt_loc = g_at[mask] / np.maximum(gmax_loc[mask], 1.0)
        f_0 = g0[mask] / np.maximum(gmax[mask], 1e-9)
        return {
            "n": int(mask.sum()),
            "frac_G_at_GT_over_gmax_ge0.7": float(np.mean(f_gt >= FRAC)),
            "frac_G_at_GT_over_LOCALmax_ge0.7": float(np.mean(f_gt_loc >= FRAC)),
            "frac_G_at_model_over_gmax_ge0.7": float(np.mean(f_0 >= FRAC)),
            "mean_G_ratio_at_GT": float(np.mean(f_gt)),
            "mean_G_ratio_at_GT_local": float(np.mean(np.minimum(f_gt_loc, 20.0))),
            "mean_G_ratio_at_model": float(np.mean(f_0)),
            "mean_n_peaks": float(np.mean(n_peak[mask])),
            "frac_multi_peak": float(np.mean(n_peak[mask] >= 2)),
            "frac_nearest_crest_is_within_3px_of_GT": float(np.mean(np.abs(dcrest[mask]) <= 3.0)),
            "frac_nearest_crest_is_within_3px_of_model": float(np.mean(np.abs(crest_t[mask]) <= 3.0)),
            "mean_abs_crest_minus_GT": float(np.nanmean(np.abs(dcrest[mask]))),
            "mean_step_across_model_bnd": float(np.mean(step0[mask])),
            "mean_step_across_GT_bnd": float(np.mean(stepgt[mask])),
            "frac_GT_is_strongest_edge_within_20px": float(np.mean(ratio20[mask] >= 0.9)),
            "mean_ratio_gt_peak_over_loc20": float(np.mean(np.minimum(ratio20[mask], 10.0))),
            "mean_G_at_GT_peak": float(np.mean(ggt[mask])),
        }

    out = {"split": args.split, "n_points": int(fin.sum()),
           "tail_threshold_px": TAIL,
           "all": stats(fin, "all"), "good_le2px": stats(good, "good"),
           "tail_gt10": stats(tail, "tail"),
           "mid_2to10": stats(fin & (at > 2.0) & (at <= TAIL), "mid")}

    # tail split by strength of the GT edge
    if tail.any():
        f_gt = g_at / np.maximum(gmax, 1e-9)
        weak = tail & (f_gt < 0.4)
        strong = tail & (f_gt >= 0.4)
        out["tail_weak_gt_edge"] = stats(weak, "weak")
        out["tail_strong_gt_edge"] = stats(strong, "strong")
        out["tail_g_ratio_hist"] = [
            {"lo": float(a), "hi": float(b),
             "n": int(np.sum(tail & (f_gt >= a) & (f_gt < b)))}
            for a, b in zip(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.01, 0.1))]

    with open(os.path.join(args.out_dir, "tail_evidence_%s.json" % args.split), "w") as fh:
        json.dump(out, fh, indent=1)

    print("=" * 86)
    print("TAIL EVIDENCE  split=%s  n_points=%d" % (args.split, out["n_points"]))
    print("  %-16s %6s %8s %9s %9s %7s %8s %8s %9s %8s" %
          ("group", "n", "Ggt>=.7", "Ggt_loc.7", "G0>=.7", "npeak", "crestGT", "stepGT",
           "GTpeak.9", "pk/loc20"))
    for k in ["all", "good_le2px", "mid_2to10", "tail_gt10",
              "tail_strong_gt_edge", "tail_weak_gt_edge"]:
        d = out.get(k)
        if not d:
            continue
        print("  %-16s %6d %8.3f %9.3f %9.3f %7.2f %8.3f %8.2f %9.3f %8.3f" %
              (k, d["n"], d["frac_G_at_GT_over_gmax_ge0.7"],
               d["frac_G_at_GT_over_LOCALmax_ge0.7"],
               d["frac_G_at_model_over_gmax_ge0.7"], d["mean_n_peaks"],
               d["frac_nearest_crest_is_within_3px_of_GT"],
               d["mean_step_across_GT_bnd"],
               d["frac_GT_is_strongest_edge_within_20px"],
               d["mean_ratio_gt_peak_over_loc20"]))
    print("-" * 86)
    for k in ["good_le2px", "mid_2to10", "tail_gt10"]:
        d = out.get(k)
        if not d:
            continue
        print("  [%s] mean G ratio at GT %.3f (vs local max %.3f) | at model %.3f | "
              "step model %.1f vs GT %.1f | multi-peak %.3f | crest=model %.3f" %
              (k, d["mean_G_ratio_at_GT"], d["mean_G_ratio_at_GT_local"],
               d["mean_G_ratio_at_model"], d["mean_step_across_model_bnd"],
               d["mean_step_across_GT_bnd"], d["frac_multi_peak"],
               d["frac_nearest_crest_is_within_3px_of_model"]))
    if "tail_g_ratio_hist" in out:
        print("  tail GT-edge-strength histogram (G(t_gt)/Gmax):")
        for b in out["tail_g_ratio_hist"]:
            print("    [%.1f,%.1f) %5d" % (b["lo"], b["hi"], b["n"]))


if __name__ == "__main__":
    main()

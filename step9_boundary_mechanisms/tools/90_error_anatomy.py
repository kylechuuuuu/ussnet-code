#!/usr/bin/env python3
"""90_error_anatomy.py -- where exactly does the residual boundary error live?

Motivation
----------
step7 showed that the learned sandwich band covers the GT boundary WORSE than a
constant-width isotropic band of the same area (70.1% vs 93.6%), i.e. the band
is a re-parameterised "the model is off by ~r px", not a shape-aware interval.
Before adding capacity to that mechanism we need to know which part of the
residual is actually structured, and what signal could possibly identify it.

Part A (val, needs the step6 sq2resencw25 npz)
  * |t_gt| along the surface normal of the resampled base contour (NPT=400),
    using the SDF zero-crossing (same estimator as step7 tools/73)
  * angular structure: which sector carries the bdice@1px loss
  * AT WHICH ANATOMY the desired boundary sits: PZ outer rim vs TZ outer edge
    vs the internal 127|255 interface  (new decomposition)
  * correlation of |t_gt| with |grad I| at the predicted and at the true point
  * along-contour coherence of the residual (circular autocorrelation)

Part B (all 102 GT, no model)
  * |grad I| and step contrast on: union outer boundary, split into PZ-outer and
    TZ-outer, and the internal 127|255 interface
  * boundary lengths -> how much extra supervision the zone split provides
  This validates (or kills) the premise of a zone-aware mechanism.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
STEP9 = os.path.dirname(HERE)
USS = os.path.dirname(STEP9)
STEP4_TOOLS = os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools")
STEP6 = os.path.join(USS, "step6_sandwich_squeeze")
STEP6_TOOLS = os.path.join(STEP6, "tools")
STEP2_SCRIPTS = os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts")

sys.path.insert(0, STEP6_TOOLS)
sys.path.insert(0, STEP2_SCRIPTS)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arb = _load("arb", os.path.join(STEP6_TOOLS, "63_arbitrate.py"))
s57 = arb.s57
s58 = arb.s58

NPZ_DEFAULT = os.path.join(STEP6, "predictions", "sq2resencw25_checkpoint_best_base_nifti")
MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
IMG_DIR = os.path.join(USS, "UltrasoundProstate", "images")

TS = np.arange(-20.0, 20.0001, 0.5)      # coarse probe grid
TG = np.arange(-20.0, 20.0001, 0.25)     # fine grid for the SDF zero crossing
NPT = 400


# --------------------------------------------------------------------- raw io
def raw_mask(stem):
    p = arb.find_file(MASK_DIR, stem, ".png")
    if p is None:
        return None
    return np.array(Image.open(p).convert("L"))


# --------------------------------------------------------------------- part A
def gt_position_along_normals(sdf, pts, nrm, tg=TG):
    """Signed position of the GT boundary along each normal (nan if not found).

    Negative = inward (inside the GT), positive = outward.  Same estimator as
    step7 tools/73_band_information.py: walk the SDF along the normal and take
    the zero crossing closest to t=0, linearly interpolated.
    """
    q = pts[:, None, :] + tg[None, :, None] * nrm[:, None, :]
    v = ndimage.map_coordinates(sdf, [q[..., 0].ravel(), q[..., 1].ravel()],
                                order=1, mode="nearest").reshape(len(pts), len(tg))
    sgn = np.sign(v)
    cross = (sgn[:, :-1] * sgn[:, 1:]) < 0
    out = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        idx = np.flatnonzero(cross[i])
        if idx.size == 0:
            continue
        j = idx[np.argmin(np.abs(tg[idx]))]
        v0, v1 = v[i, j], v[i, j + 1]
        t0, t1 = tg[j], tg[j + 1]
        out[i] = t0 + (t1 - t0) * (0.0 - v0) / (v1 - v0) if v1 != v0 else t0
    return out


def circular_halfwidth(seq):
    """First lag (in points) where the circular autocorrelation drops below .5."""
    x = seq - np.mean(seq)
    den = float(np.dot(x, x))
    if den <= 0:
        return np.nan
    n = len(x)
    for k in range(1, n // 2):
        r = float(np.dot(x, np.roll(x, k))) / den
        if r < 0.5:
            return float(k)
    return float(n // 2)


def label_at(mask, q):
    v = ndimage.map_coordinates(mask.astype(np.float32),
                                [q[:, 0], q[:, 1]], order=0, mode="nearest")
    return v.astype(np.int32)


def part_a(npz_dir, split, limit, out_dir):
    train, val = arb.read_split(USS)
    names = train if split == "train" else val
    if limit:
        names = names[:limit]

    per_case = []
    pts_rows = []
    pts_names = []
    for name in names:
        stem = arb.stem_of(name)
        gt = s57.load_gt(stem)
        raw = raw_mask(stem)
        if gt is None or raw is None:
            print("[90] WARN no gt/mask for %s" % stem)
            continue
        if raw.shape != gt.shape:
            print("[90] WARN shape mismatch %s" % stem)
            continue
        arr = arb.load_prob(npz_dir, stem)
        if arr is None:
            print("[90] WARN no npz for %s" % stem)
            continue
        p = arb.align_prob(arr, gt.shape)
        img = arb.load_image(stem)
        if img.shape != gt.shape:
            print("[90] WARN image shape %s" % stem)
            continue

        M0 = s57.largest_cc(p > 0.5)
        if not M0.any():
            print("[90] WARN empty pred %s" % stem)
            continue

        c = s57.contour_of(M0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, M0)
        sdf = s58.sdf_of(gt)
        t = gt_position_along_normals(sdf, pts, nrm)
        ok = np.isfinite(t)

        # ---- image evidence ------------------------------------------------
        imgf = ndimage.gaussian_filter(img, 1.5)
        gy, gx = np.gradient(imgf)
        gm = np.hypot(gx, gy)
        g_pred = ndimage.map_coordinates(gm, [pts[:, 0], pts[:, 1]],
                                         order=1, mode="nearest")
        gq = pts[ok] + t[ok, None] * nrm[ok]
        g_gt = ndimage.map_coordinates(gm, [gq[:, 0], gq[:, 1]],
                                       order=1, mode="nearest")
        g_all = gm.ravel()

        # ---- which anatomy does the GT boundary belong to? ----------------
        q_in = pts[ok] + (t[ok, None] - 2.5) * nrm[ok]
        q_out = pts[ok] + (t[ok, None] + 2.5) * nrm[ok]
        a_in = label_at(raw, q_in)
        a_out = label_at(raw, q_out)
        kind = np.full(ok.sum(), -1, dtype=np.int32)
        kind[(a_in == 127) & (a_out == 0)] = 0     # PZ outer rim
        kind[(a_in == 255) & (a_out == 0)] = 1     # TZ outer edge
        kind[(a_in == 127) & (a_out == 255)] = 2    # 127|255 interface from PZ
        kind[(a_in == 255) & (a_out == 127)] = 2    # ... from TZ
        kind[(a_in == 0) & (a_out == 0)] = 3        # no structure crossed

        # ---- angles --------------------------------------------------------
        yy, xx = np.nonzero(gt)
        cy, cx = float(yy.mean()), float(xx.mean())
        dy = pts[:, 0] - cy
        dx = pts[:, 1] - cx
        ang = np.degrees(np.arctan2(dy, dx))        # 90 deg = down = posterior
        ang = (ang + 180.0) % 360.0 - 180.0

        dice = s57.dice(M0, gt)
        bd1 = s57.bdice(M0, gt, 1)
        bd2 = s57.bdice(M0, gt, 2)

        tk = t[ok]
        rec = {
            "name": stem, "dice": dice, "bdice1": bd1, "bdice2": bd2,
            "n_ok": int(ok.sum()),
            "mean_abs_t": float(np.mean(np.abs(tk))),
            "p50_abs_t": float(np.median(np.abs(tk))),
            "p90_abs_t": float(np.percentile(np.abs(tk), 90)),
            "frac_t_gt1": float(np.mean(np.abs(tk) > 1.0)),
            "frac_t_gt3": float(np.mean(np.abs(tk) > 3.0)),
            "mean_signed_t": float(np.mean(tk)),
            "g_pred": float(np.mean(g_pred)), "g_gt": float(np.mean(g_gt)),
            "g_img": float(np.mean(g_all)),
            "coh_halfwidth_pts": circular_halfwidth(tk),
            "perimeter_px": float(np.sum(np.linalg.norm(
                np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1))),
            "area_gt_px": int(gt.sum()),
        }
        for k, nm in enumerate(["pz_outer", "tz_outer", "interface", "unclear"]):
            m = kind == k
            rec["n_" + nm] = int(m.sum())
            rec["abs_t_" + nm] = float(np.mean(np.abs(tk[m]))) if m.any() else None
            rec["miss1_" + nm] = float(np.mean(np.abs(tk[m]) > 1.0)) if m.any() else None
            rec["miss3_" + nm] = float(np.mean(np.abs(tk[m]) > 3.0)) if m.any() else None
        per_case.append(rec)
        pts_rows.append(np.stack([tk, ang[ok], kind.astype(float),
                                  g_pred[ok], g_gt], axis=1))
        pts_names.append(stem)

    # ---- aggregates -------------------------------------------------------
    def col(k):
        return np.array([r[k] for r in per_case if r[k] is not None], dtype=float)

    P = np.concatenate(pts_rows, axis=0)
    np.savez(os.path.join(out_dir, "error_anatomy_points_%s.npz" % split),
             data=P, names=np.array(pts_names))

    tk, ang, kind, gpred, ggt = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
    absT = np.abs(tk)
    nbins = 12
    edges = np.linspace(-180.0, 180.0, nbins + 1)
    bins = []
    tot_miss = float(np.sum(absT > 1.0))
    for i in range(nbins):
        m = (ang >= edges[i]) & (ang < edges[i + 1])
        if not m.any():
            continue
        bins.append({
            "lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(m.sum()),
            "mean_abs_t": float(np.mean(absT[m])),
            "mean_signed_t": float(np.mean(tk[m])),
            "frac_gt1": float(np.mean(absT[m] > 1.0)),
            "frac_gt3": float(np.mean(absT[m] > 3.0)),
            "miss_share": float(np.sum(absT[m] > 1.0) / max(1.0, tot_miss)),
        })

    sp = {}
    for nm, v in [("abs_t_vs_g_pred", gpred), ("abs_t_vs_g_gt", ggt)]:
        r = stats.spearmanr(absT, v)
        sp[nm] = {"rho": float(r.statistic), "p": float(r.pvalue)}
    sp["abs_t_vs_abs_signed_t"] = None

    per_kind = {}
    for k, nm in enumerate(["pz_outer", "tz_outer", "interface", "unclear"]):
        m = kind == k
        if not m.any():
            continue
        per_kind[nm] = {
            "n_points": int(m.sum()),
            "share": float(m.mean()),
            "mean_abs_t": float(np.mean(absT[m])),
            "median_abs_t": float(np.median(absT[m])),
            "frac_gt1": float(np.mean(absT[m] > 1.0)),
            "frac_gt3": float(np.mean(absT[m] > 3.0)),
            "mean_signed_t": float(np.mean(tk[m])),
            "miss_share": float(np.sum(absT[m] > 1.0) / max(1.0, tot_miss)),
            "mean_g_pred": float(np.mean(gpred[m])),
            "mean_g_gt": float(np.mean(ggt[m])),
        }
    # per-case miss accounting
    case_kind = {}
    for k, nm in enumerate(["pz_outer", "tz_outer", "interface", "unclear"]):
        case_kind[nm] = {
            "n_cases": int(sum(1 for r in per_case if r["n_" + nm] > 0)),
            "mean_n": float(np.mean([r["n_" + nm] for r in per_case])),
            "mean_abs_t": float(np.mean([r["abs_t_" + nm] for r in per_case
                                         if r["abs_t_" + nm] is not None])),
            "mean_miss1": float(np.mean([r["miss1_" + nm] for r in per_case
                                         if r["miss1_" + nm] is not None])),
            "mean_miss3": float(np.mean([r["miss3_" + nm] for r in per_case
                                         if r["miss3_" + nm] is not None])),
            "share_of_cases_worst": None,
        }

    agg = {
        "split": split, "n_cases": len(per_case), "n_points": int(len(tk)),
        "dice_mean": float(col("dice").mean()),
        "bdice1_mean": float(col("bdice1").mean()),
        "bdice2_mean": float(col("bdice2").mean()),
        "mean_abs_t": float(np.mean(absT)),
        "p50_abs_t": float(np.median(absT)),
        "p90_abs_t": float(np.percentile(absT, 90)),
        "frac_gt1": float(np.mean(absT > 1.0)),
        "frac_gt2": float(np.mean(absT > 2.0)),
        "frac_gt3": float(np.mean(absT > 3.0)),
        "frac_gt5": float(np.mean(absT > 5.0)),
        "coh_halfwidth_pts_median": float(np.nanmedian(col("coh_halfwidth_pts"))),
        "coh_halfwidth_pts_mean": float(np.nanmean(col("coh_halfwidth_pts"))),
        "coh_halfwidth_px_mean": float(np.nanmean(
            col("coh_halfwidth_pts") * col("perimeter_px") / NPT)),
        "g_pred_mean": float(np.mean(gpred)), "g_gt_mean": float(np.mean(ggt)),
        "g_img_mean": float(col("g_img").mean()),
        "angle_bins": bins, "spearman": sp,
        "per_kind_points": per_kind, "per_kind_cases": case_kind,
        "cases": per_case,
    }
    return agg


# --------------------------------------------------------------------- part B
def part_b(out_dir):
    train, val = arb.read_split(USS)
    names = list(train) + list(val)
    acc = {k: [] for k in ["outer_all", "outer_pz", "outer_tz", "interface",
                           "img_all", "contrast_outer", "contrast_interface",
                           "len_outer_all", "len_outer_pz", "len_outer_tz",
                           "len_interface"]}
    for name in names:
        stem = arb.stem_of(name)
        raw = raw_mask(stem)
        if raw is None:
            continue
        img = arb.load_image(stem)
        if img.shape != raw.shape:
            continue
        imgf = ndimage.gaussian_filter(img, 1.5)
        gy, gx = np.gradient(imgf)
        gm = np.hypot(gx, gy)

        union = raw > 0
        z1 = raw == 127
        z2 = raw == 255
        b_union = s57.surface(union)
        b1, b2 = s57.surface(z1), s57.surface(z2)
        near1 = s57.dilate(z1, 3)
        near2 = s57.dilate(z2, 3)
        outer_pz = b1 & ~near2
        outer_tz = b2 & ~near1
        iface = (b1 & near2) | (b2 & near1)

        def ctr(B, A, Breg):
            bi = s57.dilate(B, 3) & A
            bo = s57.dilate(B, 3) & Breg
            if not bi.any() or not bo.any():
                return np.nan
            return abs(float(img[bi].mean()) - float(img[bo].mean()))

        acc["outer_all"].append(float(gm[b_union].mean()))
        acc["outer_pz"].append(float(gm[outer_pz].mean()) if outer_pz.any() else np.nan)
        acc["outer_tz"].append(float(gm[outer_tz].mean()) if outer_tz.any() else np.nan)
        acc["interface"].append(float(gm[iface].mean()) if iface.any() else np.nan)
        acc["img_all"].append(float(gm.mean()))
        acc["contrast_outer"].append(ctr(b_union, union, ~union))
        acc["contrast_interface"].append(ctr(iface, z1, z2) if iface.any() else np.nan)
        acc["len_outer_all"].append(int(b_union.sum()))
        acc["len_outer_pz"].append(int(outer_pz.sum()))
        acc["len_outer_tz"].append(int(outer_tz.sum()))
        acc["len_interface"].append(int(iface.sum()))

    def st(v):
        v = np.array(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
                "n": int(v.size)}

    out = {k: st(v) for k, v in acc.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=NPZ_DEFAULT)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(STEP9, "results"))
    ap.add_argument("--skip-part-b", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    agg = part_a(args.npz_dir, args.split, args.limit, args.out_dir)
    with open(os.path.join(args.out_dir, "error_anatomy_%s.json" % args.split), "w") as fh:
        json.dump(agg, fh, indent=1)

    print("=" * 78)
    print("PART A  split=%s  n=%d cases  %d points" % (agg["split"], agg["n_cases"],
                                                       agg["n_points"]))
    print("  dice %.2f   bdice@1px %.2f   bdice@2px %.2f" %
          (agg["dice_mean"], agg["bdice1_mean"], agg["bdice2_mean"]))
    print("  |t_gt| mean %.2f  p50 %.2f  p90 %.2f px" %
          (agg["mean_abs_t"], agg["p50_abs_t"], agg["p90_abs_t"]))
    print("  frac |t|>1 %.3f   >2 %.3f   >3 %.3f   >5 %.3f" %
          (agg["frac_gt1"], agg["frac_gt2"], agg["frac_gt3"], agg["frac_gt5"]))
    print("  residual coherence: median half-width %.1f pts = %.1f px" %
          (agg["coh_halfwidth_pts_median"], agg["coh_halfwidth_px_mean"]))
    print("  |grad I|: pred pt %.2f   true boundary %.2f   image mean %.2f" %
          (agg["g_pred_mean"], agg["g_gt_mean"], agg["g_img_mean"]))
    print("  spearman(|t|,g_pred) rho=%.3f p=%.3g   (|t|,g_gt) rho=%.3f p=%.3g" %
          (agg["spearman"]["abs_t_vs_g_pred"]["rho"],
           agg["spearman"]["abs_t_vs_g_pred"]["p"],
           agg["spearman"]["abs_t_vs_g_gt"]["rho"],
           agg["spearman"]["abs_t_vs_g_gt"]["p"]))
    print("-" * 78)
    print("  ANATOMY of the wanted boundary (pooled points):")
    print("   %-10s %7s %7s %8s %8s %8s %8s %9s %8s" %
          ("kind", "n", "share", "|t|mean", "|t|med", "f>1", "f>3", "missshare", "g_gt"))
    for nm, d in agg["per_kind_points"].items():
        print("   %-10s %7d %7.3f %8.2f %8.2f %8.3f %8.3f %9.3f %8.2f" %
              (nm, d["n_points"], d["share"], d["mean_abs_t"], d["median_abs_t"],
               d["frac_gt1"], d["frac_gt3"], d["miss_share"], d["mean_g_gt"]))
    print("-" * 78)
    print("  ANATOMY of the wanted boundary (per case means):")
    for nm, d in agg["per_kind_cases"].items():
        print("   %-10s cases=%3d  n/case %6.1f  |t| %.2f  miss1 %.3f  miss3 %.3f" %
              (nm, d["n_cases"], d["mean_n"], d["mean_abs_t"], d["mean_miss1"],
               d["mean_miss3"]))
    print("-" * 78)
    print("  ANGULAR structure (90 deg = down/posterior):")
    print("   %-16s %6s %9s %9s %7s %7s %8s" %
          ("bin", "n", "|t|mean", "signed", "f>1", "f>3", "missshare"))
    for b in agg["angle_bins"]:
        print("   [%4.0f,%4.0f)      %6d %9.2f %9.2f %7.3f %7.3f %8.3f" %
              (b["lo"], b["hi"], b["n"], b["mean_abs_t"], b["mean_signed_t"],
               b["frac_gt1"], b["frac_gt3"], b["miss_share"]))
    print("-" * 78)
    print("  worst 6 cases by bdice@1px:")
    cs = sorted(agg["cases"], key=lambda r: r["bdice1"])[:6]
    for r in cs:
        print("   %-24s dice %5.2f bdice1 %5.2f |t| %.2f f>1 %.3f coh %.0fpts" %
              (r["name"], r["dice"], r["bdice1"], r["mean_abs_t"],
               r["frac_t_gt1"], r["coh_halfwidth_pts"]))

    if not args.skip_part_b:
        pb = part_b(args.out_dir)
        with open(os.path.join(args.out_dir, "interface_evidence.json"), "w") as fh:
            json.dump(pb, fh, indent=1)
        print("=" * 78)
        print("PART B  all 102 GT, no model:  |grad I| (sigma=1.5) and step contrast")
        def line(k, lab):
            d = pb[k]
            if d is None:
                print("   %-16s (none)" % lab)
                return
            print("   %-16s mean %7.2f  median %7.2f  p25 %7.2f  p75 %7.2f" %
                  (lab, d["mean"], d["median"], d["p25"], d["p75"]))
        line("img_all", "whole image")
        line("outer_all", "GT outer bnd")
        line("outer_pz", "  PZ outer")
        line("outer_tz", "  TZ outer")
        line("interface", "127|255 iface")
        print("   step contrast across the boundary (mean |I_in - I_out|):")
        line("contrast_outer", "  outer bnd")
        line("contrast_interface", "  127|255 iface")
        print("   boundary length px (mean):")
        for k in ("len_outer_all", "len_outer_pz", "len_outer_tz", "len_interface"):
            print("   %-16s %8.1f" % (k, pb[k]["mean"]))


if __name__ == "__main__":
    main()

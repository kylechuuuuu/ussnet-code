#!/usr/bin/env python3
"""92_wide_arbitration.py -- re-test the sandwich interval WITHOUT the +-8 px cap.

Why
---
step7 concluded that the learned sandwich band carries no shape information
(learned coverage 0.701 vs 0.936 for an equal-area isotropic band, 0/31 cases).
But step7 clipped the band at +-8 px (`arb.OFFS[-1]`) while the isotropic band it
was compared against was allowed +-12.8 px -- and 90/91 showed that on val 14.1%
of the contour points are more than 10 px away from the GT boundary and 6.8% are
more than 20 px away, almost all of them on the ANTERIOR side.  A +-8 px band
cannot represent the very errors that matter.

This script rebuilds the interval over a +-40 px probe range and asks:
  * does the (cautious, amplified) pair bracket the gross anterior excursions?
  * if it does, how much of the tail can a wide-range arbitration recover?
  * how much of the raw -> oracle gap does that close?

CPU only.
"""
import argparse
import csv
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
s57 = arb.s57
s58 = arb.s58

NPZ = os.path.join(STEP6, "predictions", "sq2resencw25_checkpoint_best")
NPT = 400
OFFS_W = np.arange(0.0, 40.0001, 0.5)     # wide probe for the interval
TS_W = np.arange(-48.0, 48.0001, 0.5)     # wide ridge search
TG = np.arange(-90.0, 90.0001, 0.5)       # wide SDF scan for t_gt
FRAC, SIGMA = 0.7, 1.5


def limits_off(pts, nrm, Min, Mout, offs):
    """arb.limits, but with an explicit (wider) probe grid.  Positive magnitudes."""
    qo = pts[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    vo = arb.sample_mask(Mout, qo)
    k_out = np.argmin(vo, axis=1)
    t_out = np.where(vo.all(axis=1), offs[-1], np.maximum(offs[k_out] - 0.5, 0.0))
    qi = pts[:, None, :] - offs[None, :, None] * nrm[:, None, :]
    vi = arb.sample_mask(Min, qi)
    first = np.argmax(vi, axis=1)
    t_in = np.where(vi.any(axis=1), np.maximum(offs[first] - 0.5, 0.0), offs[-1])
    return t_in, t_out


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


def hd_assd(pred, gt):
    gb, pb = s57.surface(gt), s57.surface(pred)
    dt_g = ndimage.distance_transform_edt(~gb)
    dt_p = ndimage.distance_transform_edt(~pb)
    d = np.r_[dt_g[pb], dt_p[gb]]
    return float(np.percentile(d, 95)), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-base", default=NPZ + "_base_nifti")
    ap.add_argument("--npz-cautious", default=NPZ + "_cautious_nifti")
    ap.add_argument("--npz-amplified", default=NPZ + "_amplified_nifti")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(STEP9, "results"))
    ap.add_argument("--tag", default="sq2resencw25")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train, val = arb.read_split(USS)
    names = train if args.split == "train" else val
    if args.limit:
        names = names[:args.limit]

    VARIANTS = ["raw", "free4", "band4", "bandU", "bandU_w12", "bandU_w24",
                "bandU_const", "oracle"]
    rows, cov_rows = [], []
    for name in names:
        stem = arb.stem_of(name)
        gt = s57.load_gt(stem)
        pb = arb.load_prob(args.npz_base, stem)
        pc = arb.load_prob(args.npz_cautious, stem)
        pa = arb.load_prob(args.npz_amplified, stem)
        if gt is None or pb is None or pc is None or pa is None:
            print("[92] skip %s" % stem)
            continue
        pb, pc, pa = (arb.align_prob(pb, gt.shape), arb.align_prob(pc, gt.shape),
                      arb.align_prob(pa, gt.shape))
        img = arb.load_image(stem)
        M0 = s57.largest_cc(pb > 0.5)
        Min = M0 & (pc > 0.5)
        Mout = M0 | (pa > 0.5)
        c = s57.contour_of(M0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, M0)
        t_in, t_out = limits_off(pts, nrm, Min, Mout, OFFS_W)
        sdf = s58.sdf_of(gt)
        t_gt = gt_pos(sdf, pts, nrm)
        ok = np.isfinite(t_gt)

        imgf = ndimage.gaussian_filter(img, SIGMA)
        G = arb.ridge_profile(imgf, pts, nrm, TS_W, 1.0)

        def evaluate(t, label):
            M, tt = arb.apply_t(M0, pts, nrm, t)
            rej = float(np.mean(np.abs(tt) < 1e-9) if np.any(t != 0) else 0.0)
            dice = s57.dice(M, gt)
            bd1 = s57.bdice(M, gt, 1)
            bd2 = s57.bdice(M, gt, 2)
            hd, asd = hd_assd(M, gt)
            return {"variant": label, "dice": dice, "bdice1": bd1, "bdice2": bd2,
                    "hd95": hd, "assd": asd,
                    "mean_abs_move": float(np.mean(np.abs(t))),
                    "frac_moved_gt1": float(np.mean(np.abs(t) > 1.0)),
                    "frac_moved_gt4": float(np.mean(np.abs(t) > 4.0)),
                    "frac_rejected": rej}

        def run(lo, hi, label):
            return evaluate(arb.pick(G, TS_W, FRAC, lo, hi), label)

        Z = np.zeros(NPT)
        rows.append(dict(run(Z, Z, "raw"), name=stem))
        rows.append(dict(run(-np.clip(t_in, 0, 4.0), np.clip(t_out, 0, 4.0), "band4"), name=stem))
        rows.append(dict(run(-np.full(NPT, 4.0), np.full(NPT, 4.0), "free4"), name=stem))
        rows.append(dict(run(-t_in, t_out, "bandU"), name=stem))
        rows.append(dict(run(-np.clip(t_in, 0, 12.0), np.clip(t_out, 0, 12.0),
                             "bandU_w12"), name=stem))
        rows.append(dict(run(-np.clip(t_in, 0, 24.0), np.clip(t_out, 0, 24.0),
                             "bandU_w24"), name=stem))
        ci, co = float(np.mean(t_in)), float(np.mean(t_out))
        rows.append(dict(run(-np.full(NPT, ci), np.full(NPT, co), "bandU_const"), name=stem))
        rows.append(dict(evaluate(np.where(ok, t_gt, 0.0), "oracle"), name=stem))

        # ---- does the wide interval bracket the tail? ----------------------
        lo, hi = -t_in, t_out
        tail = ok & (np.abs(t_gt) > 10.0)
        ins = ok & (t_gt >= lo) & (t_gt <= hi)
        area = float((Mout & ~Min).sum())
        per = float(max(1, s57.surface(M0).sum()))
        r_iso = area / (2.0 * per)
        cov_rows.append({
            "name": stem,
            "cov_bandU": float(ins[ok].mean()),
            "cov_band8": float((ok & (t_gt >= -np.clip(t_in, 0, 8)) &
                                (t_gt <= np.clip(t_out, 0, 8)))[ok].mean()),
            "cov_iso_same_area": float((np.abs(t_gt[ok]) <= r_iso).mean()),
            "cov_const": float((ok & (t_gt >= -ci) & (t_gt <= co))[ok].mean()),
            "n_tail": int(tail.sum()),
            "cov_bandU_tail": float(ins[tail].mean()) if tail.any() else np.nan,
            "cov_band8_tail": float((ok & (t_gt >= -np.clip(t_in, 0, 8)) &
                                     (t_gt <= np.clip(t_out, 0, 8)))[tail].mean())
            if tail.any() else np.nan,
            "band_px": area / per, "r_iso": r_iso,
            "mean_t_in": float(np.mean(t_in)), "mean_t_out": float(np.mean(t_out)),
            "sat_in": float(np.mean(t_in >= OFFS_W[-1] - 1e-6)),
            "sat_out": float(np.mean(t_out >= OFFS_W[-1] - 1e-6)),
            "raw_bdice1": rows[-8]["bdice1"], "oracle_bdice1": rows[-1]["bdice1"],
        })
        print("  %-24s bandU cov %.3f (tail %.3f)  band8 cov %.3f (tail %.3f)  "
              "iso %.3f  | in %.1f/out %.1f px sat %.2f/%.2f" %
              (stem, cov_rows[-1]["cov_bandU"], cov_rows[-1]["cov_bandU_tail"] or -1,
               cov_rows[-1]["cov_band8"], cov_rows[-1]["cov_band8_tail"] or -1,
               cov_rows[-1]["cov_iso_same_area"], cov_rows[-1]["mean_t_in"],
               cov_rows[-1]["mean_t_out"], cov_rows[-1]["sat_in"], cov_rows[-1]["sat_out"]))

    # ---- aggregate ---------------------------------------------------------
    agg_cov = {}
    for k in ["cov_bandU", "cov_band8", "cov_iso_same_area", "cov_const",
              "cov_bandU_tail", "cov_band8_tail", "band_px", "r_iso",
              "mean_t_in", "mean_t_out", "sat_in", "sat_out"]:
        v = np.array([r[k] for r in cov_rows], dtype=float)
        agg_cov[k] = float(np.nanmean(v))
    agg_cov["n_cases"] = len(cov_rows)
    for k in ["cov_bandU", "cov_band8", "cov_iso_same_area"]:
        a = np.array([r[k] for r in cov_rows], dtype=float)
        agg_cov[k + "_win_vs_band8"] = int(np.sum(a > np.array(
            [r["cov_band8"] for r in cov_rows])))
    agg = {}
    for v in VARIANTS:
        sub = [r for r in rows if r["variant"] == v]
        agg[v] = {k: float(np.mean([r[k] for r in sub]))
                  for k in ["dice", "bdice1", "bdice2", "hd95", "assd",
                            "mean_abs_move", "frac_moved_gt1", "frac_moved_gt4",
                            "frac_rejected"]}
        agg[v]["n"] = len(sub)
    with open(os.path.join(args.out_dir, "wide_arbitration_%s_%s.json" %
                           (args.tag, args.split)), "w") as fh:
        json.dump({"variants": agg, "coverage": agg_cov, "per_case": rows,
                   "per_case_coverage": cov_rows}, fh, indent=1)
    with open(os.path.join(args.out_dir, "wide_arbitration_%s_%s.csv" %
                           (args.tag, args.split)), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("=" * 92)
    print("WIDE ARBITRATION  %s  split=%s  n=%d" % (args.tag, args.split, len(cov_rows)))
    print("  coverage of the GT boundary by the interval:")
    print("    learned  +-8px (step7)      %.4f      (on the |t|>10 tail: %.4f)" %
          (agg_cov["cov_band8"], agg_cov["cov_band8_tail"]))
    print("    learned  uncapped +-40px    %.4f      (on the |t|>10 tail: %.4f)" %
          (agg_cov["cov_bandU"], agg_cov["cov_bandU_tail"]))
    print("    equal-area isotropic        %.4f   (r_iso = %.2f px)" %
          (agg_cov["cov_iso_same_area"], agg_cov["r_iso"]))
    print("    constant band, same means   %.4f   (mean in %.2f / out %.2f px)" %
          (agg_cov["cov_const"], agg_cov["mean_t_in"], agg_cov["mean_t_out"]))
    print("    saturation at +-40px: in %.3f  out %.3f" %
          (agg_cov["sat_in"], agg_cov["sat_out"]))
    print("    cases where uncapped beats capped: %d/%d" %
          (agg_cov["cov_bandU_win_vs_band8"], agg_cov["n_cases"]))
    print("-" * 92)
    print("  %-12s %7s %8s %8s %8s %8s %8s %8s" %
          ("variant", "dice", "bdice1", "bdice2", "HD95", "ASSD",
           "|move|", "f>4"))
    for v in VARIANTS:
        d = agg[v]
        print("  %-12s %7.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.3f" %
              (v, d["dice"], d["bdice1"], d["bdice2"], d["hd95"], d["assd"],
               d["mean_abs_move"], d["frac_moved_gt4"]))
    print("  (frac_rejected by the area sanity check: raw %.3f bandU %.3f oracle %.3f)" %
          (agg["raw"]["frac_rejected"], agg["bandU"]["frac_rejected"],
           agg["oracle"]["frac_rejected"]))


if __name__ == "__main__":
    main()

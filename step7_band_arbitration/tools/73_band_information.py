#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_band_arbitration/tools/73_band_information.py

The decisive question for the compressed-boundary mechanism is NOT "does the
band raise Dice" (on the validation split it does not: a plain +/-4 px window
scores the same).  It is:

    does the LEARNED, angularly varying band predict WHERE the boundary may
    still be better than a constant-width band of the same size?

If yes, the band is a genuine per-location uncertainty field and can be used as
a calibrated trust region (and as the referral signal in step 8).  If no, the
band is just a re-parameterisation of "the model is off by about r pixels".

Method -- everything is measured along the contour normal, so no rasterisation
artefact enters:

  * the model contour is resampled to NPT points; at every point the three heads
    give a signed interval [t_in, t_out] (inside = negative, outward = positive);
  * the ground-truth boundary position along the same normal, t_gt, is found as
    the nearest zero crossing of the GT signed distance field;
  * learned-band coverage   = fraction of points with t_in <= t_gt <= t_out;
  * isotropic band of the SAME AREA has half-width r = A / (2 * perimeter), and
    its coverage is fraction of points with |t_gt| <= r.

Reported per split, with a paired per-case comparison of the two coverages.

usage:
  python tools/73_band_information.py [--split val] [--npt 400]
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
STEP7 = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
STEP6 = os.path.join(USS, "step6_sandwich_squeeze")
sys.path.insert(0, os.path.join(STEP6, "tools"))
sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))
sys.path.insert(0, os.path.join(USS, "step2_compare_unet_vs_nnunet", "scripts"))

arb = importlib.import_module("63_arbitrate")                          # noqa: E402
s57, s58 = arb.s57, arb.s58                                            # noqa: E402
read_split = arb.read_split                                            # noqa: E402

NPZ = os.path.join(STEP6, "predictions", "sq2resencw25_checkpoint_best")
MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")
TG = np.arange(-20.0, 20.0001, 0.25)


def gt_position_along_normals(sdf, pts, nrm, tg=TG):
    """Signed position of the GT boundary on each normal line (nan if none)."""
    npt = pts.shape[0]
    coords = np.empty((2, npt, tg.size), dtype=np.float64)
    coords[0] = pts[:, 0][:, None] + nrm[:, 0][:, None] * tg[None, :]
    coords[1] = pts[:, 1][:, None] + nrm[:, 1][:, None] * tg[None, :]
    vals = ndimage.map_coordinates(sdf, coords, order=1, mode="nearest")
    sgn = np.sign(vals)
    cross = (sgn[:, :-1] * sgn[:, 1:]) <= 0
    out = np.full(npt, np.nan)
    for i in range(npt):
        js = np.flatnonzero(cross[i])
        if js.size == 0:
            continue
        j = js[np.argmin(np.abs(tg[js]))]          # crossing nearest to t = 0
        v0, v1 = vals[i, j], vals[i, j + 1]
        f = 0.0 if abs(v1 - v0) < 1e-9 else v0 / (v0 - v1)
        out[i] = tg[j] + f * (tg[j + 1] - tg[j])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--npt", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(STEP7, "results", "band_information.json"))
    args = ap.parse_args()

    train_names, val_names = read_split(USS)
    names = train_names if args.split == "train" else val_names
    if args.limit:
        names = names[:args.limit]
    stems = [os.path.splitext(n)[0] for n in names]
    print("[73] split=%s n=%d" % (args.split, len(names)))

    per_case, all_tgt, all_tin, all_tout, all_th = [], [], [], [], []
    for k, (n, stem) in enumerate(zip(names, stems)):
        gt = s57.load_gt(stem)
        rb = arb.load_prob(os.path.join(NPZ + "_base_nifti"), stem)
        rc = arb.load_prob(os.path.join(NPZ + "_cautious_nifti"), stem)
        ra = arb.load_prob(os.path.join(NPZ + "_amplified_nifti"), stem)
        if rb is None or rc is None or ra is None:
            print("[73] skip %s (missing npz)" % stem)
            continue
        pb = arb.align_prob(rb, gt.shape)
        pc = arb.align_prob(rc, gt.shape)
        pa = arb.align_prob(ra, gt.shape)
        M0 = s57.largest_cc(pb > 0.5)
        Min = M0 & (pc > 0.5)
        Mout = M0 | (pa > 0.5)
        pts = s57.resample_closed(s57.contour_of(M0), args.npt, 0)
        nrm = s58.normals_of(pts, M0)
        # arb.limits returns POSITIVE magnitudes (inward room, outward room);
        # the signed interval along the normal is [-t_in, +t_out].
        t_in_mag, t_out_mag = arb.limits(pts, nrm, Min, Mout)
        tin = -np.clip(t_in_mag, 0.0, float(arb.OFFS[-1]))
        tout = np.clip(t_out_mag, 0.0, float(arb.OFFS[-1]))
        sdf = s58.sdf_of(gt)
        tgt = gt_position_along_normals(sdf, pts, nrm)
        ok = ~np.isnan(tgt)
        if ok.sum() < args.npt // 2:
            print("[73] skip %s (only %d/%d normals reach the GT)" % (stem, ok.sum(), args.npt))
            continue
        ti, to, tg_ = tin[ok], tout[ok], tgt[ok]
        learned = float(((tg_ >= ti) & (tg_ <= to)).mean())
        area = float((Mout & ~Min).sum())
        per = float(max(1, s57.surface(M0).sum()))
        r_iso = area / (2.0 * per)                    # half-width of the equal-area isotropic band
        iso = float((np.abs(tg_) <= r_iso).mean())
        # pixel-space containment (step 6's definition) for continuity
        gb = s57.surface(gt)
        contain_px = float((gb & (Mout & ~Min)).sum() / max(1, gb.sum()))
        # a CONSTANT band with the same mean inward/outward room isolates the
        # angular VARIATION of the learned band from its mean asymmetry
        ti_c, to_c = float(np.mean(ti)), float(np.mean(to))
        const = float(((tg_ >= ti_c) & (tg_ <= to_c)).mean())
        miss_out = float((tg_ > to).mean())
        miss_in = float((tg_ < ti).mean())
        try:
            from scipy.stats import spearmanr
            rho = float(spearmanr(to - ti, np.abs(tg_)).statistic)
        except Exception:
            rho = float("nan")
        per_case.append({
            "name": n, "learned": learned, "iso_same_area": iso,
            "const_band": const, "miss_out": miss_out, "miss_in": miss_in,
            "rho_width_abstgt": rho,
            "band_area": area, "band_px": area / per, "r_iso": float(r_iso),
            "contain_px": contain_px, "n_ok": int(ok.sum()),
            "mean_tgt": float(np.mean(np.abs(tg_))),
            "mean_tout": float(np.mean(to)), "mean_tin": float(np.mean(ti)),
            "dice": float(s57.dice(M0, gt)), "bdice1": float(s57.bdice(M0, gt, 1)),
        })
        all_tgt.append(tg_); all_tin.append(ti); all_tout.append(to)
        # angle of each point about the GT centroid, +90 deg = image up
        cy, cx = ndimage.center_of_mass(gt)
        th = np.degrees(np.arctan2(-(pts[ok, 0] - cy), pts[ok, 1] - cx))
        all_th.append(th)
        if (k + 1) % 10 == 0:
            print("[73] %d/%d" % (k + 1, len(names)))

    if not per_case:
        raise SystemExit("no cases processed")

    L = np.array([c["learned"] for c in per_case])
    I = np.array([c["iso_same_area"] for c in per_case])
    d = L - I
    print("")
    print("[73] cases=%d" % len(per_case))
    print("[73] learned-band coverage      %.4f" % L.mean())
    print("[73] equal-area isotropic cover %.4f" % I.mean())
    print("[73] per-case delta             %+.4f   (win %d / loss %d)"
          % (d.mean(), int((d > 0).sum()), int((d < 0).sum())))
    print("[73] mean band width            %.2f px  (equal-area r = %.2f px)"
          % (np.mean([c["band_px"] for c in per_case]),
             np.mean([c["r_iso"] for c in per_case])))
    print("[73] pixel-space containment    %.4f" % np.mean([c["contain_px"] for c in per_case]))

    try:
        from scipy.stats import wilcoxon
        nz = d[np.abs(d) > 1e-9]
        if len(nz) >= 5:
            print("[73] Wilcoxon learned vs iso   p = %.4f" % float(wilcoxon(nz).pvalue))
    except Exception:
        pass

    C = np.array([c["const_band"] for c in per_case])
    MO = np.array([c["miss_out"] for c in per_case])
    MI = np.array([c["miss_in"] for c in per_case])
    RH = np.array([c["rho_width_abstgt"] for c in per_case])
    print("[73] constant-signature cover  %.4f  (same mean in/out room, no angular variation)" % C.mean())
    print("[73]   learned - constant       %+.4f   (win %d / loss %d)"
          % ((L - C).mean(), int((L > C).sum()), int((L < C).sum())))
    print("[73] miss split: outward %.4f  inward %.4f  (band is lopsided)"
          % (MO.mean(), MI.mean()))
    print("[73] mean asymmetry  t_in %.2f px  t_out %.2f px" % (np.concatenate(all_tin).mean(), np.concatenate(all_tout).mean()))
    print("[73] Spearman(width, |t_gt|) per case: mean %.3f  (positive %d / %d cases)"
          % (np.nanmean(RH), int(np.nansum(RH > 0)), len(RH)))
    try:
        from scipy.stats import wilcoxon as _w
        nzc = (L - C)[np.abs(L - C) > 1e-9]
        if len(nzc) >= 5:
            print("[73] Wilcoxon learned vs const  p = %.4f" % float(_w(nzc).pvalue))
    except Exception:
        pass

    # ------------------------------------------------ is the angular shape real?
    tg = np.concatenate(all_tgt)
    ti = np.concatenate(all_tin)
    to = np.concatenate(all_tout)
    th = np.concatenate(all_th)
    w = to - ti
    print("\n[73] |t_gt| by learned-band-width quartile (an informative band must rise):")
    qs = np.percentile(w, [0, 25, 50, 75, 100])
    for i in range(4):
        m = (w >= qs[i]) & (w <= qs[i + 1]) if i == 3 else (w >= qs[i]) & (w < qs[i + 1])
        if m.sum() == 0:
            continue
        print("[73]   Q%d width %5.2f-%5.2f px  mean|t_gt| %.2f  cover %.3f  n=%d"
              % (i + 1, qs[i], qs[i + 1], np.abs(tg[m]).mean(),
                 float(((tg[m] >= ti[m]) & (tg[m] <= to[m])).mean()), int(m.sum())))

    print("\n[73] by 30-degree sector about the GT centroid (+90 = image up):")
    print("[73]   %-8s %8s %9s %8s %8s" % ("sector", "n", "band_px", "|t_gt|", "cover"))
    edges = np.arange(-180, 181, 30)
    for i in range(len(edges) - 1):
        m = (th >= edges[i]) & (th < edges[i + 1])
        if m.sum() < 50:
            continue
        print("[73]   %+4d..%+4d %8d %9.2f %8.2f %8.3f"
              % (edges[i], edges[i + 1], int(m.sum()), w[m].mean(), np.abs(tg[m]).mean(),
                 float(((tg[m] >= ti[m]) & (tg[m] <= to[m])).mean())))

    out = {"split": args.split, "n": len(per_case), "npt": args.npt,
           "learned_coverage": float(L.mean()), "iso_same_area_coverage": float(I.mean()),
           "delta_mean": float(d.mean()), "delta_win": int((d > 0).sum()),
           "delta_loss": int((d < 0).sum()),
           "mean_band_px": float(np.mean([c["band_px"] for c in per_case])),
           "mean_r_iso": float(np.mean([c["r_iso"] for c in per_case])),
           "contain_px": float(np.mean([c["contain_px"] for c in per_case])),
           "per_case": per_case}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[73] wrote %s" % args.out)


if __name__ == "__main__":
    main()

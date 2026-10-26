#!/usr/bin/env python3
"""91_tail_anatomy.py -- is the residual a thin scatter, or a few gross bulges?

90_error_anatomy showed that on val, 9.4% of the resampled base-contour points
have NO GT boundary crossing within +-20 px along their normal.  HD95 (28.2 px)
is by construction the 95th percentile of surface distance, i.e. it is set by
exactly that tail.  So the question "what mechanism is still missing" is really
"what are those >20 px excursions".

This script measures, per case:
  * the class histogram of |t_gt| with the search window widened to +-80 px
  * whether the tail is CONTIGUOUS (one bulge / one bite per case) or scattered
  * the anatomy (PZ outer vs TZ outer) and the side (overshoot/undershoot) of it
  * HD95 / ASSD computed exactly like step2 eval_lib, and their correlation with
    the tail size across cases

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
try:
    import eval_lib as ev
except Exception:
    ev = None

NPZ_DEFAULT = os.path.join(STEP6, "predictions", "sq2resencw25_checkpoint_best_base_nifti")
TSF = np.arange(-80.0, 80.0001, 0.5)
NPT = 400
CLASS_EDGES = [2.0, 5.0, 10.0, 20.0]
CLASS_NAMES = ["le2", "2to5", "5to10", "10to20", "gt20", "nocross"]


def scan(sdf, pts, nrm, ts=TSF):
    """Nearest-to-zero SDF crossing along each normal, over a wide window."""
    q = pts[:, None, :] + ts[None, :, None] * nrm[:, None, :]
    v = ndimage.map_coordinates(sdf, [q[..., 0].ravel(), q[..., 1].ravel()],
                                order=1, mode="nearest").reshape(len(pts), len(ts))
    t = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        s = np.sign(v[i])
        idx = np.flatnonzero(s[:-1] * s[1:] < 0)
        if idx.size == 0:
            continue
        j = idx[np.argmin(np.abs(ts[idx]))]
        v0, v1 = v[i, j], v[i, j + 1]
        t[i] = ts[j] + (ts[j + 1] - ts[j]) * (0.0 - v0) / (v1 - v0) if v1 != v0 else ts[j]
    return t, v[:, np.argmin(np.abs(ts))]


def circ_run_max(flag):
    n = len(flag)
    if not flag.any():
        return 0
    if flag.all():
        return n
    f = np.r_[flag, flag]
    best = cur = 0
    for x in f:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return min(best, n)


def hd_assd(pred, gt):
    gb, pb = s57.surface(gt), s57.surface(pred)
    if gb.sum() == 0 or pb.sum() == 0:
        return np.nan, np.nan, np.nan
    dt_g = ndimage.distance_transform_edt(~gb)
    dt_p = ndimage.distance_transform_edt(~pb)
    d = np.r_[dt_g[pb], dt_p[gb]]
    return (float(np.percentile(d, 95)), float(d.mean()), float(d.max()))


def label_kind(raw, q_in, q_out):
    a_in = ndimage.map_coordinates(raw.astype(np.float32), [q_in[:, 0], q_in[:, 1]],
                                   order=0, mode="nearest").astype(np.int32)
    a_out = ndimage.map_coordinates(raw.astype(np.float32), [q_out[:, 0], q_out[:, 1]],
                                    order=0, mode="nearest").astype(np.int32)
    kind = np.full(len(a_in), 3, dtype=np.int32)
    kind[(a_in == 127) & (a_out == 0)] = 0
    kind[(a_in == 255) & (a_out == 0)] = 1
    kind[(a_in == 127) & (a_out == 255)] = 2
    kind[(a_in == 255) & (a_out == 127)] = 2
    return kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=NPZ_DEFAULT)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(STEP9, "results"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train, val = arb.read_split(USS)
    names = train if args.split == "train" else val
    if args.limit:
        names = names[:args.limit]

    cases = []
    all_t, all_kind, all_ang = [], [], []
    for name in names:
        stem = arb.stem_of(name)
        gt = s57.load_gt(stem)
        raw = arb.raw_mask(stem) if hasattr(arb, "raw_mask") else None
        if raw is None:
            p = arb.find_file(os.path.join(USS, "UltrasoundProstate", "masks"), stem, ".png")
            from PIL import Image
            raw = np.array(Image.open(p).convert("L")) if p else None
        arr = arb.load_prob(args.npz_dir, stem)
        if gt is None or raw is None or arr is None:
            print("[91] skip %s" % stem)
            continue
        p = arb.align_prob(arr, gt.shape)
        M0 = s57.largest_cc(p > 0.5)
        c = s57.contour_of(M0)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        nrm = s58.normals_of(pts, M0)
        sdf = s58.sdf_of(gt)
        t, v0 = scan(sdf, pts, nrm)

        found = np.isfinite(t)
        a = np.abs(np.nan_to_num(t, nan=999.0))
        cls = np.full(NPT, 5, dtype=np.int32)          # 5 = no crossing in +-80
        cls[found & (a <= 2.0)] = 0
        cls[found & (a > 2.0) & (a <= 5.0)] = 1
        cls[found & (a > 5.0) & (a <= 10.0)] = 2
        cls[found & (a > 10.0) & (a <= 20.0)] = 3
        cls[found & (a > 20.0)] = 4
        # sign for the tail: overshoot (pred too big, t<0) vs undershoot
        tail = (cls >= 3)
        n_tail = int(tail.sum())
        n_tail10 = int((cls >= 3).sum())
        n_tail20 = int((cls >= 4).sum())
        n_nocross = int((cls == 5).sum())
        run3 = circ_run_max(tail)
        run4 = circ_run_max(cls >= 4)
        # anatomy of the tail
        tq = np.where(found, t, np.where(v0 > 0, 40.0, -40.0))
        q_in = pts + (tq[:, None] - 2.5) * nrm
        q_out = pts + (tq[:, None] + 2.5) * nrm
        kind = label_kind(raw, q_in, q_out)
        dice = s57.dice(M0, gt)
        bd1 = s57.bdice(M0, gt, 1)
        hd95, assd, hdmax = hd_assd(M0, gt)
        yy, xx = np.nonzero(gt)
        ang = (np.degrees(np.arctan2(pts[:, 0] - yy.mean(),
                                     pts[:, 1] - xx.mean())) + 180.0) % 360.0 - 180.0

        rec = {
            "name": stem, "dice": dice, "bdice1": bd1, "hd95": hd95, "assd": assd,
            "hdmax": hdmax,
            "counts": {CLASS_NAMES[i]: int((cls == i).sum()) for i in range(6)},
            "share_tail": n_tail / float(NPT),
            "run_tail10": run3, "run_tail20": run4,
            "run_tail10_share": run3 / float(NPT),
            "mean_abs_t": float(np.mean(np.abs(t[found]))) if found.any() else None,
            "mean_signed_t": float(np.mean(t[found])) if found.any() else None,
            "tail_area_px": float(np.sum(np.abs(t[found]) * 1143.0 / NPT)
                                  * (n_tail / float(NPT))) if found.any() else None,
        }
        for k, nm in enumerate(["pz_outer", "tz_outer", "interface", "unclear"]):
            m = tail & (kind == k)
            rec["tail_" + nm] = int(m.sum())
        rec["tail_overshoot"] = int(np.sum(tail & np.isfinite(t) & (t < 0)))
        rec["tail_undershoot"] = int(np.sum(tail & np.isfinite(t) & (t > 0)))
        cases.append(rec)
        all_t.append(t)
        all_kind.append(kind)
        all_ang.append(ang)
        print("  %-24s dice %5.2f bdice1 %5.2f HD95 %6.2f  tail10 %3d (run %3d)  "
              "tail20 %3d  nocross %3d" %
              (stem, dice, bd1, hd95, n_tail, run3, n_tail20, n_nocross))

    T = np.concatenate([np.where(np.isfinite(t), t, np.nan) for t in all_t])
    K = np.concatenate(all_kind)
    A = np.concatenate(all_ang)
    absT = np.abs(T)
    tailmask = np.isfinite(T) & (absT > 10.0)

    agg = {"split": args.split, "n_cases": len(cases), "n_points": int(len(T)),
           "cases": cases,
           "class_share": {CLASS_NAMES[i]: float(np.mean(
               [c["counts"][CLASS_NAMES[i]] for c in cases]) / NPT) for i in range(6)},
           "tail_share_points": float(np.mean(absT[np.isfinite(absT)] > 10.0)),
           "tail_share_percase_mean": float(np.mean([c["share_tail"] for c in cases])),
           "run_tail10_mean": float(np.mean([c["run_tail10"] for c in cases])),
           "run_tail10_median": float(np.median([c["run_tail10"] for c in cases])),
           "cases_with_tail": int(sum(1 for c in cases if c["run_tail10"] > 0)),
           "cases_tail_contiguous_frac": float(np.mean(
               [c["run_tail10"] / max(1, c["counts"]["10to20"] + c["counts"]["gt20"] +
                                      c["counts"]["nocross"]) for c in cases])),
           "hd95_mean": float(np.mean([c["hd95"] for c in cases])),
           "assd_mean": float(np.mean([c["assd"] for c in cases])),
           "hdmax_mean": float(np.mean([c["hdmax"] for c in cases])),
           }
    # correlation of HD95 with the tail
    from scipy import stats as st
    hd = np.array([c["hd95"] for c in cases])
    for key, lab in [("run_tail10", "run of |t|>10"),
                     ("counts.gt20", "count |t|>20"),
                     ("counts.nocross", "count no-crossing"),
                     ("share_tail", "share |t|>10"),
                     ("assd", "ASSD")]:
        v = np.array([(c["counts"]["gt20"] + c["counts"]["nocross"]) if key == "counts.gt20"
                      else c["counts"]["gt20"] if key == "counts.gt20"
                      else c["counts"]["nocross"] if key == "counts.nocross"
                      else c[key] for c in cases], dtype=float)
        if key == "counts.gt20":
            v = np.array([c["counts"]["gt20"] for c in cases], dtype=float)
        r = st.spearmanr(hd, v)
        agg["hd95_vs_" + key.replace(".", "_")] = {"rho": float(r.statistic),
                                                   "p": float(r.pvalue), "label": lab}
    # anatomy of the tail points
    anat = {}
    for k, nm in enumerate(["pz_outer", "tz_outer", "interface", "unclear"]):
        m = tailmask & (K == k)
        if not m.any():
            continue
        anat[nm] = {"n": int(m.sum()), "share_of_tail": float(m.sum() / max(1, tailmask.sum())),
                    "share_of_kind": float(m.sum() / max(1, int((K == k).sum())))}
    agg["tail_anatomy"] = anat
    # angular histogram of tail points
    nb = 12
    edges = np.linspace(-180, 180, nb + 1)
    agg["tail_angle_bins"] = [
        {"lo": float(edges[i]), "hi": float(edges[i + 1]),
         "n_tail": int(np.sum(tailmask & (A >= edges[i]) & (A < edges[i + 1]))),
         "n_all": int(np.sum((A >= edges[i]) & (A < edges[i + 1])))}
        for i in range(nb)]
    agg["overshoot_total"] = int(sum(c["tail_overshoot"] for c in cases))
    agg["undershoot_total"] = int(sum(c["tail_undershoot"] for c in cases))
    agg["nocross_total"] = int(sum(c["counts"]["nocross"] for c in cases))

    tag = args.tag or args.split
    with open(os.path.join(args.out_dir, "tail_anatomy_%s.json" % tag), "w") as fh:
        json.dump(agg, fh, indent=1)

    print("=" * 78)
    print("TAIL ANATOMY  split=%s  n=%d cases" % (args.split, len(cases)))
    print("  class shares of the 400 resampled contour points (per case mean):")
    for nm in CLASS_NAMES:
        print("    |t| %-8s %.3f" % (nm, agg["class_share"][nm]))
    print("  HD95 %.2f   ASSD %.2f   maxdist %.2f" %
          (agg["hd95_mean"], agg["assd_mean"], agg["hdmax_mean"]))
    print("  tail |t|>10: %d cases affected; mean run %.1f pts, median %.1f pts" %
          (agg["cases_with_tail"], agg["run_tail10_mean"], agg["run_tail10_median"]))
    print("  tail contiguity: mean(run / n_tail_points) = %.3f  (1.0 = one bulge)" %
          agg["cases_tail_contiguous_frac"])
    print("  overshoot pts %d   undershoot pts %d   no-crossing pts %d" %
          (agg["overshoot_total"], agg["undershoot_total"], agg["nocross_total"]))
    print("  correlation across cases with HD95:")
    for k, v in agg.items():
        if k.startswith("hd95_vs_"):
            print("    %-16s rho %+.3f  p %.3g" % (v["label"], v["rho"], v["p"]))
    print("  anatomy of the tail points:")
    for nm, d in agg["tail_anatomy"].items():
        print("    %-10s n %5d  share of tail %.3f  share of kind %.3f" %
              (nm, d["n"], d["share_of_tail"], d["share_of_kind"]))
    print("  tail by angle (90 = down/posterior):")
    for b in agg["tail_angle_bins"]:
        r = b["n_tail"] / max(1, b["n_all"])
        print("    [%4.0f,%4.0f)  n_all %4d  n_tail %4d  rate %.3f" %
              (b["lo"], b["hi"], b["n_all"], b["n_tail"], r))
    print("  worst 8 cases by HD95:")
    for c in sorted(cases, key=lambda r: -r["hd95"])[:8]:
        print("    %-24s HD95 %6.2f bdice1 %5.2f run10 %3d gt20 %3d nocross %3d" %
              (c["name"], c["hd95"], c["bdice1"], c["run_tail10"],
               c["counts"]["gt20"], c["counts"]["nocross"]))


if __name__ == "__main__":
    main()

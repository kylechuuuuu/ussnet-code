#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_boundary_mechanisms/tools/99r_pool_sweep.py

Wider ensemble search.  Until now the pool only held the step-6 arms; but the
project has two more sources of genuinely independent probability fields:

  * step4: 7 arms of a DIFFERENT architecture family (ResUNet / pre-encoder
    boundary models, `fp*`), 102 cases each, named by case id;
  * step8: the strongest model re-run on appearance-perturbed images
    (`sq2shift_*`, 31 val cases each, named by the original Chinese filename) -
    i.e. image-level test-time augmentation that is already on disk.

  phase 1  singles:  bdice@1px / dice of every pool member alone
  phase 2  greedy:   start from the shipped seed {res_best:6, w25:1, w25s:1,
                     sq2:1} and repeatedly add the (member, weight) that most
                     improves the full-val bdice@1px, stopping when nothing
                     improves.  This is selection on val, so the result is an
                     upper bound; run 99p-style hold-out before believing it.

usage:
  python tools/99r_pool_sweep.py --phase singles
  python tools/99r_pool_sweep.py --phase greedy [--rounds 8]
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np

USS = "/media/72TB_disk/personal/zk/USSnet"
S4 = os.path.join(USS, "step4_our_new_model_resunet_boundary", "predictions")
S6 = os.path.join(USS, "step6_sandwich_squeeze", "predictions")
S8 = os.path.join(USS, "step8_domain_shift", "predictions")
S1 = os.path.join(USS, "step1_nnunet_baseline", "predictions")
S9 = os.path.join(USS, "step9_boundary_mechanisms")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


k99 = load_mod("k99", os.path.join(S9, "tools", "99k_weight_sweep.py"))
arb = load_mod("arb", os.path.join(USS, "step6_sandwich_squeeze", "tools",
                                   "63_arbitrate.py"))


def load_probs2(path):
    """fg = union.  step4 armE is a 3-class (bg/PZ/TZ) model, everything else
    is already (bg, fg).  Reduce to (bg, fg) so members can be averaged."""
    q = k99.load_probs(path)
    if q.shape[0] > 2:
        q = np.stack([q[0], 1.0 - q[0]], axis=0)
    return q
s57 = load_mod("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary",
                                   "tools", "57_shape_prior_ceiling.py"))

# label, directory, naming ("cid" -> USPxxx.npz, "name" -> <chinese>.npz)
POOL = [
    ("nnunet", os.path.join(S1, "nnunet_2d_fold0_checkpoint_best_nifti"), "cid"),
    ("res_best", os.path.join(S6, "sq2resencw25_checkpoint_best_base_nifti"), "cid"),
    ("res_final", os.path.join(S6, "sq2resencw25_checkpoint_final_base_nifti"), "cid"),
    ("w25_best", os.path.join(S6, "sq2w25_checkpoint_best_base_nifti"), "cid"),
    ("w25s_best", os.path.join(S6, "sq2w25s_checkpoint_best_base_nifti"), "cid"),
    ("sq2_best", os.path.join(S6, "sq2_checkpoint_best_base_nifti"), "cid"),
    ("fpA_best", os.path.join(S4, "fpA_stock_fploss_checkpoint_best_nifti"), "cid"),
    ("fpB_best", os.path.join(S4, "fpB_resenc_fploss_dir_checkpoint_best_nifti"), "cid"),
    ("fpArmC_best", os.path.join(S4, "fp_armC_preenc_checkpoint_best_nifti"), "cid"),
    ("fpArmE_best", os.path.join(S4, "fp_armE_zone3_checkpoint_best_nifti"), "cid"),
    ("fpA_final", os.path.join(S4, "fpA_stock_fploss_checkpoint_final_nifti"), "cid"),
    ("fpB_final", os.path.join(S4, "fpB_resenc_fploss_dir_checkpoint_final_nifti"), "cid"),
    ("fpArmC_final", os.path.join(S4, "fp_armC_preenc_checkpoint_final_nifti"), "cid"),
    ("fpArmE_final", os.path.join(S4, "fp_armE_zone3_checkpoint_final_nifti"), "cid"),
    ("sh_in_domain", os.path.join(S8, "sq2shift_in_domain_base_nifti"), "name"),
    ("sh_jpeg30", os.path.join(S8, "sq2shift_jpeg_30_base_nifti"), "name"),
    ("sh_jpeg15", os.path.join(S8, "sq2shift_jpeg_15_base_nifti"), "name"),
    ("sh_gamma080", os.path.join(S8, "sq2shift_gamma_080_base_nifti"), "name"),
    ("sh_gamma125", os.path.join(S8, "sq2shift_gamma_125_base_nifti"), "name"),
    ("sh_gamma060", os.path.join(S8, "sq2shift_gamma_060_base_nifti"), "name"),
    ("sh_gainP25", os.path.join(S8, "sq2shift_affine_gain_p25_base_nifti"), "name"),
    ("sh_gainM25", os.path.join(S8, "sq2shift_affine_gain_m25_base_nifti"), "name"),
    ("sh_tgcP60", os.path.join(S8, "sq2shift_tgc_p60_base_nifti"), "name"),
    ("sh_tgcM60", os.path.join(S8, "sq2shift_tgc_m60_base_nifti"), "name"),
    ("sh_blur20", os.path.join(S8, "sq2shift_blur_20_base_nifti"), "name"),
    ("sh_speckle15", os.path.join(S8, "sq2shift_speckle_15_base_nifti"), "name"),
]

SEED = {"res_best": 6, "w25_best": 1, "w25s_best": 1, "sq2_best": 1}


def name2cid():
    import csv
    p = os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            out[r["original_filename"]] = r["case_id"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["singles", "greedy"], default="singles")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-cand", type=int, default=28)
    ap.add_argument("--min-bdice", type=float, default=20.0,
                    help="drop members whose own bdice1 is below this")
    a = ap.parse_args()

    n2c = name2cid()
    _t, val = arb.read_split(USS)
    gts = {n: np.asarray(s57.load_gt(n)).astype(bool) for n in val}
    stems = [arb.stem_of(n) for n in val]

    P, labels = {}, []
    for lab, d, naming in POOL:
        paths = [os.path.join(d, (n2c[st + ".png"] if naming == "cid" else st) + ".npz")
                 for st in stems]
        if not all(os.path.exists(p) for p in paths):
            print("  skip %-14s (missing npz)" % lab)
            continue
        P[lab] = [load_probs2(p) for p in paths]
        labels.append(lab)
    print("pool size: %d" % len(labels))

    def mean_bdice(W, idxs):
        tot = 0.0
        for vi in idxs:
            acc = None
            for lab, w in W.items():
                q = P[lab][vi]
                acc = q * w if acc is None else acc + q * w
            pred = s57.largest_cc(acc[1] >= 0.5 * float(sum(W.values())))
            tot += s57.bdice(pred, gts[val[vi]], tol=1)
        return tot / len(idxs)

    idx_all = list(range(len(val)))
    t0 = time.time()

    if a.phase == "singles":
        rows = []
        for lab in labels:
            bd = mean_bdice({lab: 1}, idx_all)
            rows.append((lab, bd))
            print("  %-14s bdice1 %8.3f   (%.0fs)" % (lab, bd, time.time() - t0))
        print("\n%-14s %8s" % ("member", "bdice1"))
        for lab, bd in sorted(rows, key=lambda r: -r[1]):
            print("%-14s %8.3f" % (lab, bd))
        out = os.path.join(S9, "results", "pool_singles_val.json")
        with open(out, "w") as f:
            json.dump([{"label": l, "bdice1": b} for l, b in rows], f, indent=2)
        print("\nwrote %s" % out)
        return 0

    # ---- greedy ----
    rows = []
    for lab in labels:
        rows.append((lab, mean_bdice({lab: 1}, idx_all)))
    singles = dict(rows)
    keep = [lab for lab in labels
            if singles[lab] >= a.min_bdice and lab not in SEED]
    keep = keep[:a.max_cand]
    print("greedy candidates (%d): %s" % (len(keep), keep))

    W = dict(SEED)
    cur = mean_bdice(W, idx_all)
    print("\nseed %s -> bdice1 %.3f" % (W, cur))
    hist = [dict(round=0, added=None, weight=None, bdice1=cur, W=dict(W))]

    for r in range(1, a.rounds + 1):
        best = None
        for lab in keep:
            if lab in W:
                continue
            for w in (1, 2, 4):
                W2 = dict(W)
                W2[lab] = W2.get(lab, 0) + w
                bd = mean_bdice(W2, idx_all)
                if best is None or bd > best[0]:
                    best = (bd, lab, w, W2)
        if best is None or best[0] <= cur + 1e-4:
            print("round %d: no improvement (best %.3f) -> stop" % (r, best[0] if best else cur))
            break
        cur, lab, w, W = best
        print("round %d: + %s x%d -> bdice1 %.3f   [%.0fs]"
              % (r, lab, w, cur, time.time() - t0))
        hist.append(dict(round=r, added=lab, weight=w, bdice1=cur, W=dict(W)))

    out = os.path.join(S9, "results", "pool_greedy_val.json")
    with open(out, "w") as f:
        json.dump({"seed": SEED, "history": hist, "final": W,
                   "final_bdice1": cur, "singles": singles}, f, indent=2)
    print("\nFINAL %s -> bdice1 %.3f" % (W, cur))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

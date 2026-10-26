#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/31_causal_summary.py

Aggregate the causal-control battery on fold 3 into one table:
    gDa  = aligned anchor (same recipe as the historical gD run)
    r1   = --w-distill 0        (no teacher term)
    r2   = --scramble-feats     (spatial-content-free teacher targets)
    r3   = --clean-prob 1.0     (clean-only distillation)
    e5   = UltraFedFM teacher   (teacher-dim 768)
    gD   = historical run (unaligned view seeds, kept for continuity)
    u    = 5-split protocol reference (fold3, final epoch)

Reads: results/l2vit_eval_<tag>.json (+_heldout), runs/<tag>/train.log,
       step6/results/arbitrate_<tag>_full31.json (optional, run 63 with
       --names-file data/fold3_val_names.txt after the masks are exported).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
STEP6 = "/media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze"

TAGS = [
    ("gDa", "l2vit_gDa_fold3", "aligned anchor"),
    ("R1", "l2vit_r1nodist_fold3", "no distillation"),
    ("R2", "l2vit_r2scr_fold3", "scrambled targets"),
    ("R3", "l2vit_r3clean_fold3", "clean-only distill"),
    ("E5", "l2vit_e5ufd_fold3", "UltraFedFM teacher"),
    ("gD", "l2vit_gD_fold3", "historical (unaligned seeds)"),
    ("u", "l2vit_u_fold3_final", "5-split protocol ref"),
]


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def train_log_stats(tag):
    p = os.path.join(ROOT, "runs", tag, "train.log")
    if not os.path.exists(p):
        return None
    best = None
    for line in open(p, encoding="utf-8", errors="ignore"):
        m = re.search(r"sel ep\s+(\d+) val_dice\s+([\d.]+) best\s+([\d.]+) @ep(\d+)", line)
        if m:
            ep = int(m.group(4)); v = float(m.group(3))
            if best is None or v > best[0]:
                best = (v, ep)
    return best


def main():
    print("| arm | what | sel val_dice(best@ep) | in_domain D/bd1 | cB D | blur35 D | worst D | heldout b45/s50/cC | arb raw→free (δ) | contain |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for lbl, tag, note in TAGS:
        ev = jload(os.path.join(RES, "l2vit_eval_%s.json" % tag))
        ho = jload(os.path.join(RES, "l2vit_eval_%s_heldout.json" % tag))
        tr = train_log_stats(tag)
        arb = jload(os.path.join(STEP6, "results", "arbitrate_%s_full31.json" % tag))
        s = (ev or {}).get("summary", {})
        ind = s.get("in_domain", {})
        cb = s.get("compound_B", {})
        b35 = s.get("blur_35", {})
        all_d = [v["dice"] for k, v in s.items()] if s else []
        worst = min(all_d) if all_d else None
        h = (ho or {}).get("summary", {})
        hs = "/".join("%.2f" % h[c]["dice"] for c in ("blur_45", "speckle_50", "compound_C")) if h else "-"
        if arb:
            sm = arb["summary"]["mean"]
            ad = "%.2f→%.2f (%+.2f)" % (sm["raw"], sm["free"], sm["free"] - sm["raw"])
            ct = "%.3f" % sm["contain_lrn"]
        else:
            ad = ct = "-"
        def f(x, fmt="%.2f"):
            return fmt % x if isinstance(x, (int, float)) else "-"
        print("| %s | %s | %s | %s/%s | %s | %s | %s | %s | %s | %s |" % (
            lbl, note,
            ("%.4f@%d" % tr) if tr else "-",
            f(ind.get("dice")), f(ind.get("bdice1")),
            f(cb.get("dice")), f(b35.get("dice")),
            f(worst), hs, ad, ct))
    return 0


if __name__ == "__main__":
    sys.exit(main())

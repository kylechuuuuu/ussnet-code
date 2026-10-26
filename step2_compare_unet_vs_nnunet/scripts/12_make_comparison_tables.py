#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_compare_unet_vs_nnunet/scripts/12_make_comparison_tables.py

Merge the per-method summaries produced by 10_evaluate_predictions.py into one
comparison table (markdown + csv), with

  * mean ± std per metric per method,
  * a Delta column per method relative to --baseline (sign/colour of the Delta is
    resolved with the metric's direction: e.g. lower HD95 is better),
  * a paired Wilcoxon signed-rank test per metric over the images common to all
    methods (each non-baseline method vs the baseline),
  * a per-image Dice table so the failure cases are visible, not hidden in the mean.

    python scripts/12_make_comparison_tables.py \
        --tags unet_73 nnunet_2d_fold0_checkpoint_final our_resunet_boundary \
        --labels "U-Net (baseline)" "nnU-Net 2d (baseline)" "OURS" \
        --baseline unet_73
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STEP2 = os.path.dirname(HERE)

METRICS = [("dice", "Dice / F1 (%)", "higher"),
           ("precision", "Precision (%)", "higher"),
           ("recall", "Recall (%)", "higher"),
           ("iou", "IoU (%)", "higher"),
           ("vol_diff_pct", "Volume diff (%)", "lower"),
           ("hd95", "HD95 (px)", "lower"),
           ("assd", "ASSD (px)", "lower"),
           ("correctness", "Topo correctness (%)", "higher"),
           ("completeness", "Topo completeness (%)", "higher"),
           ("quality", "Topo quality (%)", "higher"),
           ("n_pred_cc", "Pred. components", "lower"),
           ("perimeter_pred", "Pred. perimeter (px)", "lower"),
           ("skel_pred_len", "Pred. skeleton length (px)", "lower"),
           ("d127_recall", "Recall of outer zone (127) (%)", "higher"),
           ("d255_recall", "Recall of inner zone (255) (%)", "higher")]


def load(tag, metrics_dir):
    s = json.load(open(os.path.join(metrics_dir, "%s_summary.json" % tag)))
    rows = list(csv.DictReader(open(os.path.join(metrics_dir, "%s_per_image.csv" % tag))))
    per = {}
    for r in rows:
        per[r["name"]] = {k: float(v) for k, v in r.items()
                          if k not in ("name", "split")}
    return s, per


def fmt_delta(d, direction, digits=2):
    """Signed delta with an arrow that resolves 'which way is better'."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "n/a"
    better = (d > 0) if direction == "higher" else (d < 0)
    if abs(d) < 1e-9:
        arrow = "="
    else:
        arrow = "\u25b2" if better else "\u25bc"
    return "%+.*f %s" % (digits, d, arrow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--metrics-dir", default=os.path.join(STEP2, "metrics"))
    ap.add_argument("--baseline", default=None,
                    help="tag used as the reference for the deltas and the paired test "
                         "(default: the first tag)")
    ap.add_argument("--title", default="Step 2 - comparison of all models, one shared evaluator")
    ap.add_argument("--out-prefix", default="comparison_table",
                    help="output files are <metrics-dir>/<prefix>.md and .csv")
    args = ap.parse_args()
    assert len(args.tags) == len(args.labels)

    sums, pers = {}, {}
    for t in args.tags:
        sums[t], pers[t] = load(t, args.metrics_dir)

    base = args.baseline or args.tags[0]
    assert base in args.tags, "baseline %s is not one of the tags" % base
    base_idx = args.tags.index(base)
    others = [i for i in range(len(args.tags)) if i != base_idx]
    common = sorted(set.intersection(*[set(pers[t]) for t in args.tags]))

    # ---------------- markdown / csv table ----------------
    header = ["Metric"] + ["%s (mean+-std)" % l for l in args.labels] + \
             ["Delta %s vs %s" % (args.labels[i], args.labels[base_idx]) for i in others]
    md = ["# %s" % args.title, "",
          "Baseline for the deltas and the paired test: **%s**. "
          "Validation split of the U-Net baseline (`ultrasound_prostate_73`, fold 0 of "
          "`splits_final.json`), %d images common to all methods." % (args.labels[base_idx], len(common)),
          "",
          "| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    csv_rows = []
    for key, name, direction in METRICS:
        cells, vals = [], []
        for t in args.tags:
            m, s = sums[t].get(key + "_mean"), sums[t].get(key + "_std")
            cells.append("%.2f +- %.2f" % (m, s) if m is not None else "n/a")
            vals.append(m)
        deltas = []
        for i in others:
            if vals[i] is None or vals[base_idx] is None:
                deltas.append("n/a")
            else:
                deltas.append(fmt_delta(vals[i] - vals[base_idx], direction))
        md.append("| %s | %s | %s |" % (name, " | ".join(cells), " | ".join(deltas)))
        csv_rows.append([name] + [("%.4f" % v if v is not None else "") for v in vals] + deltas)

    # ---------------- paired tests against the baseline ----------------
    try:
        from scipy.stats import wilcoxon
        has_scipy = True
    except ImportError:
        has_scipy = False

    if len(args.tags) >= 2 and has_scipy:
        for i in others:
            t_other = args.tags[i]
            md += ["", "## Paired test over the %d common images (%s vs %s, Wilcoxon signed-rank)"
                       % (len(common), args.labels[i], args.labels[base_idx]),
                   "",
                   "| Metric | %s mean | %s mean | p | significant @0.05 |"
                   % (args.labels[i], args.labels[base_idx]),
                   "|---|---|---|---|---|"]
            for key, name, direction in METRICS:
                a = np.array([pers[t_other][n][key] for n in common])
                b = np.array([pers[args.tags[base_idx]][n][key] for n in common])
                ok = ~(np.isnan(a) | np.isnan(b))
                a, b = a[ok], b[ok]
                if len(a) < 5:
                    continue
                try:
                    stat, p = wilcoxon(a, b)
                except Exception:
                    continue
                md.append("| %s | %.2f | %.2f | %.4f | %s |"
                          % (name, a.mean(), b.mean(), p, "yes" if p < 0.05 else "no"))

    # ---------------- per-image Dice ----------------
    md += ["", "## Per-image Dice (all methods, sorted by %s, worst first)" % args.labels[base_idx],
           "",
           "| image | " + " | ".join(args.labels) + " | best |",
           "|---|" + "---|" * (len(args.labels) + 1)]
    for n in sorted(common, key=lambda x: pers[base][x]["dice"]):
        vals = [pers[t][n]["dice"] for t in args.tags]
        best = args.labels[int(np.argmax(vals))]
        md.append("| %s | %s | %s |" % (n, " | ".join("%.2f" % v for v in vals), best))

    out_md = os.path.join(args.metrics_dir, args.out_prefix + ".md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    out_csv = os.path.join(args.metrics_dir, args.out_prefix + ".csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(csv_rows)

    wins = {t: 0 for t in args.tags}
    for n in common:
        vals = [pers[t][n]["dice"] for t in args.tags]
        wins[args.tags[int(np.argmax(vals))]] += 1
    print("wrote", out_md)
    print("wrote", out_csv)
    print("per-image Dice wins:", dict(zip(args.labels, (wins[t] for t in args.tags))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

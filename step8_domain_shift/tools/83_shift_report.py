#!/usr/bin/env python
"""83_shift_report.py -- mechanism 2 report: domain shift, robustness, and the
gap between mean Dice and clinical usability.

Consumes results/shift_metrics.csv (tools/82_clinical_metrics.py) and
results/shift_suite_val.json (tools/80_shift_suite.py) and produces

  * the model x condition robustness table (mean Dice / bdice@1px / usable rate),
  * the degradation curve: Delta Dice against the appearance-shift magnitude
    (z-score residual) with a bootstrap CI on the slope,
  * GT-free risk scores: AUROC for catching a GROSS failure (Dice < 80), and
  * the referral curve: if you refer the highest-risk X% of cases, what Dice /
    usable-rate do you keep, and what fraction of gross failures do you catch?

Writes results/shift_report.json, figures/step8_robustness.png and
step8_report.md.

Usage:
  python -u tools/83_shift_report.py [--metrics ...] [--suite ...] [--out-dir .]
"""
import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCORES = ["unc_mean", "unc_area_px", "conf_margin", "band_px", "contain"]
# +1: a HIGHER value means MORE risk.  -1: a LOWER value means MORE risk
# (a confident boundary margin, and a band that contains the GT, are safety signs)
RISK_SIGN = {"unc_mean": +1.0, "unc_area_px": +1.0, "conf_margin": -1.0,
             "band_px": +1.0, "contain": -1.0}
OUTCOMES = ["unusable", "fail_gross", "fail_boundary"]
# the outcome the referral analysis must catch: a clinically obvious collapse
TARGET = "fail_gross"
LEVELS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


# --------------------------------------------------------------------- stats
def auroc(score, label):
    """Mann-Whitney AUROC of a RISK score for predicting label==1."""
    s = np.asarray(score, float)
    y = np.asarray(label, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):                      # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def risk_coverage(score, dice, usable, fail, levels=LEVELS):
    """Refer the highest-risk cases; describe the retained set at each coverage."""
    s = np.asarray(score, float)
    d = np.asarray(dice, float)
    u = np.asarray(usable, int)
    f = np.asarray(fail, int)
    ok = np.isfinite(s)
    s, d, u, f = s[ok], d[ok], u[ok], f[ok]
    if len(d) == 0:
        return []
    order = np.argsort(s, kind="mergesort")          # safest first
    d, u, f = d[order], u[order], f[order]
    n = len(d)
    out = []
    for lv in levels:
        k = max(1, int(round(lv * n)))
        keep, ref = slice(0, k), slice(k, n)
        n_ref = n - k
        caught = int(f[ref].sum())
        total = int(f.sum())
        out.append({"coverage": float(lv), "n": int(k),
                    "mean_dice": float(d[keep].mean()),
                    "usable_rate": float(u[keep].mean()),
                    "min_dice": float(d[keep].min()),
                    "n_referred": int(n_ref),
                    "referral_rate": float(n_ref / n),
                    "referred_mean_dice": float(d[ref].mean()) if n_ref else float("nan"),
                    "fail_caught": caught, "fail_total": total,
                    "fail_recall": float(caught / total) if total else float("nan")})
    return out


def slope_ci(x, y, nboot=4000, seed=0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    b = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        idx = rng.integers(0, len(x), len(x))
        xx, yy = x[idx], y[idx]
        if np.ptp(xx) <= 0:
            continue
        bs.append(float(np.polyfit(xx, yy, 1)[0]))
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return b, float(lo), float(hi)


def risk_of(rows, s):
    return [RISK_SIGN.get(s, 1.0) * r[s] for r in rows]


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=os.path.join(ROOT, "results", "shift_metrics.csv"))
    ap.add_argument("--suite", default=os.path.join(ROOT, "results", "shift_suite_val.json"))
    ap.add_argument("--out-dir", default=ROOT)
    ap.add_argument("--metric", default="dice")
    args = ap.parse_args()

    with open(args.metrics) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("no rows in %s" % args.metrics)

    for r in rows:
        for k, v in list(r.items()):
            if k in ("model", "head", "condition", "name", "split"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = float("nan")
        r["usable"] = int(float(r.get("usable", 0) or 0) >= 0.5)
        r["unusable"] = 1 - r["usable"]
        r["fail_gross"] = int(float(r.get("fail_gross", 0) or 0) >= 0.5)
        r["fail_boundary"] = int(float(r.get("fail_boundary", 0) or 0) >= 0.5)
        # a deliberately mis-biased sandwich head is a different "model" from
        # the base head; keep them apart instead of averaging them together
        h = (r.get("head") or "-").strip()
        if h not in ("", "-"):
            r["model"] = "%s:%s" % (r["model"], h)

    zres = {}
    if os.path.exists(args.suite):
        for c, d in json.load(open(args.suite)).get("conditions", {}).items():
            zres[c] = float(d.get("zscore_residual_px", float("nan")))

    models = sorted({r["model"] for r in rows})
    conds = sorted({r["condition"] for r in rows},
                   key=lambda c: (zres.get(c, float("inf")), c))

    # ---------------------------------------------------- 1. robustness table
    table = {}
    for m in models:
        for c in conds:
            sub = [r for r in rows if r["model"] == m and r["condition"] == c]
            if not sub:
                continue
            table.setdefault(m, {})[c] = {
                "n": len(sub),
                "dice": float(np.mean([r[args.metric] for r in sub])),
                "dice_std": float(np.std([r[args.metric] for r in sub])),
                "bdice1": float(np.nanmean([r["bdice1"] for r in sub])),
                "hd95": float(np.nanmean([r["hd95"] for r in sub])),
                "usable_rate": float(np.mean([r["usable"] for r in sub])),
                "fail_gross_rate": float(np.mean([r["fail_gross"] for r in sub])),
                "z_resid": zres.get(c, float("nan")),
            }

    print("\n[83] mean %s / mean bdice@1px / usable-rate   (sorted by appearance-shift magnitude)"
          % args.metric)
    print("%-18s %6s" % ("condition", "z-res") + "".join("%26s" % m for m in models))
    for c in conds:
        line = "%-18s %6.3f" % (c, zres.get(c, float("nan")))
        for m in models:
            e = table.get(m, {}).get(c)
            line += ("%26s" % ("%.2f/%.1f/%3.0f%%" % (e["dice"], e["bdice1"], 100 * e["usable_rate"]))
                     if e else "%26s" % "-")
        print(line)

    # ------------------------------------------------- 2. degradation vs shift
    deg = {}
    for m in models:
        cs = [c for c in conds if c in table.get(m, {})]
        xs = [table[m][c]["z_resid"] for c in cs]
        base = table[m].get("in_domain", {}).get("dice", float("nan"))
        ydrop = [table[m][c]["dice"] - base for c in cs]
        b, lo, hi = slope_ci(xs, ydrop)
        deg[m] = {"in_domain_dice": base,
                  "worst_condition": min(table[m], key=lambda c: table[m][c]["dice"]),
                  "worst_dice": min(table[m][c]["dice"] for c in table[m]),
                  "slope_dice_per_unit_z": b, "slope_ci": [lo, hi],
                  "xs": xs, "ydrop": ydrop}
    print("\n[83] degradation vs appearance-shift magnitude")
    for m in models:
        d = deg[m]
        print("  %-24s in-domain %.2f  worst %.2f (%s)  slope %+.2f Dice / unit z  [%+.2f, %+.2f]"
              % (m, d["in_domain_dice"], d["worst_dice"], d["worst_condition"],
                 d["slope_dice_per_unit_z"], d["slope_ci"][0], d["slope_ci"][1]))

    # --------------------------------------------- 3. AUROC of GT-free scores
    auroc_tab = {}
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        auroc_tab[m] = {}
        for oc in OUTCOMES:
            y = [r[oc] for r in sub]
            if sum(y) == 0:
                continue
            auroc_tab[m][oc] = {}
            for s in SCORES:
                if s not in sub[0]:
                    continue
                v = risk_of(sub, s)
                if int(np.isfinite(np.asarray(v, float)).sum()) < 8:
                    continue
                auroc_tab[m][oc][s] = auroc(v, y)
    print("\n[83] AUROC (risk score -> outcome), pooled over conditions")
    print("  %-22s %-14s %s" % ("model", "outcome", "  ".join("%s" % s for s in SCORES)))
    for m in models:
        for oc in OUTCOMES:
            if oc not in auroc_tab.get(m, {}):
                continue
            cells = []
            for s in SCORES:
                v = auroc_tab[m][oc].get(s)
                cells.append("%s=%.3f" % (s, v) if v is not None else "%s=-" % s)
            print("  %-22s %-14s %s" % (m, oc, "  ".join(cells)))

    # ------------------------------------------- 4. risk-coverage / referral
    rc = {}
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        rc[m] = {}
        for s in SCORES:
            if s not in sub[0]:
                continue
            v = risk_of(sub, s)
            if int(np.isfinite(np.asarray(v, float)).sum()) < 8:
                continue
            rc[m][s] = risk_coverage(v, [r[args.metric] for r in sub],
                                     [r["usable"] for r in sub],
                                     [r[TARGET] for r in sub])
    print("\n[83] referral curve -- keep the safest X%% (score = unc_area_px);"
          " the outcome to catch is a GROSS failure (%s)" % TARGET)
    print("  %-22s %6s %7s %11s %11s %10s %14s"
          % ("model", "keep", "n", "mean dice", "usable rate", "referred", "gross caught"))
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        base = float(np.mean([r[args.metric] for r in sub]))
        bu = float(np.mean([r["usable"] for r in sub]))
        nf = int(sum(r[TARGET] for r in sub))
        print("  %-22s %6s %7d %11.2f %10.1f%% %10s %14s"
              % (m, "all", len(sub), base, 100 * bu, "-", "0/%d" % nf))
        for e in rc[m].get("unc_area_px", []):
            print("  %-22s %5.0f%% %7d %11.2f %10.1f%% %10d %8d/%d (%.0f%%)"
                  % (m, 100 * e["coverage"], e["n"], e["mean_dice"], 100 * e["usable_rate"],
                     e["n_referred"], e["fail_caught"], e["fail_total"], 100 * e["fail_recall"]))

    # ---------------------------------------------------------------- figures
    figdir = os.path.join(args.out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax = axes[0][0]
    w = 0.8 / max(1, len(models))
    xs = np.arange(len(conds))
    for i, m in enumerate(models):
        ax.bar(xs + i * w, [table.get(m, {}).get(c, {}).get("dice", np.nan) for c in conds],
               w, label=m)
    ax.set_xticks(xs + 0.4 - w / 2)
    ax.set_xticklabels(conds, rotation=90, fontsize=7)
    ax.set_ylabel("mean Dice")
    ax.set_title("Robustness under acquisition shift")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0][1]
    for m in models:
        ax.scatter(deg[m]["xs"], deg[m]["ydrop"], s=22, label=m)
        b = deg[m]["slope_dice_per_unit_z"]
        if np.isfinite(b) and deg[m]["xs"]:
            xr = np.linspace(0, max(deg[m]["xs"]), 20)
            ax.plot(xr, b * xr, "--", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("appearance-shift magnitude (z-score residual, px)")
    ax.set_ylabel("Delta Dice vs in-domain")
    ax.set_title("Degradation curve")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    for m in models:
        e = rc[m].get("unc_area_px")
        if not e:
            continue
        cov = [1.0] + [x["coverage"] for x in e]
        rec = [0.0] + [x["fail_recall"] for x in e]
        ax.plot(cov, rec, "o-", label=m)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("fraction of cases retained (safest first)")
    ax.set_ylabel("fraction of gross failures caught")
    ax.set_title("Referral curve (score = ambiguous-pixel count)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    labels, vals = [], []
    for m in models:
        for oc in OUTCOMES:
            v = auroc_tab.get(m, {}).get(oc, {}).get("unc_area_px")
            if v is not None and np.isfinite(v):
                labels.append("%s\n%s" % (m, oc))
                vals.append(v)
    ax.bar(range(len(vals)), vals, color="#4477aa")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("AUROC (ambiguous-pixel count)")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("GT-free detection of a bad case")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    f1 = os.path.join(figdir, "step8_robustness.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print("\n[83] wrote %s" % f1)

    # ------------------------------------------------------------- json + md
    rep = {"metric": args.metric, "target": TARGET, "models": models, "conditions": conds,
           "z_residual": zres, "table": table, "degradation": deg,
           "auroc": auroc_tab, "risk_coverage": rc, "risk_sign": RISK_SIGN,
           "thresholds": {"usable_dice": 90.0, "usable_bdice1": 20.0,
                          "usable_hd95": 10.0, "gross_dice": 80.0}}
    jf = os.path.join(args.out_dir, "results", "shift_report.json")
    with open(jf, "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print("[83] wrote %s" % jf)

    L = ["# Step 8 - domain shift, robustness and clinical usability", "",
         "Generated by `tools/83_shift_report.py` from `results/shift_metrics.csv`.", "",
         "## 1. Robustness table (mean %s / mean bdice@1px / usable-rate)" % args.metric, "",
         "| condition | z-residual | " + " | ".join(models) + " |",
         "|---|---|" + "---|" * len(models)]
    for c in conds:
        cells = []
        for m in models:
            e = table.get(m, {}).get(c)
            cells.append("%.2f / %.1f / %.0f%%" % (e["dice"], e["bdice1"], 100 * e["usable_rate"])
                         if e else "-")
        L.append("| %s | %.3f | %s |" % (c, zres.get(c, float("nan")), " | ".join(cells)))
    L += ["", "## 2. Degradation against appearance-shift magnitude", "",
          "| model | in-domain Dice | worst condition | worst Dice | slope (Dice / unit z) | 95% CI |",
          "|---|---|---|---|---|---|"]
    for m in models:
        d = deg[m]
        L.append("| %s | %.2f | %s | %.2f | %+.2f | [%+.2f, %+.2f] |"
                 % (m, d["in_domain_dice"], d["worst_condition"], d["worst_dice"],
                    d["slope_dice_per_unit_z"], d["slope_ci"][0], d["slope_ci"][1]))
    L += ["", "## 3. GT-free risk scores (AUROC, higher = better risk ranking)", "",
          "| model | outcome | " + " | ".join(SCORES) + " |",
          "|---|---|" + "---|" * len(SCORES)]
    for m in models:
        for oc in OUTCOMES:
            if oc not in auroc_tab.get(m, {}):
                continue
            cells = ["%.3f" % auroc_tab[m][oc][s] if s in auroc_tab[m][oc] else "-" for s in SCORES]
            L.append("| %s | %s | %s |" % (m, oc, " | ".join(cells)))
    L += ["", "## 4. Referral curve (keep the safest X% by `unc_area_px`)", "",
          "The outcome to catch is a **gross failure** (`%s`, i.e. Dice < 80). The strict" % TARGET,
          "`usable` label (Dice >= 90, bdice@1px >= 20, HD95 <= 10) is met by very few",
          "in-domain cases, so it is reported as a rate but is not the referral target.", "",
          "| model | keep | n | mean %s | usable rate | referred | gross failures caught |" % args.metric,
          "|---|---|---|---|---|---|---|"]
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        nf = int(sum(r[TARGET] for r in sub))
        L.append("| %s | all | %d | %.2f | %.1f%% | - | 0/%d |"
                 % (m, len(sub), np.mean([r[args.metric] for r in sub]),
                    100 * np.mean([r["usable"] for r in sub]), nf))
        for e in rc[m].get("unc_area_px", []):
            L.append("| %s | %.0f%% | %d | %.2f | %.1f%% | %d | %d/%d (%.0f%%) |"
                     % (m, 100 * e["coverage"], e["n"], e["mean_dice"], 100 * e["usable_rate"],
                        e["n_referred"], e["fail_caught"], e["fail_total"], 100 * e["fail_recall"]))
    L += ["", "![robustness](figures/step8_robustness.png)", ""]
    mf = os.path.join(args.out_dir, "step8_report.md")
    with open(mf, "w") as f:
        f.write("\n".join(L) + "\n")
    print("[83] wrote %s" % mf)


if __name__ == "__main__":
    main()

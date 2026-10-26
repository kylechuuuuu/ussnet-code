"""Compare the delivered model against nnU-Net (same post-processing): quantitative + qualitative (v2)."""
import csv, importlib.util, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
S2 = os.path.join(USS, "step2_compare_unet_vs_nnunet")
IMG = os.path.join(USS, "UltrasoundProstate", "images")
NN = os.path.join(S6, "predictions", "nnunet_best_med3open2fill_png")
OU = os.path.join(S6, "predictions", "final_tta_art_png")
FIG = os.path.join(S9, "figures")


def lm(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s57 = lm("s57", os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools",
                             "57_shape_prior_ceiling.py"))

n2c = {}
for r in csv.DictReader(open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv"))):
    n2c[r["original_filename"]] = r["case_id"]


def read_csv(tag):
    out = {}
    for r in csv.DictReader(open(os.path.join(S2, "metrics", tag + "_per_image.csv"))):
        if r.get("split", "val") != "val":
            continue
        out[r["name"]] = r
    return out


cn = read_csv("nnunet_best_med3open2fill")
co = read_csv("final_tta_art_canon")
names = [n for n in co if n in cn]
assert len(names) == 31


def g(r, k):
    return float(r[k])


d_dice = np.array([g(co[n], "dice") - g(cn[n], "dice") for n in names])
d_hd = np.array([g(cn[n], "hd95") - g(co[n], "hd95") for n in names])
d_assd = np.array([g(cn[n], "assd") - g(co[n], "assd") for n in names])

bd_nn, bd_ou = {}, {}
for n in names:
    gt = s57.load_gt(n)
    bd_nn[n] = s57.bdice(np.asarray(Image.open(os.path.join(NN, n)).convert("L")) > 127, gt, tol=1)
    bd_ou[n] = s57.bdice(np.asarray(Image.open(os.path.join(OU, n)).convert("L")) > 127, gt, tol=1)
d_bd = np.array([bd_ou[n] - bd_nn[n] for n in names])

rng = np.random.default_rng(0)


def ci(d):
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def signp(d):
    from math import comb
    w = int((d > 0).sum()); l = int((d < 0).sum()); N = w + l; k = min(w, l)
    return sum(comb(N, i) for i in range(0, k + 1)) * 2.0 / (2 ** N), w, l


labels = [n2c.get(n, n.replace(".png", "")) for n in names]

# ---------------- quantitative ----------------
order = np.argsort(d_dice)
fig = plt.figure(figsize=(22, 8))
ax1 = fig.add_axes([0.045, 0.28, 0.26, 0.62])
ax2 = fig.add_axes([0.345, 0.28, 0.26, 0.62])
ax3 = fig.add_axes([0.635, 0.02, 0.36, 0.96]); ax3.axis("off")
for ax, (vr_nn, vr_ou, lbl), ttl in [
        (ax1, (np.array([g(cn[names[i]], "dice") for i in order]),
               np.array([g(co[names[i]], "dice") for i in order]), "Dice"), "per-case Dice (sorted by gain)"),
        (ax2, (np.array([bd_nn[names[i]] for i in order]),
               np.array([bd_ou[names[i]] for i in order]), "bdice@1px"), "per-case boundary Dice @1px (sorted by Dice gain)")]:
    x = np.arange(len(order))
    ax.vlines(x, vr_nn, vr_ou, color=np.where(vr_ou >= vr_nn, "#1f9e8f", "#c0392b"), lw=2.4, alpha=.85)
    ax.plot(x, vr_nn, "o", color="#7f8c8d", ms=5.5, label="nnU-Net + same postproc")
    ax.plot(x, vr_ou, "o", color="#12b3a0", ms=5.5, label="ours (final_tta_art)")
    ax.set_title(ttl, fontsize=12)
    ax.set_ylabel(lbl, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=8.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9.5, loc="lower right" if lbl == "Dice" else "upper left")
fig.text(0.045, 0.955, "delivered final_tta_art  vs  nnU-Net (both with the SAME post-processing)   -   fold-0 val, n=31",
         fontsize=14, weight="bold")

txt = ["summary (ours - nnU-Net, per-case paired)", ""]
for lbl, d, unit in [("Dice", d_dice, ""), ("bdice@1px", d_bd, ""),
                     ("HD95", d_hd, " px"), ("ASSD", d_assd, " px")]:
    pv, w, l = signp(d)
    lo, hi = ci(d)
    txt.append("%-10s %+7.3f%-3s  CI95 [%+6.3f,%+6.3f]   %2d W / %2d L   p=%.4f"
               % (lbl, d.mean(), unit, lo, hi, w, l, pv))
txt += ["", "mean absolute values (val 31)",
        "  raw nnU-Net (no pp) : Dice 96.106   bdice 35.230   HD95 30.948   ASSD 5.605",
        "  nnU-Net + postproc  : Dice 96.121   bdice 35.356   HD95 28.843   ASSD 5.420",
        "  step6 single (res)  : Dice 96.377   bdice 35.330   HD95 30.202   ASSD 5.320",
        "  ours final_tta_art  : Dice 96.434   bdice %6.3f   HD95 25.426   ASSD 4.943"
        % np.mean([bd_ou[n] for n in names]),
        "", "reading the plot: each vertical stick is one patient;",
        "the gray dot is nnU-Net, the teal dot is ours; a teal stick",
        "above the gray dot means we win that case.",
        "", "what this licenses: Dice +0.313 with 22:9 wins (p=0.029) is",
        "the only per-case robust claim; bdice/HD95/ASSD deltas have",
        "bigger means but the per-case spread dominates (see W/L).",
        "fold 0 only, and the recipe was chosen while looking at val."]
ax3.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=12.5, family="monospace")
o1 = os.path.join(FIG, "compare_nnunet_quant.png")
fig.savefig(o1, dpi=110); print("wrote", o1)

# ---------------- qualitative ----------------
jj = np.argsort(d_dice)
sel = [int(jj[-1]), int(jj[-2]), int(jj[len(jj) // 2]), int(jj[len(jj) // 2 - 1]), int(jj[1]), int(jj[0])]
fig, axes = plt.subplots(2, 3, figsize=(21, 13.5))
for ax, i in zip(axes.ravel(), sel):
    n = names[i]
    img = np.asarray(Image.open(os.path.join(IMG, n)).convert("L"))
    gt = s57.load_gt(n)
    mn = np.asarray(Image.open(os.path.join(NN, n)).convert("L")) > 127
    mo = np.asarray(Image.open(os.path.join(OU, n)).convert("L")) > 127
    if mn.shape != img.shape:      # pad/crop model masks to the image frame
        def fit(m):
            out = np.zeros_like(img, dtype=bool)
            h = min(m.shape[0], img.shape[0]); w = min(m.shape[1], img.shape[1])
            out[:h, :w] = m[:h, :w]
            return out
        mn, mo = fit(mn), fit(mo)
    ax.imshow(img, cmap="gray")
    ax.contour(mn.astype(float), levels=[0.5], colors=["#ff5252"], linewidths=1.4)
    ax.contour(mo.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.4)
    ax.contour(gt.astype(float), levels=[0.5], colors=["#00ff6a"], linewidths=2.4,
               linestyles="dashed")
    ax.set_title("%s\nDice  nnUNet %.2f -> ours %.2f (%+0.2f)      bdice  %.1f -> %.1f (%+0.1f)"
                 % (labels[i], g(cn[n], "dice"), g(co[n], "dice"), g(co[n], "dice") - g(cn[n], "dice"),
                    bd_nn[n], bd_ou[n], d_bd[i]), fontsize=12.5, pad=10)
    ax.axis("off")
handles = [plt.Line2D([], [], color=c, lw=2.6, ls=ls, label=l) for c, l, ls in
           [("#00ff6a", "ground truth", "--"), ("#ff5252", "nnU-Net + same post-processing", "-"),
            ("#00e5ff", "ours (final_tta_art)", "-")]]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14, frameon=False)
fig.suptitle("top-2 / median-2 / bottom-2 patients, ordered by per-case Dice gain", fontsize=15)
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.06, hspace=0.28, wspace=0.05)
o2 = os.path.join(FIG, "compare_nnunet_qual.png")
fig.savefig(o2, dpi=100); print("wrote", o2)

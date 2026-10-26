"""Redraw the 16-case montage: GT as a translucent fill, model contours on top (never hidden)."""
import csv, importlib.util, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

USS = "/media/72TB_disk/personal/zk/USSnet"
S6 = os.path.join(USS, "step6_sandwich_squeeze")
S9 = os.path.join(USS, "step9_boundary_mechanisms")
IMG = os.path.join(USS, "UltrasoundProstate", "images")
NN = os.path.join(S6, "predictions", "nnunet_best_med3open2fill_png")
OU = os.path.join(S6, "predictions", "final_tta_art_png")

s = importlib.util.spec_from_file_location("s57", os.path.join(
    USS, "step4_our_new_model_resunet_boundary", "tools", "57_shape_prior_ceiling.py"))
s57 = importlib.util.module_from_spec(s); s.loader.exec_module(s57)

stats = {r["name"]: r for r in json.load(open(os.path.join(S9, "results", "compare_more_val.json")))}
n2c = {r["original_filename"]: r["case_id"] for r in
       csv.DictReader(open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")))}


def fit(m, shape):
    if m.shape == shape:
        return m
    out = np.zeros(shape, dtype=bool)
    h, w = min(m.shape[0], shape[0]), min(m.shape[1], shape[1])
    out[:h, :w] = m[:h, :w]
    return out


items = []
for n, r in sorted(stats.items(), key=lambda kv: kv[1]["dd"]):
    fn = n if n.endswith(".png") else n + ".png"
    img = np.asarray(Image.open(os.path.join(IMG, fn)).convert("L"))
    items.append((r, img, fit(s57.load_gt(fn), img.shape),
                  fit(np.asarray(Image.open(os.path.join(NN, fn)).convert("L")) > 127, img.shape),
                  fit(np.asarray(Image.open(os.path.join(OU, fn)).convert("L")) > 127, img.shape)))
print("loaded", len(items))

sel = sorted(set(np.linspace(0, len(items) - 1, 16).round().astype(int).tolist()))
fig, axes = plt.subplots(4, 4, figsize=(19, 20))
for ax, i in zip(axes.ravel(), sel):
    r, img, gt, mn, mo = items[i]
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(~gt, gt), cmap="spring", alpha=.22, vmin=0, vmax=1)
    ax.contour(gt.astype(float), [0.5], colors=["#005f28"], linewidths=1.6, linestyles="dashed")
    ax.contour(mn.astype(float), [0.5], colors=["#ff3b30"], linewidths=1.9)
    ax.contour(mo.astype(float), [0.5], colors=["#00d5ff"], linewidths=1.9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999"); sp.set_linewidth(.6)
    ax.set_title("%s   Dice %+.2f   bdice %+.1f   HD95 %+.1f px" % (r["cid"], r["dd"], r["db"], r["dh"]),
                 fontsize=11, pad=5)
handles = [plt.Line2D([], [], color=c, lw=2.8, ls=ls, label=l) for c, l, ls in
           [("#005f28", "ground truth (fill + dashed outline)", "--"),
            ("#ff3b30", "nnU-Net + same post-processing", "-"),
            ("#00d5ff", "ours (final_tta_art)", "-")]]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14, frameon=False)
fig.suptitle("16 of 31 val patients, evenly spread over the per-case Dice-gain ranking\n"
             "(reading order = worst for us -> best for us)", fontsize=15)
fig.subplots_adjust(left=.02, right=.98, top=.94, bottom=.05, hspace=.10, wspace=.03)
o = os.path.join(S9, "figures", "compare_nnunet_more16.png")
fig.savefig(o, dpi=95); print("wrote", o)

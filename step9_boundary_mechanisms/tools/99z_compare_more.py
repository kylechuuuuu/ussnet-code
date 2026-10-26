"""More examples: 16-case montage spread over the Dice-gain range + zoomed anterior-notch detail."""
import csv, importlib.util, json, os
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

s = importlib.util.spec_from_file_location("s57", os.path.join(
    USS, "step4_our_new_model_resunet_boundary", "tools", "57_shape_prior_ceiling.py"))
s57 = importlib.util.module_from_spec(s); s.loader.exec_module(s57)

n2c = {r["original_filename"]: r["case_id"] for r in
       csv.DictReader(open(os.path.join(USS, "step1_nnunet_baseline", "case_id_mapping.csv")))}


def read_csv(tag):
    return {r["name"]: r for r in csv.DictReader(
        open(os.path.join(S2, "metrics", tag + "_per_image.csv"))) if r.get("split", "val") == "val"}


cn, co = read_csv("nnunet_best_med3open2fill"), read_csv("final_tta_art_canon")
names = sorted([n for n in co if n in cn])
assert len(names) == 31


def fit(m, shape):
    if m.shape == shape:
        return m
    out = np.zeros(shape, dtype=bool)
    h, w = min(m.shape[0], shape[0]), min(m.shape[1], shape[1])
    out[:h, :w] = m[:h, :w]
    return out


rows = []
for n in names:
    img = np.asarray(Image.open(os.path.join(IMG, n)).convert("L"))
    gt = fit(s57.load_gt(n), img.shape)
    mn = fit(np.asarray(Image.open(os.path.join(NN, n)).convert("L")) > 127, img.shape)
    mo = fit(np.asarray(Image.open(os.path.join(OU, n)).convert("L")) > 127, img.shape)
    rows.append(dict(name=n, cid=n2c.get(n, n), img=img, gt=gt, mn=mn, mo=mo,
                     dice_nn=float(cn[n]["dice"]), dice_ou=float(co[n]["dice"]),
                     hd_nn=float(cn[n]["hd95"]), hd_ou=float(co[n]["hd95"]),
                     bd_nn=s57.bdice(mn, gt, tol=1), bd_ou=s57.bdice(mo, gt, tol=1)))
    print("done", n, flush=True)
for r in rows:
    r["dd"] = r["dice_ou"] - r["dice_nn"]
    r["db"] = r["bd_ou"] - r["bd_nn"]
    r["dh"] = r["hd_nn"] - r["hd_ou"]
rows.sort(key=lambda r: r["dd"])
json.dump([{k: v for k, v in r.items() if k not in ("img", "gt", "mn", "mo")} for r in rows],
          open(os.path.join(S9, "results", "compare_more_val.json"), "w"), indent=1)


def draw(ax, r, crop=None):
    img, gt, mn, mo = r["img"], r["gt"], r["mn"], r["mo"]
    if crop is not None:
        y0, y1, x0, x1 = crop
        img, gt, mn, mo = img[y0:y1, x0:x1], gt[y0:y1, x0:x1], mn[y0:y1, x0:x1], mo[y0:y1, x0:x1]
    ax.imshow(img, cmap="gray")
    ax.contour(mn.astype(float), [0.5], colors=["#ff5252"], linewidths=1.5)
    ax.contour(mo.astype(float), [0.5], colors=["#00e5ff"], linewidths=1.5)
    ax.contour(gt.astype(float), [0.5], colors=["#00ff6a"], linewidths=2.6, linestyles="dashed")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#888"); sp.set_linewidth(.6)


# ---------- figure 3: 16 cases evenly spread over the gain ranking ----------
sel = sorted(set(np.linspace(0, len(rows) - 1, 16).round().astype(int).tolist()))
fig, axes = plt.subplots(4, 4, figsize=(19, 20))
for ax, i in zip(axes.ravel(), sel):
    r = rows[i]
    draw(ax, r)
    ax.set_title("%s   Dice %+.2f   bdice %+.1f   HD95 %+.1fpx"
                 % (r["cid"], r["dd"], r["db"], r["dh"]), fontsize=10.5, pad=5)
handles = [plt.Line2D([], [], color=c, lw=2.8, ls=ls, label=l) for c, l, ls in
           [("#00ff6a", "ground truth", "--"), ("#ff5252", "nnU-Net + same post-processing", "-"),
            ("#00e5ff", "ours (final_tta_art)", "-")]]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14, frameon=False)
fig.suptitle("16 of 31 val patients, evenly spread over the per-case Dice-gain ranking "
             "(left to right, top to bottom = worst to best for us)", fontsize=15)
fig.subplots_adjust(left=.02, right=.98, top=.945, bottom=.045, hspace=.10, wspace=.03)
o3 = os.path.join(FIG, "compare_nnunet_more16.png"); fig.savefig(o3, dpi=95); print("wrote", o3)


# ---------- figure 4: zoom on the anterior notch (3 best / 3 worst) ----------
pick = sel[::-1][:3] + sel[:3]          # 3 best (largest gain) + 3 worst
fig, axes = plt.subplots(2, 3, figsize=(19, 12))
for ax, i in zip(axes.ravel(), pick):
    r = rows[i]
    yy, xx = np.where(r["gt"] | r["mo"] | r["mn"])
    y0, y1 = yy.min(), yy.max(); x0, x1 = xx.min(), xx.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    cy = y0 + int(0.22 * h)                       # anterior band
    ch = max(int(0.42 * h), 40)
    cw = max(int(0.75 * w), 40)
    ccx = (x0 + x1) // 2
    crop = (max(cy - ch // 2, 0), min(cy + ch // 2, r["img"].shape[0]),
            max(ccx - cw // 2, 0), min(ccx + cw // 2, r["img"].shape[1]))
    draw(ax, r, crop)
    ax.set_title("%s   Dice %+.2f   bdice %+.1f   HD95 %+.1fpx" % (r["cid"], r["dd"], r["db"], r["dh"]),
                 fontsize=13, pad=8)
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14, frameon=False)
fig.suptitle("zoom on the anterior margin: 3 patients where we help most (top) "
             "and 3 where we hurt most (bottom)", fontsize=15)
fig.subplots_adjust(left=.02, right=.98, top=.92, bottom=.06, hspace=.12, wspace=.03)
o4 = os.path.join(FIG, "compare_nnunet_zoom.png"); fig.savefig(o4, dpi=105); print("wrote", o4)

print("\nranking (worst->best for us):")
for r in rows:
    print("  %-9s Dice %+6.2f  bdice %+6.1f  HD95 %+6.1f  (nnU-Net %.2f/%.1f/%.1f | ours %.2f/%.1f/%.1f)"
          % (r["cid"], r["dd"], r["db"], r["dh"], r["dice_nn"], r["bd_nn"], r["hd_nn"],
             r["dice_ou"], r["bd_ou"], r["hd_ou"]))

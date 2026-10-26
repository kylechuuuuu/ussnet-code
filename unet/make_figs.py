#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build qualitative comparison panels: ultrasound | GT overlay | prediction overlay.

Green = correct (TP), Red = false positive, Blue = false negative.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VAL_IMG = "datasets/ultrasound_prostate_73/val/images"
VAL_MSK = "datasets/ultrasound_prostate_73/val/masks"
PRED = "results/ultrasound_prostate_73"
SRC_MSK = "../UltrasoundProstate/masks"
OUT = "results/figs"


def to_big(im, size=(512, 384)):
    return im.convert("RGB").resize(size, Image.BILINEAR)


def overlay(img, pred, gt):
    """img: RGB PIL. pred/gt: bool arrays same size as img."""
    base = np.array(img.convert("RGB")).astype(np.float32)
    base = base * 0.55 + 90  # dim a bit
    out = base.copy()
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    out[tp] = out[tp] * 0.25 + np.array([0, 255, 0]) * 0.75
    out[fp] = out[fp] * 0.25 + np.array([255, 40, 40]) * 0.75
    out[fn] = out[fn] * 0.25 + np.array([40, 90, 255]) * 0.75
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def contour(img, mask, color, width=2):
    """draw a boundary outline of mask onto img"""
    from scipy import ndimage

    m = mask.astype(np.uint8)
    er = ndimage.binary_erosion(mask, iterations=width)
    edge = mask & ~er
    out = np.array(img.convert("RGB")).copy()
    out[edge] = color
    return Image.fromarray(out)


def panel(name, f1, p, r):
    gi = Image.open(os.path.join(VAL_IMG, name)).convert("RGB")
    gt = np.array(Image.open(os.path.join(VAL_MSK, name)).convert("L")) > 127
    pr = np.array(
        Image.open(os.path.join(PRED, name)).convert("L")
    ) > 127
    # resize all to a common display size
    W, H = 640, 480
    gi_r = gi.resize((W, H), Image.BILINEAR)
    gt_r = np.array(Image.fromarray(gt).resize((W, H), Image.NEAREST))
    pr_r = np.array(Image.fromarray(pr).resize((W, H), Image.NEAREST))

    p1 = gi_r
    p2 = overlay(gi_r, gt_r, gt_r)  # GT alone (green)
    p3 = overlay(gi_r, pr_r, gt_r)  # TP green / FP red / FN blue

    gap = 8
    header = 46
    canvas = Image.new("RGB", (W * 3 + gap * 2, H + header), (18, 18, 18))
    canvas.paste(p1, (0, header))
    canvas.paste(p2, (W + gap, header))
    canvas.paste(p3, ((W + gap) * 2, header))

    d = ImageDraw.Draw(canvas)
    try:
        fpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        f = ImageFont.truetype(fpath, 20)
        fsmall = ImageFont.truetype(fpath, 17)
    except Exception:
        f = ImageFont.load_default()
        fsmall = f

    d.text((8, 4), f"{name}", fill=(255, 255, 255), font=fsmall)
    d.text((8, 24), f"F1 {f1:.2f}  P {p:.2f}  R {r:.2f}", fill=(255, 210, 90), font=fsmall)
    d.text((W + gap + 8, 6), "GT (green)", fill=(120, 255, 120), font=f)
    d.text((W + gap + 8, 26), "", fill=(255, 255, 255), font=f)
    d.text(((W + gap) * 2 + 8, 6), "PRED  vs  GT", fill=(255, 255, 255), font=f)
    d.text(
        ((W + gap) * 2 + 8, 26),
        "green=TP  red=FP  blue=FN",
        fill=(200, 200, 200),
        font=fsmall,
    )
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    # parse metrics from test output if provided
    metrics = {}
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        for line in open(sys.argv[1], encoding="utf-8"):
            if line.startswith("Image:"):
                body = line.split("Image:")[1]
                nm = body.split("|")[0].strip()
                vals = {}
                for part in body.split("|")[1:]:
                    if ":" in part:
                        k, v = part.split(":")
                        vals[k.strip()] = float(v)
                metrics[nm] = vals

    names = sorted(metrics.keys()) if metrics else sorted(
        n for n in os.listdir(PRED) if n.endswith(".png")
    )
    if not names:
        print("no names"); return

    # order by F1 ascending so worst-case is visible first
    if metrics:
        names.sort(key=lambda n: metrics[n].get("F1", 0))

    print("building %d panels" % len(names))
    for n in names:
        m = metrics.get(n, {})
        c = panel(
            n,
            m.get("F1", float("nan")),
            m.get("Precision", float("nan")),
            m.get("Recall", float("nan")),
        )
        c.save(os.path.join(OUT, n.replace(".png", "_cmp.png")))

    # contact sheet of the worst 6 and a random 6
    def sheet(sel, outname, cols=1):
        ims = [Image.open(os.path.join(OUT, n.replace(".png", "_cmp.png"))) for n in sel]
        if not ims:
            return
        w, h = ims[0].size
        rows = len(ims)
        canvas = Image.new("RGB", (w, h * rows), (0, 0, 0))
        for i, im in enumerate(ims):
            canvas.paste(im, (0, i * h))
        canvas.thumbnail((1500, 4000))
        canvas.save(os.path.join(OUT, outname))
        print("saved", outname, canvas.size)

    sheet(names[:4], "worst4.png")
    sheet(names[len(names) // 2 - 2 : len(names) // 2 + 2], "median4.png")
    sheet(names[-4:], "best4.png")


if __name__ == "__main__":
    main()

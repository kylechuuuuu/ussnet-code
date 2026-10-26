#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a seeded 7:3 train/val split of the UltrasoundProstate dataset.

Source : USSnet/UltrasoundProstate/{images,masks}   (102 pairs, PNG, 1024x768)
Target : USSnet/unet/datasets/ultrasound_prostate_73/{train,val}/{images,masks}

Mask binarization follows the convention already used by the existing
`datasets/ultrasound_prostate` split: any non-zero source value (127 or 255,
i.e. the inner+outer prostate contour labels) becomes foreground 255.
"""
import os
import random
import shutil

import numpy as np
from PIL import Image

SEED = 42
TRAIN_RATIO = 0.7
SRC = "../UltrasoundProstate"
DST = "datasets/ultrasound_prostate_73"
IMG_EXT = (".png", ".jpg", ".jpeg")

random.seed(SEED)


def main():
    src_images = os.path.join(SRC, "images")
    src_masks = os.path.join(SRC, "masks")

    names = sorted(
        n for n in os.listdir(src_images) if n.lower().endswith(IMG_EXT)
    )
    # keep only pairs that actually have a mask
    pairs = []
    for n in names:
        stem = os.path.splitext(n)[0]
        mask_name = stem + ".png"
        if os.path.exists(os.path.join(src_masks, mask_name)):
            pairs.append((n, mask_name))
        else:
            print("!! missing mask for", n)

    print("total pairs:", len(pairs))

    random.Random(SEED).shuffle(pairs)

    n_train = int(round(len(pairs) * TRAIN_RATIO))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    print("train: %d   val: %d" % (len(train_pairs), len(val_pairs)))

    if os.path.isdir(DST):
        shutil.rmtree(DST)

    for split, items in (("train", train_pairs), ("val", val_pairs)):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(DST, split, sub), exist_ok=True)

        for img_name, mask_name in items:
            # image: copy as RGB PNG
            im = Image.open(os.path.join(src_images, img_name)).convert("RGB")
            im.save(os.path.join(DST, split, "images", img_name))

            # mask: binarize any non-zero -> 255
            m = np.array(Image.open(os.path.join(src_masks, mask_name)).convert("L"))
            m = np.where(m > 0, 255, 0).astype(np.uint8)
            Image.fromarray(m, mode="L").save(
                os.path.join(DST, split, "masks", mask_name)
            )

    # manifest for reproducibility
    with open(os.path.join(DST, "split_manifest_seed%d.txt" % SEED), "w") as f:
        f.write("seed=%d train_ratio=%s\n" % (SEED, TRAIN_RATIO))
        for split, items in (("train", train_pairs), ("val", val_pairs)):
            f.write("\n[%s] %d\n" % (split, len(items)))
            for img_name, _ in items:
                f.write(img_name + "\n")

    # sanity check
    for split in ("train", "val"):
        ni = len(os.listdir(os.path.join(DST, split, "images")))
        nm = len(os.listdir(os.path.join(DST, split, "masks")))
        assert ni == nm, (split, ni, nm)
        print("%s: %d images / %d masks" % (split, ni, nm))

    ti = set(os.listdir(os.path.join(DST, "train", "images")))
    vi = set(os.listdir(os.path.join(DST, "val", "images")))
    assert not (ti & vi), "train/val overlap!"
    print("no overlap; done ->", DST)


if __name__ == "__main__":
    main()

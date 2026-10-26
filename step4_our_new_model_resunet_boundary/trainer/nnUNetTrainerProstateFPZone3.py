#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/nnUNetTrainerProstateFPZone3.py

Arm E: the SAME network, the SAME fingerprint loss and the SAME 71/31 fold-0
split as arm A -- only the *output parameterisation* changes.

Dataset001 binarises the two source contour labels (127 and 255) into one
foreground class, so a model trained on it never sees the internal zone
boundary.  `Dataset002_UltrasoundProstate3` keeps them apart (1 = source 127,
2 = source 255); its union is byte-identical to Dataset001's foreground
(`tools/72_build_zone3_dataset.py` asserts this for all 102 cases), and both
zones are present in all 102 masks (`tools/71_zone_split_stats.py`), so the
internal boundary is a real annotation and not an artefact.

The union `argmax != 0` is therefore a *differently parameterised* estimate of
exactly the same outer contour:

  * extra information -- the encoder must explain the zonal anatomy too;
  * decorrelated errors -- a 3-way softmax makes different mistakes from a
    binary sigmoid, which is what an ensemble wants.

Nothing else is touched: the fingerprint JSON is the one derived from the
binary outer boundary (the union boundary is the same curve), `fg_prob` in
`fp_boundary_losses.py` maps the 3-class softmax back to a union probability
for the boundary/length/direction terms, and `fg_from_target` already returns
the union because nnU-Net hands the loss integer class labels in a (B,1,X,Y)
tensor.

Launch: `bash scripts/58_train_zone3.sh <gpu> [epochs]`
"""
import os

from nnUNetTrainerProstateFPBoundary import nnUNetTrainerProstateFPBoundary


class nnUNetTrainerProstateFPZone3(nnUNetTrainerProstateFPBoundary):
    ARM_TAG = "E_zone3_fp_loss"
    ARM_DESC = "arm E (3-class zone parameterisation + fingerprint loss)"

    # NOTE: no __init__ and no _build_loss.  Arm E differs from arm A only in the
    # dataset it is pointed at, and the inherited __init__ signature must stay
    # untouched (nnU-Net rebuilds my_init_kwargs from locals()).


if __name__ == "__main__":
    t = nnUNetTrainerProstateFPZone3
    print("%s : %s" % (t.__name__, t.ARM_DESC))
    print("epochs default          : %s" % os.environ.get("STEP4_EPOCHS", 500))
    print("dataset (must be 002)   : %s" % os.environ.get("nnUNet_raw", "(unset)"))

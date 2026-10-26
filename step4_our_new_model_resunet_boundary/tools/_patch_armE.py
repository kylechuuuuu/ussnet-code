#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arm E plumbing: make the fingerprint loss class-count agnostic, give the base
trainer overridable arm labels, and add the 3-class trainer.  Idempotent."""
import io
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(TOOLS)
TR = os.path.join(STEP4, "trainer")


def patch(path, reps):
    src = io.open(path, encoding="utf-8").read()
    orig = src
    for old, new, n in reps:
        if n == 0:
            continue
        if new in src and old not in src:
            continue
        got = src.count(old)
        assert got == n, "%s: expected %d of %r, found %d" % (
            os.path.basename(path), n, old[:60], got)
        src = src.replace(old, new)
    if src != orig:
        io.open(path, "w", encoding="utf-8").write(src)
        print("[armE] patched %s" % os.path.basename(path))
    else:
        print("[armE] %s already up to date" % os.path.basename(path))


# ------------------------------------------------- 1. class-count agnostic FG
FG_HELPER = '''

def fg_prob(logits: torch.Tensor) -> torch.Tensor:
    """Probability of the FOREGROUND, whatever the number of classes.

    For a binary model this is exactly `softmax(logits)[:, 1:2]`.  For a k-class
    model (k > 2) the foreground we ultimately score is the UNION of the
    foreground classes, i.e. `1 - P(background)`; writing `[:, 1:2]` there would
    silently score only the first zone while the boundary/length terms claim to
    describe the gland outline.  Both branches are the same tensor when k == 2.
    """
    p = torch.softmax(logits, dim=1)
    if p.shape[1] == 2:
        return p[:, 1:2]
    return 1.0 - p[:, 0:1]
'''

patch(os.path.join(TR, "fp_boundary_losses.py"), [
    ("\ndef total_variation(prob: torch.Tensor) -> torch.Tensor:",
     FG_HELPER + "\ndef total_variation(prob: torch.Tensor) -> torch.Tensor:", 1),
    ("        prob = torch.softmax(net_output, dim=1)[:, 1:2]",
     "        prob = fg_prob(net_output)", 2),
])

# --------------------------------------------- 2. overridable arm labels
patch(os.path.join(TR, "nnUNetTrainerProstateFPBoundary.py"), [
    ("class nnUNetTrainerProstateFPBoundary(nnUNetTrainer):\n",
     "class nnUNetTrainerProstateFPBoundary(nnUNetTrainer):\n"
     '    # Overridable so that a subclass can keep the log and the\n'
     '    # fp_config_used.json honest without copying on_train_start.\n'
     '    ARM_TAG = "A_stock_arch_fp_loss"\n'
     '    ARM_DESC = "arm A (stock PlainConvUNet)"\n\n', 1),
    ('            "[fp] arm A (stock PlainConvUNet) | epochs %d | network %s | "\n'
     '            "output_folder %s"\n'
     '            % (self.num_epochs, type(self.network).__name__, self.output_folder))',
     '            "[fp] %s | epochs %d | network %s | output_folder %s"\n'
     '            % (self.ARM_DESC, self.num_epochs, type(self.network).__name__,\n'
     '               self.output_folder))', 1),
    ('                json.dump({"arm": "A_stock_arch_fp_loss",',
     '                json.dump({"arm": self.ARM_TAG,\n'
     '                           "arm_label": self.ARM_DESC,', 1),
])

# ------------------------------------------------------- 3. the arm E trainer
ARM_E = '''#!/usr/bin/env python3
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
'''
p = os.path.join(TR, "nnUNetTrainerProstateFPZone3.py")
if not os.path.exists(p):
    io.open(p, "w", encoding="utf-8").write(ARM_E)
    print("[armE] wrote trainer/nnUNetTrainerProstateFPZone3.py")
else:
    print("[armE] trainer/nnUNetTrainerProstateFPZone3.py exists")

# ------------------------------------------ 4. register it in the shim installer
patch(os.path.join(STEP4, "scripts", "53_install_trainer_shim.sh"), [
    ("from nnUNetTrainerProstateFPPreEncR34 import (           # noqa: E402,F401\n"
     "    nnUNetTrainerProstateFPPreEncR34)",
     "from nnUNetTrainerProstateFPPreEncR34 import (           # noqa: E402,F401\n"
     "    nnUNetTrainerProstateFPPreEncR34)\n"
     "from nnUNetTrainerProstateFPZone3 import (               # noqa: E402,F401\n"
     "    nnUNetTrainerProstateFPZone3)", 1),
    ('           "nnUNetTrainerProstateFPPreEncR34"]',
     '           "nnUNetTrainerProstateFPPreEncR34",\n'
     '           "nnUNetTrainerProstateFPZone3"]', 1),
    ('        "nnUNetTrainerProstateFPPreEncR34"]',
     '        "nnUNetTrainerProstateFPPreEncR34",\n'
     '        "nnUNetTrainerProstateFPZone3"]', 1),
])

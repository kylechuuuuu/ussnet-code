#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_sandwich_squeeze/trainer/sandwich_config.py

Configuration of the SANDWICH SQUEEZE model (夹逼定理 v2).

One stock nnU-Net backbone produces three versions of the same segmentation:

    base      (基础版本) : the ordinary segmentation
    cautious  (谨慎版本) : supervised by erode(GT, r_c)   -- the inner bound
    amplified (放大版本) : supervised by dilate(GT, r_a)  -- the outer bound

The true boundary is pinched between them,  cautious ⊆ base ⊆ amplified, and
the FINAL boundary is determined at inference by arbitration inside the
interval (tools/63_arbitrate.py): image evidence may move the base boundary,
but only as far as the interval allows.

Why fixed radii (and not step 5's annealing to 1 px):
    step 5 measured that at r = 1 px the bound targets coincide with GT for
    99.29 % of the pixels -- the bounds carried no information left.  The
    residual contour displacement of every model on this dataset is 1-3 px
    (median ~2.6 px, step 4 section 5.2), so the interval must be at that
    scale or it cannot bound the error.  Default: r_c = 3 px inwards,
    r_a = 4 px outwards (the outward side matters more: 51-53 % of the
    residual error pixels are false negatives within 2 px).

The angular modulation (边界指纹 / boundary fingerprint, step 4) widens both
bounds where the ultrasound edge evidence is weak (sectors +75/+105 deg), so
the arbitration has more legal range exactly where the model is least sure.

Everything has an environment override (STEP6_*) so ablations never need code
edits; the trainer writes the effective configuration into the run folder.
"""
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
STEP6_ROOT = os.path.dirname(_HERE)
STEP4_ROOT = os.path.join(os.path.dirname(STEP6_ROOT), "step4_our_new_model_resunet_boundary")
DEFAULT_FP_JSON = os.path.join(STEP4_ROOT, "data", "boundary_fingerprint.json")


def _env(name, default):
    """Env override that treats UNSET and EMPTY identically (same rule as step 4/5)."""
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val


env_or = _env


class SandwichConfig(object):
    def __init__(self, payload, source, angular_default=True):
        self.source = source
        self.angular_profile = payload.get("angular_profile", {})
        self.shape_statistics = payload.get("shape_statistics", {})

        # ---- the two versions of the squeeze --------------------------------
        self.r_c = float(_env("STEP6_RC", 3.0))       # cautious  = erode(GT, r_c)
        self.r_a = float(_env("STEP6_RA", 4.0))       # amplified = dilate(GT, r_a)
        self.w_c = float(_env("STEP6_WC", 0.5))       # loss weight of the cautious head
        self.w_a = float(_env("STEP6_WA", 0.5))       # loss weight of the amplified head
        self.head_bias = float(_env("STEP6_HEAD_BIAS", 1.0))

        # ---- angular modulation from the boundary fingerprint ---------------
        self.angular = int(_env("STEP6_ANGULAR", 1 if angular_default else 0)) == 1
        inv = [float(x) for x in (self.angular_profile.get("dir_sector_weights") or [])]
        if not inv:
            rel = [float(x) for x in (self.angular_profile.get("edge_strength_rel") or [])]
            if rel:
                mean_rel = sum(rel) / float(len(rel))
                inv = [mean_rel / max(1e-6, r) for r in rel]
        if inv:
            mean_inv = sum(inv) / float(len(inv))
            # clip: the interval never collapses and never explodes
            self.sector_scale = [min(1.5, max(0.5, v / mean_inv)) for v in inv]
        else:
            self.sector_scale = []
        self.sector_deg = list(self.angular_profile.get("sector_deg_centers") or [])
        if self.angular and not self.sector_scale:
            self.angular = False

        # ---- misc ------------------------------------------------------------
        self.rmax = int(_env("STEP6_RMAX", max(8, int(math.ceil(max(self.r_c, self.r_a) * 1.5)) + 2)))
        self.num_epochs = int(_env("STEP6_EPOCHS", 500))
        self.log_every = int(_env("STEP6_LOG_EVERY", 25))
        self.scale_px = float(self.shape_statistics.get("mean_linear_scale_px", 280.6))

    def summary_lines(self):
        lines = [
            "[sw] base      head : stock DC+CE vs GT                            (基础版本)",
            "[sw] cautious  head : erode(GT, %.1f px) %s   (谨慎版本)"
            % (self.r_c, "[angular]" if self.angular else "[isotropic]"),
            "[sw] amplified head : dilate(GT, %.1f px) %s  (放大版本)"
            % (self.r_a, "[angular]" if self.angular else "[isotropic]"),
            "[sw] nesting enforced at inference: cautious ⊆ base ⊆ amplified",
            "[sw] fingerprint: %s" % self.source,
        ]
        if self.angular and self.sector_scale:
            lines.append("[sw] sector scale (deg -> x radius): %s" % ", ".join(
                "%.0f->%.2f" % (d, s) for d, s in zip(self.sector_deg, self.sector_scale)))
        return lines


def load_sandwich_config(angular_default=True):
    path = _env("STEP6_FP_JSON", DEFAULT_FP_JSON)
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return SandwichConfig(payload, path, angular_default=angular_default)

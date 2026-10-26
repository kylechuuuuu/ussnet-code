#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_our_new_model_resunet_boundary/trainer/fp_config.py

Loads the *boundary/geometry* half of the dataset fingerprint written by
`tools/50_extract_boundary_fingerprint.py` and turns it into the configuration of
the redesigned model.

This is what "using the fingerprint" means for the new model: the loss shape is
not a hand-tuned constant, it is declared by the dataset.

    data/boundary_fingerprint.json
        boundary_contrast   -> is a boundary term justified at all? how sharp?
        shape_statistics    -> how wide must the tolerance band be? is the GT
                               smooth enough to justify a length penalty?
        angular_profile     -> which sectors deserve more weight?
        scale_spread        -> how much does the foreground size vary?

Environment overrides (used for ablation runs, all optional):

    STEP4_FP_JSON        path to the fingerprint json
    STEP4_W_TOL          weight of the tolerant boundary Dice   (default from fp)
    STEP4_W_BCE          weight of the sharp 1 px boundary BCE   (default from fp)
    STEP4_W_PER          weight of the length/perimeter term     (default from fp)
    STEP4_TOL_BAND       odd kernel of the tolerant band         (default from fp)
    STEP4_WARMUP         epochs over which the boundary terms ramp in
    STEP4_USE_DIRECTION  1 -> apply the angular sector weights (default 0)
    STEP4_DISABLE_EXTRA  1 -> ablate everything back to plain nnU-Net loss
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
STEP4_ROOT = os.path.dirname(_HERE)
DEFAULT_JSON = os.path.join(STEP4_ROOT, "data", "boundary_fingerprint.json")


def _env(name, default):
    """Env override that treats UNSET and EMPTY identically.

    scripts/env_step4.sh deliberately exports the ablation knobs as empty
    strings (`export STEP4_W_TOL="${STEP4_W_TOL:-}"`), so the two-argument
    `os.environ.get(name, default)` form would return `` (not `default`) and
    `float("")` raises ValueError.  Restore the intended "empty means unset".
    """
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val


class FPConfig(object):
    """Fingerprint descriptors + the loss parameters derived from them."""

    def __init__(self, payload, source):
        self.source = source
        self.meta = payload.get("meta", {})
        self.boundary_contrast = payload.get("boundary_contrast", {})
        self.shape_statistics = payload.get("shape_statistics", {})
        self.angular_profile = payload.get("angular_profile", {})
        self.scale_spread = payload.get("scale_spread", {})
        d = dict(payload.get("derived_loss_config", {}))

        # --- env overrides (ablation) -------------------------------------
        d["weight_tolerant_bdice"] = float(_env(
            "STEP4_W_TOL", d.get("weight_tolerant_bdice", 0.5)))
        d["weight_boundary_bce"] = float(_env(
            "STEP4_W_BCE", d.get("weight_boundary_bce", 0.2)))
        d["weight_perimeter"] = float(_env(
            "STEP4_W_PER", d.get("weight_perimeter", 0.05)))
        d["tolerant_band"] = int(_env(
            "STEP4_TOL_BAND", d.get("tolerant_band", 7)))
        d["warmup_epochs"] = int(_env(
            "STEP4_WARMUP", d.get("warmup_epochs", 100)))
        self.use_direction = int(_env("STEP4_USE_DIRECTION", 0)) == 1
        self.disabled = int(_env("STEP4_DISABLE_EXTRA", 0)) == 1
        if d["tolerant_band"] % 2 == 0:
            d["tolerant_band"] += 1
        if d.get("bce_band", 3) % 2 == 0:
            d["bce_band"] = 3
        self.loss = d

    # ------------------------------------------------------------------ report
    def summary_lines(self):
        d = self.loss
        bc, ss, sc = self.boundary_contrast, self.shape_statistics, self.scale_spread
        L = []
        L.append("[fp] fingerprint source            : %s" % self.source)
        L.append("[fp]   (computed on fold-0 TRAIN split, %s images)"
                 % self.meta.get("n_train", "?"))
        L.append("[fp] boundary_contrast  |gradI| boundary/image = %.2fx, "
                 "edge support p75 = %.1f %%, p90 = %.1f %%"
                 % (bc.get("grad_ratio_mean", 0), 100 * bc.get("edge_support_p75_mean", 0),
                    100 * bc.get("edge_support_p90_mean", 0)))
        L.append("[fp] shape_statistics   perimeter/sqrt(area) = %.3f, "
                 "opening-invariance r = %.0f px, mean linear scale = %.1f px"
                 % (ss.get("perimeter_over_sqrt_area_mean", 0),
                    ss.get("opening_invariance_radius_median", 0),
                    ss.get("mean_linear_scale_px", 0)))
        L.append("[fp] angular_profile    weakest sector(s) = %s"
                 % self.weakest_sectors())
        L.append("[fp] scale_spread       area CV = %.3f, fg fraction %.4f..%.4f"
                 % (sc.get("area_cv", 0), sc.get("fg_frac_min", 0), sc.get("fg_frac_max", 0)))
        L.append("[fp] => derived loss config:")
        L.append("[fp]      tolerant band  = %d  (+- %d px)   weight %.3f"
                 % (d["tolerant_band"], d.get("tolerance_px", d["tolerant_band"] // 2),
                    d["weight_tolerant_bdice"]))
        L.append("[fp]      sharp band     = %d  beta %.1f          weight %.4f"
                 % (d.get("bce_band", 3), d.get("bce_beta", 4.0), d["weight_boundary_bce"]))
        L.append("[fp]      length penalty                          weight %.4f"
                 % d["weight_perimeter"])
        L.append("[fp]      direction weighting                     %s"
                 % ("ON  sector weights = %s" % d.get("dir_sector_weights")
                    if self.use_direction else "off (this arm)"))
        L.append("[fp]      boundary terms ramp in over %d epochs" % d["warmup_epochs"])
        if self.disabled:
            L.append("[fp]      *** STEP4_DISABLE_EXTRA=1 : extra terms OFF (ablation) ***")
        return L

    def weakest_sectors(self):
        rel = self.angular_profile.get("edge_strength_rel") or []
        cen = self.angular_profile.get("sector_deg_centers") or []
        if not rel or not cen:
            return "n/a"
        pairs = sorted(zip(rel, cen))[:3]
        return ", ".join("%+d deg (rel %.3f)" % (int(c), r) for r, c in pairs)


def load_fp_config(path=None):
    path = path or os.environ.get("STEP4_FP_JSON") or DEFAULT_JSON
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "boundary fingerprint not found: %s\n"
            "run:  python tools/50_extract_boundary_fingerprint.py" % path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return FPConfig(payload, path)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_squeeze_boundary/trainer/squeeze_config.py

Configuration of the SQUEEZE-THEOREM (夹逼定理) boundary model.

The model decides the prostate boundary by *pinching* it between two bounds that
are annealed onto the boundary, instead of reading it off a single threshold.
This file holds the parameters of that construction.  Wherever a number can be
derived from the *measured* geometry of this dataset it is taken from the
boundary/geometry fingerprint that step 4 computed on the fold-0 training split
(shared file, single source of truth, still no validation data involved):

    .../step4_our_new_model_resunet_boundary/data/boundary_fingerprint.json
        shape_statistics.mean_linear_scale_px  = 280.6 px   -> band width
        shape_statistics.opening_invariance... = 30 px      -> band is safe
        angular_profile.edge_strength_rel      = weak at +75/+105 deg -> the
                                                 angular band width (STEP5_ANGULAR)
        boundary_contrast.grad_ratio_mean      = 3.47x      -> the boundary is a
                                                 real edge, so pinching it is
                                                 meaningful at all

THE SQUEEZE PARAMETERS
----------------------
R0        initial half-width of the interval (px).  6 px = 2 % of the 280.6 px
          linear scale, and ~2x the measured 1-3 px residual contour
          displacement of both baselines -> a bound that starts OUTSIDE the error.
R_min     final half-width (px).  1.0 px: the interval closes to the smallest
          scale at which a boundary can still be said to be determined, i.e. the
          distance-transform/erosion limit of the discrete grid.
warmup    epochs the band stays at R0 before the annealing starts.
schedule  geometric: r(t) = R_min * (R0/R_min) ** ((T - t) / (T - warmup))
          -> the sandwich closes exponentially; r(T) = R_min.
lambda_order  weight of the ordering hinge  p_in <= p_main <= p_out
lambda_gap    weight of the squeeze gap (bounds must collapse where the middle
              head is undecided)
w_in, w_out   weight of the two bound heads relative to the middle head
angular       1 -> the band half-width is angularly modulated with the
              fingerprint's edge-strength profile (wider where the ultrasound
              evidence is weaker: +75/+105 deg), 0 -> isotropic band.

Everything can be overridden by environment variables (ablations / reruns); the
trainer writes the *effective* configuration into the run folder as
squeeze_config_used.json.
"""
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
STEP5_ROOT = os.path.dirname(_HERE)
STEP4_ROOT = os.path.join(os.path.dirname(STEP5_ROOT), "step4_our_new_model_resunet_boundary")
DEFAULT_FP_JSON = os.path.join(STEP4_ROOT, "data", "boundary_fingerprint.json")


def _env(name, default):
    """Env override that treats UNSET and EMPTY identically (same rule as step 4).

    scripts/env_step5.sh deliberately exports optional knobs as empty strings, so
    `os.environ.get(name, default)` would return '' and e.g. float('') would raise.
    """
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val


# public alias: the trainer/network modules read optional knobs with this
env_or = _env


class SqueezeConfig(object):
    def __init__(self, payload, source, angular_default=True):
        self.source = source
        self.meta = payload.get("meta", {})
        self.shape_statistics = payload.get("shape_statistics", {})
        self.angular_profile = payload.get("angular_profile", {})
        self.boundary_contrast = payload.get("boundary_contrast", {})

        scale = float(self.shape_statistics.get("mean_linear_scale_px", 280.6))

        # ---- the interval (band) geometry ---------------------------------
        self.R0 = float(_env("STEP5_R0", 6.0))
        self.R_min = float(_env("STEP5_RMIN", 1.0))
        self.warmup = int(_env("STEP5_WARMUP", 5))
        self.schedule = str(_env("STEP5_SCHEDULE", "geometric")).lower()
        if self.R_min > self.R0:
            raise ValueError("STEP5_RMIN (%s) must not exceed STEP5_R0 (%s)" % (self.R_min, self.R0))

        # ---- the squeeze terms -------------------------------------------
        self.lambda_order = float(_env("STEP5_W_ORDER", 0.5))
        self.lambda_gap = float(_env("STEP5_W_GAP", 0.2))
        self.w_in = float(_env("STEP5_W_IN", 0.5))
        self.w_out = float(_env("STEP5_W_OUT", 0.5))

        # ---- angular (fingerprint-driven) band ---------------------------
        # WHERE THE EVIDENCE IS WEAK THE INTERVAL MUST BE ALLOWED TO OPEN:
        # the multiplier is ~ 1 / relative-edge-strength (the fingerprint stores
        # exactly that as `dir_sector_weights`, mean 1, weakest sectors +75/+105
        # deg -> 1.44 / 1.48), clipped so the band never collapses or explodes.
        self.angular = int(_env("STEP5_ANGULAR", 1 if angular_default else 0)) == 1
        inv = [float(x) for x in (self.angular_profile.get("dir_sector_weights") or [])]
        if not inv:
            rel = [float(x) for x in (self.angular_profile.get("edge_strength_rel") or [])]
            if rel:
                mean_rel = sum(rel) / float(len(rel))
                inv = [mean_rel / max(1e-6, r) for r in rel]
        if inv:
            mean_inv = sum(inv) / float(len(inv))
            self.sector_scale = [min(1.5, max(0.5, v / mean_inv)) for v in inv]
        else:
            self.sector_scale = []
        self.sector_deg = list(self.angular_profile.get("sector_deg_centers") or [])
        if self.angular and not self.sector_scale:
            # fingerprint without angular profile -> fall back to isotropic
            self.angular = False

        # ---- misc ---------------------------------------------------------
        self.rmax = int(_env("STEP5_RMAX", max(8, int(math.ceil(self.R0)) + 2)))
        self.head_bias = float(_env("STEP5_HEAD_BIAS", 1.0))
        self.num_epochs = int(_env("STEP5_EPOCHS", 500))
        self.log_every = int(_env("STEP5_LOG_EVERY", 25))
        self.scale_px = scale

    # ------------------------------------------------------------------ maths
    def radius(self, epoch):
        """Band half-width r(t) in original-pixel units at the given epoch.

        This is the annealing half of the squeeze theorem:  it makes the two
        supervision targets erode/dilate(GT, r(t)) converge onto GT itself, so
        the interval [inner, outer] has zero width in the limit and the boundary
        is pinned to a single curve.
        """
        T = max(2, int(self.num_epochs))
        w = max(0, min(self.warmup, T - 1))
        if epoch <= w:
            return float(self.R0)
        if T - 1 <= w:
            return float(self.R_min)
        frac = float(epoch - w) / float(T - 1 - w)          # 0 .. 1
        frac = min(1.0, max(0.0, frac))
        if self.schedule == "step":
            # four plateaus: R0, R0/2, R0/4, R_min
            steps = [self.R0, max(self.R_min, self.R0 / 2.0),
                     max(self.R_min, self.R0 / 4.0), self.R_min]
            idx = min(len(steps) - 1, int(frac * len(steps)))
            return float(steps[idx])
        return float(self.R0 * (self.R_min / self.R0) ** frac)

    def radius_schedule(self, n=None):
        n = int(n or self.num_epochs)
        return [self.radius(e) for e in range(n)]

    # ----------------------------------------------------------------- report
    def summary_lines(self):
        L = []
        L.append("[sq] fingerprint            : %s" % self.source)
        L.append("[sq]   (fold-0 TRAIN split only, n=%s)" % self.meta.get("n_train", "?"))
        L.append("[sq] linear scale           : %.1f px  ->  R0 = %.1f px (%.2f %% of scale)"
                 % (self.scale_px, self.R0, 100.0 * self.R0 / max(1e-6, self.scale_px)))
        L.append("[sq] radius curriculum      : r(t) %s  %.2f -> %.2f px over %d epochs (warmup %d)"
                 % (self.schedule, self.R0, self.R_min, self.num_epochs, self.warmup))
        L.append("[sq] squeeze terms          : order hinge w=%.3f | gap w=%.3f | heads w_in=%.3f w_out=%.3f"
                 % (self.lambda_order, self.lambda_gap, self.w_in, self.w_out))
        if self.angular:
            weak = sorted(zip(self.sector_scale, self.sector_deg))[-3:]
            L.append("[sq] angular band           : ON  (widest where image evidence is weakest: %s)"
                     % ", ".join("%+d deg x%.2f" % (int(d), s) for s, d in reversed(weak)))
            L.append("[sq]   sector multipliers    : %s"
                     % ", ".join("%.2f" % s for s in self.sector_scale))
        else:
            L.append("[sq] angular band           : OFF (isotropic band, ablation arm)")
        L.append("[sq] head init offset       : middle=0, inner=-%.2f, outer=+%.2f (logit, interval starts OPEN)"
                 % (self.head_bias, self.head_bias))
        return L


def load_squeeze_config(path=None, angular_default=True):
    path = path or os.environ.get("STEP5_FP_JSON") or DEFAULT_FP_JSON
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "boundary fingerprint not found: %s\n"
            "(step 5 reuses step 4's fingerprint; run "
            "step4_our_new_model_resunet_boundary/tools/50_extract_boundary_fingerprint.py)" % path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return SqueezeConfig(payload, path, angular_default=angular_default)

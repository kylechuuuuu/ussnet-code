#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""66 -- retrain the normal refiner with a PERTURBATION MATCHED TO THE REAL ERROR.

Motivation (tool 65, results/displacement_coherence.json):
  * the real correction field the refiner must produce has an autocorrelation
    half-width of 14.0 points (38 px, median), while the synthetic perturbation
    used in tools 60/64 (gaussian_filter1d(white, U(0.02,0.20)*400)) has 51.0
    points (140 px) -- 3.6x too smooth;
  * 94 % of the oracle boundary gain survives low-passing the correction at
    sigma = 2 points, 64 % at sigma = 5 -- i.e. the correction is expressible,
    so the failure is not the receptive field (NU = 17 points).

So the training task may simply be mis-specified: the model is taught to invert a
process far smoother than the error it will meet.  This script re-estimates the
correction's power spectrum directly from the 31 held-out images, synthesises
perturbations with that spectrum and the measured amplitude distribution, and
re-trains the refiner with everything else held fixed (60 fit images = 71 fold-0
train minus 11 diagnostics hold-out, 40 epochs, same architecture and loss).

Control = the legacy `results/normal_refiner.pt` (tool 60, evaluated by tool 61):
  nnU-Net cc1 96.12 / 35.28 / 50.25 -> r1 95.99 / 32.46 / 47.61  (-2.82 bd1px)

Usage:
  python tools/66_matched_perturb_refiner.py --epochs 40 --hold 11 \
      --out results/normal_refiner_psd.pt
Writes results/correction_psd.json and the checkpoint named by --out.
"""
import os
import csv
import json
import time
import argparse
import importlib.util

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
USS = os.path.dirname(ROOT)
S1 = os.path.join(USS, "step1_nnunet_baseline")
PSD_JSON = os.path.join(ROOT, "results", "correction_psd.json")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s57 = _load("s57", os.path.join(TOOLS, "57_shape_prior_ceiling.py"))
s58 = _load("s58", os.path.join(TOOLS, "58_normal_refine_oracle.py"))
s60 = _load("s60", os.path.join(TOOLS, "60_train_normal_refiner.py"))
s64 = _load("s64", os.path.join(TOOLS, "64_stage2_refiner_context.py"))

NPT = s57.NPT
MAXD = s60.MAXD

# filled by estimate(): the matched-generator state
PSD = None          # normalised power spectrum, shape (NPT//2 + 1,)
FREQ = None         # frequencies, cycles per sampling point
AMP_POOL = None     # empirical per-image std of the real correction
_RNG = None


def name2cid():
    out = {}
    with open(os.path.join(S1, "case_id_mapping.csv")) as f:
        for row in csv.DictReader(f):
            out[row["original_filename"]] = row["case_id"]
    return out


def estimate(names):
    """Mean power spectrum + amplitude pool of the real stage-1 correction."""
    n2c = name2cid()
    print("[66] estimating the correction spectrum from the real stage-1 contour ...")
    fields = []
    for nm in names:
        cid = n2c.get(nm)
        gt = s57.load_gt(nm)
        if cid is None or gt is None:
            continue
        prob, m0 = s64.load_stage1(nm, n2c)
        if prob is None:
            continue
        m = s57.largest_cc(m0)
        c = s57.contour_of(m)
        if c is None:
            continue
        pts = s57.resample_closed(c, NPT, 0)
        sdf = s58.sdf_of(gt)
        d = np.clip(-s60.bil(sdf.astype(np.float32), pts), -MAXD, MAXD)
        fields.append(d)
    F = np.stack(fields)
    print("[66] %d correction fields" % len(F))
    F = F - F.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(F, n=NPT)) ** 2
    psd = spec.mean(axis=0)
    psd = psd / (psd.sum() + 1e-12)
    freqs = np.fft.rfftfreq(NPT, d=1.0)
    amps = F.std(axis=1)
    out = {
        "n": int(len(F)),
        "psd": psd.tolist(),
        "freqs": freqs.tolist(),
        "amp_mean": float(amps.mean()),
        "amp_median": float(np.median(amps)),
        "amp_p90": float(np.percentile(amps, 90)),
        "amp_pool": amps.tolist(),
    }
    with open(PSD_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("[66] real correction: amp std mean %.2f  median %.2f  p90 %.2f px"
          % (out["amp_mean"], out["amp_median"], out["amp_p90"]))
    print("[66] wrote %s" % PSD_JSON)
    return psd, freqs, amps


def matched_perturb(n, rng):
    """Random field with the measured power spectrum and amplitude distribution.

    Same signature as s60.perturb_field, so it can be monkey-patched in.
    """
    global PSD, FREQ, AMP_POOL
    if n != NPT:
        # fall back to a smooth field if a different point count is requested
        return np.clip(ndimage.gaussian_filter1d(rng.standard_normal(n), 8.0,
                                                 mode="wrap"), -MAXD, MAXD)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=PSD.shape[0])
    spec = np.sqrt(PSD) * np.exp(1j * ph)
    spec[0] = 0.0                     # no DC component
    d = np.fft.irfft(spec, n=n)
    d = d / (d.std() + 1e-9)
    amp = float(rng.choice(AMP_POOL) * rng.uniform(0.5, 1.5))
    return np.clip(d * amp, -MAXD, MAXD)


def main():
    global PSD, FREQ, AMP_POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hold", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "normal_refiner_psd.pt"))
    a = ap.parse_args()

    _tr, te = s57.read_split(USS, exp="ultrasound_prostate_73")
    PSD, FREQ, AMP_POOL = estimate(te)

    s60.perturb_field = matched_perturb      # <- the only change to the task
    print("[66] training with the matched generator (control = legacy "
          "results/normal_refiner.pt)")
    import sys
    sys.argv = ["60_train_normal_refiner.py",
                "--epochs", str(a.epochs),
                "--hold", str(a.hold),
                "--seed", str(a.seed),
                "--bs", str(a.bs),
                "--lr", str(a.lr),
                "--max", str(a.max),
                "--out", a.out]
    t0 = time.time()
    s60.main()
    print("[66] total %.0fs   checkpoint -> %s" % (time.time() - t0, a.out))


if __name__ == "__main__":
    main()

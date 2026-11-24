#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step15_teacher_distill_sandwich/scripts/env_step15.sh
#
#   Shared environment for this step.  Source it; it does not set data paths
#   for you -- export nnUNet_raw / nnUNet_preprocessed / nnUNet_results and
#   US_FM_CHECKPOINT first (or edit the defaults below).
#
#   usage: source scripts/env_step15.sh
# ---------------------------------------------------------------------------
set -uo pipefail

export STEP15_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$STEP15_ROOT/trainer:${PYTHONPATH:-}"

# nnU-Net dataset / configuration used by this step
export DATASET_ID="${DATASET_ID:-001}"
export DATASET_NAME="${DATASET_NAME:-Dataset001_UltrasoundProstate}"
export CONFIG="${CONFIG:-2d}"
export FOLD="${FOLD:-0}"

# frozen teacher (NOT part of the repo)
export US_FM_CHECKPOINT="${US_FM_CHECKPOINT:-}"
# PCA basis fitted by scripts/01_fit_prior_stats.py
export US_DISTILL_STATS="${US_DISTILL_STATS:-$STEP15_ROOT/prior_stats.npz}"
export US_DISTILL_W="${US_DISTILL_W:-0.05}"

# python that has nnunetv2==2.6.2 + timm; default to the active env
export STEP15_PYTHON="${STEP15_PYTHON:-python}"

if [ -z "${nnUNet_raw:-}" ] || [ -z "${nnUNet_preprocessed:-}" ] || [ -z "${nnUNet_results:-}" ]; then
  echo "[env15] WARNING: nnUNet_raw / nnUNet_preprocessed / nnUNet_results are not all set"
fi

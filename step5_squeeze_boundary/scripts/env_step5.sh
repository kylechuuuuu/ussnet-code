#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/env_step5.sh
#
#   Environment for the squeeze-theorem (夹逼定理) boundary model.
#
#       source scripts/env_step5.sh
#
#   Reuses step 1's dataset/preprocessing/fold-0 split (so every number is
#   directly comparable with the nnU-Net baseline and the step-4 arms) and
#   step 4's boundary fingerprint (the measured geometry of THIS dataset), but
#   writes everything it produces into step 5's own folder:
#
#     step1_nnunet_baseline/data/nnUNet_raw          <- inputs, shared, read-only
#     step1_nnunet_baseline/data/nnUNet_preprocessed <- plans/fingerprint, read-only
#     step4_.../data/boundary_fingerprint.json      <- shared measurement (read-only)
#     step5_.../data/nnUNet_results                  <- OUR training results
#
#   TRAINER REGISTRATION -- zk ships nnunetv2 2.6.2, whose `-tr <Name>` lookup
#   only scans the files inside <site-packages>/nnunetv2/training/nnUNetTrainer
#   (no nnUNet_extTrainer support before 2.8), so the classes are registered by
#   a one-file shim:
#
#       bash scripts/70_install_trainer_shim.sh        # idempotent
#
# ---------------------------------------------------------------------------
set -e

STEP5_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USS_ROOT="$(cd "$STEP5_ROOT/.." && pwd)"

# step 1 provides nnUNet_raw / nnUNet_preprocessed / venv / DATASET_ID / CONFIG / FOLD
# shellcheck source=/dev/null
source "$USS_ROOT/step1_nnunet_baseline/scripts/env.sh"

export nnUNet_results="$STEP5_ROOT/data/nnUNet_results"
export nnUNet_extTrainer="$STEP5_ROOT/trainer"
export PYTHONPATH="$STEP5_ROOT/trainer:${PYTHONPATH:-}"

export STEP5_VENV="$NNUNET_VENV"
export STEP5_ROOT
export STEP5_FP_JSON="${STEP5_FP_JSON:-$USS_ROOT/step4_our_new_model_resunet_boundary/data/boundary_fingerprint.json}"

# --- squeeze knobs (all optional; the defaults live in trainer/squeeze_config.py)
export STEP5_EPOCHS="${STEP5_EPOCHS:-500}"
export STEP5_R0="${STEP5_R0:-}"            # initial band half-width (px)
export STEP5_RMIN="${STEP5_RMIN:-}"        # final band half-width (px)
export STEP5_WARMUP="${STEP5_WARMUP:-}"
export STEP5_SCHEDULE="${STEP5_SCHEDULE:-}"
export STEP5_W_ORDER="${STEP5_W_ORDER:-}"  # ordering hinge weight
export STEP5_W_GAP="${STEP5_W_GAP:-}"      # squeeze gap weight
export STEP5_W_IN="${STEP5_W_IN:-}"
export STEP5_W_OUT="${STEP5_W_OUT:-}"
export STEP5_ANGULAR="${STEP5_ANGULAR:-}"  # 1 = angular band, 0 = isotropic
export STEP5_HEAD_BIAS="${STEP5_HEAD_BIAS:-}"
export STEP5_LOG_EVERY="${STEP5_LOG_EVERY:-}"

# `nnUNet_compile` is deliberately left at nnU-Net's default (compile ON) because
# the step-1 baseline and the step-4 arms were trained that way.  Set
# nnUNet_compile=False for a conservative single-run fallback.
export nnUNet_compile="${nnUNet_compile:-}"

mkdir -p "$nnUNet_results" "$STEP5_ROOT/logs" "$STEP5_ROOT/figures" \
         "$STEP5_ROOT/predictions" "$STEP5_ROOT/metrics"

echo "-----------------------------------------------------------------"
echo "[step5] STEP5_ROOT     = $STEP5_ROOT"
echo "[step5] nnUNet_raw     = $nnUNet_raw            (shared, read-only)"
echo "[step5] nnUNet_preproc = $nnUNet_preprocessed   (shared, read-only)"
echo "[step5] nnUNet_results = $nnUNet_results        (step5 only)"
echo "[step5] fingerprint    = $STEP5_FP_JSON"
echo "[step5] epochs         = $STEP5_EPOCHS"
echo "[step5] angular band   = ${STEP5_ANGULAR:-<trainer default>}"
echo "[step5] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "-----------------------------------------------------------------"

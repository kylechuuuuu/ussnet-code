#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step6_sandwich_squeeze/scripts/env_step6.sh
#
#   Environment for the sandwich squeeze (夹逼定理 v2) boundary model.
#
#       source scripts/env_step6.sh
#
#   Reuses step 1's dataset/preprocessing/fold-0 split and step 4's boundary
#   fingerprint (read-only), and writes everything it produces into step 6's
#   own folder:
#
#     step1_nnunet_baseline/data/nnUNet_raw          <- inputs, shared, read-only
#     step1_nnunet_baseline/data/nnUNet_preprocessed <- plans/fingerprint, read-only
#     step4_.../data/boundary_fingerprint.json       <- shared measurement (read-only)
#     step6_.../data/nnUNet_results                   <- OUR training results
#
#   TRAINER REGISTRATION -- zk ships nnunetv2 2.6.2, whose `-tr <Name>` lookup
#   only scans the files inside <site-packages>/nnunetv2/training/nnUNetTrainer
#   (no nnUNet_extTrainer support before 2.8), so the class is registered by a
#   one-file shim:
#
#       bash scripts/70_install_trainer_shim.sh        # idempotent
#
# ---------------------------------------------------------------------------
set -e

STEP6_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USS_ROOT="$(cd "$STEP6_ROOT/.." && pwd)"

# step 1 provides nnUNet_raw / nnUNet_preprocessed / venv / DATASET_ID / CONFIG / FOLD
# shellcheck source=/dev/null
source "$USS_ROOT/step1_nnunet_baseline/scripts/env.sh"

export nnUNet_results="$STEP6_ROOT/data/nnUNet_results"
export nnUNet_extTrainer="$STEP6_ROOT/trainer"
export PYTHONPATH="$STEP6_ROOT/trainer:${PYTHONPATH:-}"

export STEP6_VENV="$NNUNET_VENV"
export STEP6_ROOT
export USS_ROOT
export STEP6_FP_JSON="${STEP6_FP_JSON:-$USS_ROOT/step4_our_new_model_resunet_boundary/data/boundary_fingerprint.json}"

# --- sandwich knobs (all optional; the defaults live in trainer/sandwich_config.py)
export STEP6_EPOCHS="${STEP6_EPOCHS:-500}"
export STEP6_RC="${STEP6_RC:-}"            # cautious radius  (erode GT)   px
export STEP6_RA="${STEP6_RA:-}"            # amplified radius (dilate GT)  px
export STEP6_WC="${STEP6_WC:-}"            # cautious head loss weight
export STEP6_WA="${STEP6_WA:-}"            # amplified head loss weight
export STEP6_ANGULAR="${STEP6_ANGULAR:-}"  # 1 = fingerprint-angular bounds
export STEP6_HEAD_BIAS="${STEP6_HEAD_BIAS:-}"
export STEP6_LOG_EVERY="${STEP6_LOG_EVERY:-}"

# `nnUNet_compile` stays at nnU-Net's default (compile ON) exactly like the
# step-1 baseline, the step-4 arms and the step-5 arms -> same epoch times,
# same comparability.
export nnUNet_compile="${nnUNet_compile:-}"

mkdir -p "$nnUNet_results" "$STEP6_ROOT/logs" "$STEP6_ROOT/figures" \
         "$STEP6_ROOT/predictions" "$STEP6_ROOT/metrics" "$STEP6_ROOT/results"

echo "-----------------------------------------------------------------"
echo "[step6] STEP6_ROOT     = $STEP6_ROOT"
echo "[step6] nnUNet_raw     = $nnUNet_raw            (shared, read-only)"
echo "[step6] nnUNet_preproc = $nnUNet_preprocessed   (shared, read-only)"
echo "[step6] nnUNet_results = $nnUNet_results        (step6 only)"
echo "[step6] fingerprint    = $STEP6_FP_JSON"
echo "[step6] epochs         = $STEP6_EPOCHS"
echo "[step6] radii          = r_c=${STEP6_RC:-<3.0>} r_a=${STEP6_RA:-<4.0>}"
echo "[step6] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "-----------------------------------------------------------------"

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/env_step4.sh
#
# Environment for the redesigned, FINGERPRINT-DRIVEN model.
#
#   source scripts/env_step4.sh
#
# It deliberately REUSES step 1's preprocessing (same dataset, same fold-0
# split, so the comparison against nnU-Net is fair and the plans are identical)
# but writes everything it produces into step 4's own folder, so each
# experiment folder stays self-contained:
#
#   step1_nnunet_baseline/data/nnUNet_raw          <- inputs (NIfTI), shared, never written
#   step1_nnunet_baseline/data/nnUNet_preprocessed <- fingerprint + plans, shared, read-only
#   step4_.../data/boundary_fingerprint.json       <- OUR added fingerprint half
#   step4_.../data/nnUNet_results                  <- OUR training results
#
# TRAINER REGISTRATION -- READ THIS BEFORE RUNNING `-tr`
# -----------------------------------------------------
# The baseline we must beat (Dice 96.106) was trained with `~/anaconda3/envs/zk`,
# and that is the env used here for a fair comparison.  zk ships nnunetv2
# **2.6.2**, which has NO `nnUNet_extTrainer` support: it resolves `-tr <Name>`
# only by scanning the FILES inside
#   <site-packages>/nnunetv2/training/nnUNetTrainer
# and it has no registration constant.  A project-local trainer must therefore
# be registered through a one-file shim placed in that folder:
#
#     bash scripts/53_install_trainer_shim.sh     # idempotent
#
# `nnUNet_extTrainer` and `PYTHONPATH` are still exported below: they are
# harmless under 2.6.2 and correct if this ever runs under nnU-Net >= 2.8, but
# they are NOT sufficient on their own for the zk env.  Likewise,
# `build_network_architecture` must use 2.6.2's SIX-argument signature
# `(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
#   num_input_channels, num_output_channels, enable_deep_supervision)`
# -- nnU-Net >= 2.8 replaced the first three with plans_manager/
# configuration_manager.  See scripts/53_install_trainer_shim.sh.
# ---------------------------------------------------------------------------
set -e

STEP4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USS_ROOT="$(cd "$STEP4_ROOT/.." && pwd)"

# step 1 provides nnUNet_raw / nnUNet_preprocessed / PATH to the venv /
# CUDA_VISIBLE_DEVICES=0 / DATASET_ID / CONFIG / FOLD.
# shellcheck source=/dev/null
source "$USS_ROOT/step1_nnunet_baseline/scripts/env.sh"

# --- our results live with us, not inside step 1 ---------------------------
export nnUNet_results="$STEP4_ROOT/data/nnUNet_results"

# --- our custom trainers ----------------------------------------------------
export nnUNet_extTrainer="$STEP4_ROOT/trainer"
export PYTHONPATH="$STEP4_ROOT/trainer:${PYTHONPATH:-}"
export STEP4_TRAINER="${STEP4_TRAINER:-nnUNetTrainerProstateFPBoundary}"
export STEP4_TAG="${STEP4_TAG:-fpA_stock_fploss}"
export STEP4_VENV="$NNUNET_VENV"

# --- knobs read by the fingerprint-driven trainers --------------------------
# The loss weights are NOT hand-tuned here: they are DERIVED from
# data/boundary_fingerprint.json (see tools/50_extract_boundary_fingerprint.py
# and trainer/fp_config.py).  These env vars exist for ablations.
export STEP4_EPOCHS="${STEP4_EPOCHS:-500}"
export STEP4_USE_DIRECTION="${STEP4_USE_DIRECTION:-0}"
export STEP4_FP_JSON="${STEP4_FP_JSON:-$STEP4_ROOT/data/boundary_fingerprint.json}"
export STEP4_WARMUP="${STEP4_WARMUP:-}"          # empty -> use the fingerprint's value (100)
export STEP4_W_TOL="${STEP4_W_TOL:-}"            # tolerant boundary Dice weight
export STEP4_W_BCE="${STEP4_W_BCE:-}"            # sharp boundary BCE weight
export STEP4_W_PER="${STEP4_W_PER:-}"            # perimeter (length) weight
export STEP4_TOL_BAND="${STEP4_TOL_BAND:-}"      # tolerance band in px (odd)
export STEP4_DISABLE_EXTRA="${STEP4_DISABLE_EXTRA:-0}"   # 1 -> plain Dice+CE (ablation)

mkdir -p "$nnUNet_results" "$STEP4_ROOT/logs" "$STEP4_ROOT/figures" \
         "$STEP4_ROOT/predictions" "$STEP4_ROOT/metrics"

echo "-----------------------------------------------------------------"
echo "[step4] STEP4_ROOT     = $STEP4_ROOT"
echo "[step4] nnUNet_raw     = $nnUNet_raw            (shared, read-only)"
echo "[step4] nnUNet_preproc = $nnUNet_preprocessed   (shared, read-only)"
echo "[step4] nnUNet_results = $nnUNet_results        (step4 only)"
echo "[step4] trainer        = $STEP4_TRAINER"
echo "[step4] fingerprint    = $STEP4_FP_JSON"
echo "[step4] epochs         = $STEP4_EPOCHS"
echo "[step4] direction w.   = $STEP4_USE_DIRECTION"
echo "[step4] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "-----------------------------------------------------------------"

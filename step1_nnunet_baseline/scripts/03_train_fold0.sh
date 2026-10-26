#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step1_nnunet_baseline/scripts/03_train_fold0.sh
#   Train nnU-Net 2d on fold 0 (= the exact U-Net baseline split) on GPU 0.
#
#   usage:  bash scripts/03_train_fold0.sh [fold] [trainer]
#           bash scripts/03_train_fold0.sh 0 nnUNetTrainerBenchmark_5epochs
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

FOLD="${1:-0}"
TRAINER="${2:-nnUNetTrainer}"
mkdir -p logs
LOG="logs/03_train_fold${FOLD}_${TRAINER}.log"

echo "dataset=${DATASET_NAME} config=${CONFIG} fold=${FOLD} trainer=${TRAINER}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | sed -n '1p'

nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" \
    -tr "${TRAINER}" \
    -device cuda \
    2>&1 | tee "${LOG}"

echo "training finished -> ${LOG}"

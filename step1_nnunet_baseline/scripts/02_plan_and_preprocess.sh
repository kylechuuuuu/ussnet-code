#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step1_nnunet_baseline/scripts/02_plan_and_preprocess.sh
#   Extract the nnU-Net dataset fingerprint and run 2d preprocessing.
#   Everything is written under step1/data/nnUNet_preprocessed/.
#
#   NOTE: nnU-Net 2.8.1 has no separate `nnUNetv2_verify_dataset_integrity`
#   console script (it only exists as a flag).  The integrity check is
#   therefore requested through --verify_dataset_integrity below.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

mkdir -p logs
LOG="logs/02_plan_and_preprocess.log"

echo "[1/1] plan_and_preprocess (configuration ${CONFIG}) - fingerprint + preprocessing"
nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID}" \
    -c "${CONFIG}" \
    -np 8 \
    --verify_dataset_integrity \
    2>&1 | tee "${LOG}"

echo
echo "done.  next:"
echo "  python scripts/01_write_splits_final.py"
echo "  bash   scripts/03_train_fold0.sh 0 nnUNetTrainerBenchmark_5epochs   # timing only"
echo "  bash   scripts/03_train_fold0.sh                                    # real run"

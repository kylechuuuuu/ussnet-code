#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step1_nnunet_baseline/scripts/04_predict_fold0.sh
#   Run inference with the fold-0 checkpoint on GPU 0.
#   Predicts ALL 102 cases (train + val) so that the comparison in step 2 can
#   also report train-set behaviour (over-/under-fitting) for both models.
#
#   usage: bash scripts/04_predict_fold0.sh [fold] [checkpoint]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

FOLD="${1:-0}"
CKPT="${2:-checkpoint_best.pth}"
OUT="predictions/nnunet_${CONFIG}_fold${FOLD}_${CKPT%.pth}_nifti"
mkdir -p logs "${OUT}"

echo "predicting with ${CKPT} (fold ${FOLD}) -> ${OUT}"

# -stepsize 0.5 is the nnU-Net default; -npp 3 -nps 3 are the default TTA
nnUNetv2_predict \
    -i "${nnUNet_raw}/${DATASET_NAME}/imagesTr" \
    -o "${OUT}" \
    -d "${DATASET_ID}" \
    -c "${CONFIG}" \
    -f "${FOLD}" \
    -chk "${CKPT}" \
    -device cuda \
    --save_probabilities \
    2>&1 | tee "logs/04_predict_fold${FOLD}_${CKPT%.pth}.log"

echo "predictions -> ${OUT}"
ls "${OUT}" | wc -l

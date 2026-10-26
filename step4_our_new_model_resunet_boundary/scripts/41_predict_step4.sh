#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/41_predict_step4.sh
#
#   Inference for the redesigned model: same nnU-Net sliding-window prediction
#   with mirroring TTA that the baseline uses, over ALL 102 cases, so step 2 can
#   report val AND train behaviour.  Probabilities (.npz) are saved on purpose:
#   they are what makes a fair probability-space ensemble with the nnU-Net
#   baseline possible (both models use the same plans, so the .npz live in the
#   same preprocessed geometry).
#
#   usage: bash scripts/41_predict_step4.sh [fold] [checkpoint]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step4.sh

FOLD="${1:-0}"
CKPT="${2:-checkpoint_best.pth}"
OUT="$STEP4_ROOT/predictions/${STEP4_TAG}_${CONFIG}_fold${FOLD}_${CKPT%.pth}_nifti"
mkdir -p "$OUT" "$STEP4_ROOT/logs"

echo "[step4] predicting with $CKPT (fold $FOLD) -> $OUT"

"$STEP4_VENV/bin/nnUNetv2_predict" \
    -i "${nnUNet_raw}/${DATASET_NAME}/imagesTr" \
    -o "${OUT}" \
    -d "${DATASET_ID}" \
    -c "${CONFIG}" \
    -f "${FOLD}" \
    -chk "${CKPT}" \
    -device cuda \
    --save_probabilities \
    2>&1 | tee "$STEP4_ROOT/logs/41_predict_fold${FOLD}_${CKPT%.pth}.log"

echo "[step4] seg + probability maps -> $OUT"
ls "$OUT" | wc -l

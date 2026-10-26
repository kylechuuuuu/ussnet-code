#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/79_finish_ctrl200.sh
#
#   Finish the RESUMED control run (nnUNetTrainerProstateStock200, fold 0):
#   predict -> export PNG -> official evaluation, using checkpoint_best of the
#   resumed run.  (The original 78 wrapper would retrain from scratch, so the
#   post-training chain is repeated here for the resume.)
#
#   usage: bash scripts/79_finish_ctrl200.sh [gpu]
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step5.sh > /dev/null 2>&1 || true
set +e
GPU="${1:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"
USS_ROOT="$(cd "$STEP5_ROOT/.." && pwd)"

TRAINER="nnUNetTrainerProstateStock200"
TAG="ctrl_stock200_resumed"
RUN="$nnUNet_results/Dataset001_UltrasoundProstate/${TRAINER}__nnUNetPlans__2d/fold_0"
NIFTI="$STEP5_ROOT/predictions/${TAG}_nifti"
PNG="$STEP5_ROOT/predictions/${TAG}_png"

echo "[79] $(date -u +%Y-%m-%dT%H:%M:%SZ) predict checkpoint_best of the resumed run"
mkdir -p "$NIFTI" "$PNG"
"$STEP5_VENV/bin/nnUNetv2_predict" \
    -i "$nnUNet_raw/Dataset001_UltrasoundProstate/imagesTr" \
    -o "$NIFTI" -d "$DATASET_ID" -c "$CONFIG" -f "$FOLD" \
    -chk checkpoint_best.pth -tr "$TRAINER" -device cuda \
    2>&1 | tee "$STEP5_ROOT/logs/79_predict_${TAG}.log" | tail -2

"$STEP5_VENV/bin/python" "$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py" \
    --pred "$NIFTI" --out "$PNG" --threshold 0.5 \
    2>&1 | tee "$STEP5_ROOT/logs/79_export_${TAG}.log" | tail -2

"$STEP5_VENV/bin/python" "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
    --pred "$PNG" --tag "${TAG}_best" --split val \
    2>&1 | tee "$STEP5_ROOT/logs/79_eval_${TAG}.log" | tail -3

echo "[79] $(date -u +%Y-%m-%dT%H:%M:%SZ) done (evaluator tag: ${TAG}_best)"

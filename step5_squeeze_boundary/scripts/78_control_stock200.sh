#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step5_squeeze_boundary/scripts/78_control_stock200.sh
#
#   H1 (method) vs H2 (budget) control experiment.
#
#   Trains the STOCK nnU-Net recipe for exactly the same number of epochs as the
#   squeeze arms (STEP5_EPOCHS, default 200), on the same fold, plans,
#   preprocessing and augmentation, then predicts/exports/scores it with the
#   same pipeline.  Its boundary Dice@1px is the budget-matched reference:
#
#       stock(200)  ~= 32       -> the squeeze arm lost nothing, 200 epochs is
#                                  simply not 1000        (H2: budget)
#       stock(200)  ~= 35       -> the squeeze machinery itself costs boundary
#                                  accuracy               (H1: method)
#
#   usage: setsid nohup bash scripts/78_control_stock200.sh 7 200 \
#              > logs/78_control.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_step5.sh > /dev/null 2>&1 || true
set +e

GPU="${1:-7}"
EPOCHS="${2:-200}"
export CUDA_VISIBLE_DEVICES="$GPU"
export STEP5_EPOCHS="$EPOCHS"

TRAINER="nnUNetTrainerProstateStock200"
TAG="ctrl_stock${EPOCHS}"
RUN="$nnUNet_results/Dataset001_UltrasoundProstate/${TRAINER}__nnUNetPlans__2d/fold_0"
NIFTI="$STEP5_ROOT/predictions/${TAG}_nifti"
PNG="$STEP5_ROOT/predictions/${TAG}_png"

echo "[78] $(date -u +%Y-%m-%dT%H:%M:%SZ) control run: stock nnU-Net, $EPOCHS epochs, gpu $GPU"
"$STEP5_VENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER" -device cuda \
    2>&1 | tee "$STEP5_ROOT/logs/78_train_${TAG}.log"
echo "[78] $(date -u +%Y-%m-%dT%H:%M:%SZ) training rc=$?"

for CK in checkpoint_best.pth checkpoint_final.pth; do
    [ -f "$RUN/$CK" ] || { echo "[78] missing $CK"; continue; }
    CTAG="${TAG}_${CK%.pth}"
    echo "[78] $(date -u +%Y-%m-%dT%H:%M:%SZ) predict with $CK -> $CTAG"
    mkdir -p "$NIFTI" "$PNG"
    "$STEP5_VENV/bin/nnUNetv2_predict" \
        -i "$nnUNet_raw/Dataset001_UltrasoundProstate/imagesTr" \
        -o "$NIFTI" -d "$DATASET_ID" -c "$CONFIG" -f "$FOLD" \
        -chk "$CK" -tr "$TRAINER" -device cuda \
        2>&1 | tee "$STEP5_ROOT/logs/78_predict_${CTAG}.log"
    "$STEP5_VENV/bin/python" "$USS_ROOT/step1_nnunet_baseline/scripts/05_export_nifti_to_png.py" \
        --pred "$NIFTI" --out "$PNG" --threshold 0.5 \
        2>&1 | tee "$STEP5_ROOT/logs/78_export_${CTAG}.log"
    "$STEP5_VENV/bin/python" "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/10_evaluate_predictions.py" \
        --pred "$PNG" --tag "$CTAG" --split val \
        2>&1 | tee "$STEP5_ROOT/logs/78_eval_${CTAG}.log"
done

echo "[78] $(date -u +%Y-%m-%dT%H:%M:%SZ) rebuild REPORT.md (control enters the comparison table)"
"$STEP5_VENV/bin/python" tools/64_make_report.py 2>&1 | tail -2
echo "[78] $(date -u +%Y-%m-%dT%H:%M:%SZ) control run done"

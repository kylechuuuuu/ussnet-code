#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step1_nnunet_baseline/scripts/07_watch_then_predict_and_evaluate.sh
#
#   The stock nnU-Net fold-0 training takes ~7 h, so instead of coming back at a
#   guessed wall-clock time this watcher blocks until the training process is
#   really gone, verifies that it ended normally, then runs the whole
#   predict -> export PNG -> shared-evaluator chain for BOTH checkpoints.
#
#   Why two checkpoints: nnU-Net's own recipe reports the model at the END of
#   training, while the U-Net baseline reported its best-on-validation weights.
#   Reporting both keeps the comparison honest in either direction:
#     checkpoint_final.pth -> no selection on the 31 validation images at all
#     checkpoint_best.pth  -> selected on the same 31 images as the U-Net
#
#   usage: setsid nohup bash scripts/07_watch_then_predict_and_evaluate.sh &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP1_ROOT="$(pwd)"
USS_ROOT="$(cd "$STEP1_ROOT/.." && pwd)"
source scripts/env.sh

LOG="$STEP1_ROOT/logs/07_watch_predict_evaluate.log"
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[watch] started            : $(date -u)"

TRAIN_LOG="$STEP1_ROOT/logs/03_train_fold0_nnUNetTrainer.log"
RUN_DIR="$nnUNet_results/$DATASET_NAME/nnUNetTrainer__nnUNetPlans__2d/fold_0"

# ---- 1. wait for the training process to disappear ------------------------
while pgrep -f "nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD}" >/dev/null 2>&1; do
    sleep 60
done
echo "[watch] training process gone: $(date -u)"

# ---- 2. did it end normally? ---------------------------------------------
if ! grep -q "Training done" "$TRAIN_LOG"; then
    echo "[watch] ABORT: '$TRAIN_LOG' has no 'Training done.' line - the run did not finish normally."
    tail -30 "$TRAIN_LOG"
    exit 1
fi
for ck in checkpoint_final.pth checkpoint_best.pth; do
    [ -f "$RUN_DIR/$ck" ] || { echo "[watch] ABORT: missing $RUN_DIR/$ck"; exit 1; }
done
echo "[watch] checkpoints found in $RUN_DIR"

# ---- 3. predict with both checkpoints (all 102 cases) --------------------
for ck in checkpoint_final.pth checkpoint_best.pth; do
    echo "[watch] === predict $ck ==="
    bash scripts/04_predict_fold0.sh "$FOLD" "$ck"
done

# ---- 4. export to PNG + score with the shared evaluator ------------------
for ck in checkpoint_final.pth checkpoint_best.pth; do
    short="${ck%.pth}"                       # checkpoint_final / checkpoint_best
    tag="nnunet_${CONFIG}_fold${FOLD}_${short}"
    echo "[watch] === export + evaluate $tag ==="
    bash "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/11_export_nnunet_png_and_evaluate.sh" \
        "$STEP1_ROOT/predictions/${tag}_nifti" "$tag" val
done

echo "[watch] ALL DONE: $(date -u)"
echo "=================================================================="

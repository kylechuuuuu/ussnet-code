#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/45_chain_train_predict_compare.sh
#
#   The redesigned model must not share the GPU with the stock nnU-Net run the
#   user asked for (both on card 0), so this chain waits until step 1 has
#   finished training AND finished predicting and scoring, and only then starts:
#
#     1. train the redesigned model                       (40_)
#     2. predict + score it, final and best checkpoints   (41_, 42_)
#     3. probability ensemble with the nnU-Net baseline   (49_)
#     4. final tables and figures for the whole project   (step2/15_)
#
#   The epoch budget is read from STEP4_EPOCHS.txt (digits only) at the moment
#   the training starts, so it can still be adjusted while nnU-Net is running.
#
#   usage: setsid nohup bash scripts/45_chain_train_predict_compare.sh &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP4_ROOT="$(pwd)"
USS_ROOT="$(cd "$STEP4_ROOT/.." && pwd)"
source scripts/env_step4.sh

LOG="$STEP4_ROOT/logs/45_chain_train_predict_compare.log"
exec >>"$LOG" 2>&1

WATCH_LOG="$USS_ROOT/step1_nnunet_baseline/logs/07_watch_predict_evaluate.log"
echo "=== [chain] started $(date -u) ; waiting for the nnU-Net stage to finish ==="

while ! grep -qF "[watch] ALL DONE" "$WATCH_LOG" 2>/dev/null; do
    if grep -qF "[watch] ABORT" "$WATCH_LOG" 2>/dev/null; then
        echo "[chain] the nnU-Net stage reported ABORT - step 4 NOT started"
        exit 1
    fi
    sleep 120
done
echo "[chain] nnU-Net stage finished at $(date -u)"

EP_FILE="$STEP4_ROOT/STEP4_EPOCHS.txt"
if [ -f "$EP_FILE" ]; then
    EPOCHS="$(tr -cd '0-9' < "$EP_FILE")"
else
    EPOCHS="$STEP4_EPOCHS"
fi
[ -n "$EPOCHS" ] || EPOCHS=1000
export STEP4_EPOCHS="$EPOCHS"
echo "[chain] training the redesigned model for $EPOCHS epochs"

# ---- 1. train -------------------------------------------------------------
bash scripts/40_train_step4.sh 0
echo "[chain] training done at $(date -u)"

# ---- 2. predict + evaluate both checkpoints -------------------------------
for ck in checkpoint_final.pth checkpoint_best.pth; do
    if [ "$ck" = "checkpoint_final.pth" ]; then
        tag="$STEP4_TAG"                       # our_resunet_boundary
    else
        tag="${STEP4_TAG}_best"                # our_resunet_boundary_best
    fi
    nifti="$STEP4_ROOT/predictions/${STEP4_TAG}_${CONFIG}_fold0_${ck%.pth}_nifti"
    echo "[chain] === predict $ck -> tag $tag ==="
    bash scripts/41_predict_step4.sh 0 "$ck"
    bash scripts/42_export_png_and_evaluate.sh "$nifti" "$tag" val
done

# ---- 3. ensemble with the nnU-Net baseline (final checkpoints) ------------
echo "[chain] === ensemble with nnU-Net (checkpoint_final) ==="
bash scripts/49_ensemble_and_evaluate.sh checkpoint_final

# ---- 4. final tables + figures -------------------------------------------
echo "[chain] === final comparison ==="
bash "$USS_ROOT/step2_compare_unet_vs_nnunet/scripts/15_make_final_comparison.sh"

echo "=== [chain] ALL DONE $(date -u) ==="

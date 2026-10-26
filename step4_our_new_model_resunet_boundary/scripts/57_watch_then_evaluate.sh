#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/57_watch_then_evaluate.sh
#
#   Wait for ONE pretrained-encoder arm (C = resnet18, D = resnet34) to finish,
#   verify it ended normally, then run the whole downstream chain for it:
#
#     1. predict all 102 cases (NIfTI + probabilities) for both checkpoints
#     2. NIfTI -> PNG in the U-Net baseline's convention
#     3. score the 31 fold-0 validation images with the ONE shared evaluator
#     4. head-to-head comparison table vs nnU-Net (all models present)
#     5. probability-level ensemble with every available member
#
#   Arms C and D run CONCURRENTLY on GPU 1 / GPU 0, so this watcher must wait
#   for ITS arm only -- matching the whole "nnUNetv2_train 001 2d 0" pattern
#   would block arm C's evaluation until arm D is also done.  The trainer name
#   is therefore anchored with a trailing "[ ]-device": arm C's command line
#   ends "...FPPreEnc -device cuda" while arm D's is "...FPPreEncR34 -device
#   cuda", so the anchored pattern matches exactly one of them.
#
#   Deliberately strict: aborts instead of producing a half comparison.
#
#   usage: setsid nohup bash scripts/57_watch_then_evaluate.sh C \
#              > logs/57_wrap_C.log 2>&1 < /dev/null &
#          setsid nohup bash scripts/57_watch_then_evaluate.sh D \
#              > logs/57_wrap_D.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP4_ROOT="$(pwd)"
source scripts/env_step4.sh

ARM="${1:-C}"
case "$ARM" in
    C) TR="nnUNetTrainerProstateFPPreEnc";    SHORT="fp_armC_preenc";     GPU=1 ;;
    D) TR="nnUNetTrainerProstateFPPreEncR34"; SHORT="fp_armD_preenc_r34"; GPU=0 ;;
    *) echo "[57] arm must be C or D (got '$ARM')" >&2; exit 2 ;;
esac

LOG="$STEP4_ROOT/logs/57_watch_${ARM}.log"
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[57] watcher for arm $ARM started : $(date -u)"

TRAIN_LOG="$STEP4_ROOT/logs/51_train_${SHORT}_fold${FOLD}_ep500_gpu${GPU}.log"
RUN="$nnUNet_results/$DATASET_NAME/${TR}__nnUNetPlans__2d/fold_0"

# ---- 1. wait for THIS arm's training processes to disappear --------------
while pgrep -f "nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TR}[ ]-device" >/dev/null 2>&1; do
    sleep 60
done
echo "[57] arm $ARM: no training process left : $(date -u)"

# ---- 2. did it end normally? ---------------------------------------------
if ! grep -q "Training done" "$TRAIN_LOG"; then
    echo "[57] ABORT: no 'Training done.' in $TRAIN_LOG"
    tail -40 "$TRAIN_LOG"
    exit 1
fi
for ck in checkpoint_best.pth checkpoint_final.pth; do
    [ -f "$RUN/$ck" ] || { echo "[57] ABORT: missing $RUN/$ck"; exit 1; }
done
echo "[57] arm $ARM self-reported : $(grep 'Mean Validation Dice' "$TRAIN_LOG" | tail -1)"
echo "[57] baseline nnU-Net       : 0.9603267240914037 (Dice 96.106 on the 31 val images)"

# ---- 3. predict + export + evaluate (this arm, both checkpoints) ----------
bash "$STEP4_ROOT/scripts/54_predict_and_evaluate.sh" "$GPU" "$ARM" \
     "checkpoint_best.pth checkpoint_final.pth"

STEP2="$USS_ROOT/step2_compare_unet_vs_nnunet"
PY="$STEP4_VENV/bin/python"

# ---- 4. the head-to-head table, nnU-Net as the reference ------------------
TAGS=(unet_73 nnunet_2d_fold0_checkpoint_best
      fpA_stock_fploss_checkpoint_best
      fpB_resenc_fploss_dir_checkpoint_best
      fp_armC_preenc_checkpoint_best
      fp_armD_preenc_r34_checkpoint_best)
LABELS=("U-Net (baseline)" "nnU-Net 2d (best)"
        "Ours arm A (stock + fp loss)"
        "Ours arm B (ResEnc + fp loss + dir)"
        "Ours arm C (pretrained r18 + fp loss)"
        "Ours arm D (pretrained r34 + fp loss)")

have_tags=(); have_labels=()
for i in "${!TAGS[@]}"; do
    if [ -f "$STEP2/metrics/${TAGS[$i]}_per_image.csv" ]; then
        have_tags+=("${TAGS[$i]}"); have_labels+=("${LABELS[$i]}")
    else
        echo "[57] note: no metrics for ${TAGS[$i]} - excluded from the table"
    fi
done

if [ "${#have_tags[@]}" -ge 2 ]; then
    echo "[57] === comparison table (baseline = nnU-Net) ==="
    "$PY" "$STEP2/scripts/12_make_comparison_tables.py" \
        --tags "${have_tags[@]}" --labels "${have_labels[@]}" \
        --baseline nnunet_2d_fold0_checkpoint_best \
        --title "USSnet fold-0: U-Net vs nnU-Net vs the fingerprint-driven models" \
        --out-prefix comparison_fp 2>&1 | tee "$STEP4_ROOT/logs/57_comparison_table_${ARM}.log"
fi

# ---- 5. ensemble over every member that now exists -----------------------
echo "[57] === ensemble (every available member) ==="
ENSEMBLE_OUT="$STEP4_ROOT/results/ensemble_eval_withC.json" \
    "$PY" "$STEP4_ROOT/tools/69_ensemble_eval.py" 2>&1 | tee "$STEP4_ROOT/logs/57_ensemble_${ARM}.log"

echo "[57] arm $ARM ALL DONE : $(date -u)"
echo "=================================================================="

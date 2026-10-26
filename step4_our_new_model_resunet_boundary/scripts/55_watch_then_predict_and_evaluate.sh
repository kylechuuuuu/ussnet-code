#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# step4_our_new_model_resunet_boundary/scripts/55_watch_then_predict_and_evaluate.sh
#
#   The 500-epoch runs take ~3.5-4 h each, so instead of coming back at a
#   guessed wall-clock time this watcher blocks until BOTH training processes
#   are really gone, verifies that each ended normally, then runs the whole
#   predict -> NIfTI->PNG -> shared-evaluator chain for both arms and both
#   checkpoints.
#
#   It is deliberately strict: if a run did not print "Training done." or a
#   checkpoint is missing, it ABORTS instead of producing half a comparison.
#
#   usage: setsid nohup bash scripts/55_watch_then_predict_and_evaluate.sh \
#              > logs/55_watch.log 2>&1 < /dev/null &
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP4_ROOT="$(pwd)"
source scripts/env_step4.sh

LOG="$STEP4_ROOT/logs/55_watch.log"
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[55] watcher started : $(date -u)"

LOG_A="$STEP4_ROOT/logs/51_train_fp_armA_stock_arch_fold0_ep500_gpu0.log"
LOG_B="$STEP4_ROOT/logs/51_train_fp_armB_resenc_dir_fold0_ep500_gpu1.log"
TR_A="nnUNetTrainerProstateFPBoundary"
TR_B="nnUNetTrainerProstateResUNetFPBoundary"
RUN_A="$nnUNet_results/$DATASET_NAME/${TR_A}__nnUNetPlans__2d/fold_0"
RUN_B="$nnUNet_results/$DATASET_NAME/${TR_B}__nnUNetPlans__2d/fold_0"

# ---- 1. wait for both training processes to disappear ---------------------
while pgrep -f "nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD}" >/dev/null 2>&1; do
    sleep 60
done
echo "[55] no training process left : $(date -u)"

# ---- 2. did both end normally? -------------------------------------------
fail=0
for pair in "A:$LOG_A:$RUN_A" "B:$LOG_B:$RUN_B"; do
    arm="${pair%%:*}"; rest="${pair#*:}"; log="${rest%%:*}"; run="${rest#*:}"
    if ! grep -q "Training done" "$log"; then
        echo "[55] ABORT: arm $arm log has no 'Training done.' -> did not finish normally"
        tail -30 "$log"
        fail=1
    fi
    for ck in checkpoint_best.pth checkpoint_final.pth; do
        [ -f "$run/$ck" ] || { echo "[55] ABORT: arm $arm missing $run/$ck"; fail=1; }
    done
    echo "[55] arm $arm checkpoints: $(ls "$run" | grep -c '^checkpoint.*pth$') found in $run"
done
[ "$fail" -eq 0 ] || { echo "[55] stopping; nothing evaluated."; exit 1; }

# ---- 3. report the final validation Dice nnU-Net itself computed ----------
for pair in "A:$LOG_A" "B:$LOG_B"; do
    arm="${pair%%:*}"; log="${pair#*:}"
    echo "[55] arm $arm self-reported: $(grep 'Mean Validation Dice' "$log" | tail -1)"
done
echo "[55] baseline nnU-Net best  : 0.9603267240914037 (Dice 96.106 on the same 31 images)"

# ---- 4. predict + export + evaluate --------------------------------------
bash "$STEP4_ROOT/scripts/54_predict_and_evaluate.sh" 0 "A B" \
     "checkpoint_best.pth checkpoint_final.pth"

# ---- 5. the head-to-head table, with nnU-Net as the reference ------------
#      (the whole point is "beat nnU-Net", so the paired deltas and the
#       Wilcoxon test are taken against the nnU-Net baseline, not the U-Net)
STEP2="$USS_ROOT/step2_compare_unet_vs_nnunet"
PY="$STEP4_VENV/bin/python"
TAGS=(unet_73 nnunet_2d_fold0_checkpoint_best
      fpA_stock_fploss_checkpoint_best
      fpB_resenc_fploss_dir_checkpoint_best)
LABELS=("U-Net (baseline)" "nnU-Net 2d (best)"
        "Ours arm A (stock + fp loss)"
        "Ours arm B (ResEnc + fp loss + dir)")

have_tags=(); have_labels=()
for i in "${!TAGS[@]}"; do
    if [ -f "$STEP2/metrics/${TAGS[$i]}_per_image.csv" ]; then
        have_tags+=("${TAGS[$i]}"); have_labels+=("${LABELS[$i]}")
    else
        echo "[55] note: no metrics for ${TAGS[$i]} - excluded from the table"
    fi
done

if [ "${#have_tags[@]}" -ge 2 ]; then
    echo "[55] === comparison table (baseline = nnU-Net) ==="
    "$PY" "$STEP2/scripts/12_make_comparison_tables.py" \
        --tags "${have_tags[@]}" --labels "${have_labels[@]}" \
        --baseline nnunet_2d_fold0_checkpoint_best \
        --title "USSnet fold-0: U-Net vs nnU-Net vs the fingerprint-driven model" \
        --out-prefix comparison_fp 2>&1 | tee "$STEP4_ROOT/logs/55_comparison_table.log"
fi

# ---- 6. the clarity figures for arm B's best checkpoint ------------------
OURS_TAG=fpB_resenc_fploss_dir_checkpoint_best
if [ -f "$STEP2/metrics/${OURS_TAG}_per_image.csv" ]; then
    echo "[55] === figures for $OURS_TAG ==="
    "$PY" "$STEP4_ROOT/scripts/56_make_fp_figures.py" \
        --ours-tag "$OURS_TAG" \
        --ours-dir "$STEP4_ROOT/predictions/${OURS_TAG}_png" \
        --ours-label "Ours (ResEnc + fingerprint loss + direction)" \
        2>&1 | tee "$STEP4_ROOT/logs/55_figures.log"
fi

echo "[55] ALL DONE : $(date -u)"
echo "=================================================================="

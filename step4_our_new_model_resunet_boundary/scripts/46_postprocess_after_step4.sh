#!/usr/bin/env bash
# 46 —— step4 训练完成之后的收尾：诊断（误差分解 + 概率诊断 + 训练曲线）+ 更新对比表和图。
# 幂等：任何一步失败都会打印出来但不会中断后面的步骤；可以反复运行。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP4="$(cd "$HERE/.." && pwd)"
USS="$(cd "$STEP4/.." && pwd)"
STEP2="$USS/step2_compare_unet_vs_nnunet"
STEP3="$USS/step3_failure_analysis"
PY="$HOME/anaconda3/envs/zk/bin/python"      # 需要 scipy/skimage/matplotlib
LOG="$STEP4/logs/46_postprocess_after_step4.log"
mkdir -p "$STEP4/logs"

run () { echo "=== [46] $* ==="; "$@" ; echo "=== [46] exit=$? ==="; }

{
  echo "############ 46 started $(date -u) ############"

  run bash "$STEP3/scripts/32_three_way_error_decomposition.sh"
  run bash "$STEP3/scripts/37_probability_diagnostics_all.sh"

  # 训练曲线：nnU-Net 与我们自己的 training_log_*.txt
  NN_LOG=$(ls -1 "$USS/step1_nnunet_baseline/data/nnUNet_results/Dataset001_UltrasoundProstate"/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_*.txt 2>/dev/null | head -1)
  OUR_LOG=$(ls -1 "$STEP4/data/nnUNet_results/Dataset001_UltrasoundProstate"/nnUNetTrainerProstateResUNetBoundary__nnUNetPlans__2d/fold_0/training_log_*.txt 2>/dev/null | head -1)
  CURVE_ARGS=()
  [ -n "${NN_LOG:-}" ] && CURVE_ARGS+=(--log "$NN_LOG" --label "nnU-Net 2d (stock)")
  [ -n "${OUR_LOG:-}" ] && CURVE_ARGS+=(--log "$OUR_LOG" --label "OURS ResUNet+boundary")
  if [ ${#CURVE_ARGS[@]} -ge 4 ]; then
    run "$PY" "$STEP3/scripts/34_plot_training_curves.py" "${CURVE_ARGS[@]}" \
        --out "$STEP3/figures/training_curves.png" --title "fold 0 training curves (pseudo Dice / loss)" 
  fi
  # step4 自己的训练曲线（单独一张，附带边界项权重的说明）
  if [ -n "${OUR_LOG:-}" ]; then
    run "$PY" "$STEP3/scripts/34_plot_training_curves.py" --log "$OUR_LOG" --label "OURS ResUNet+boundary" \
        --out "$STEP4/figures/46_our_training_curves.png" --title "OURS ResUNet+boundary, fold 0" 
  fi

  # 最终对比表 + 图（15 会自动发现已经就绪的模型）
  run bash "$STEP2/scripts/15_make_final_comparison.sh"

  echo "############ 46 finished $(date -u) ############"
} >> "$LOG" 2>&1

tail -40 "$LOG"
echo "[46] full log -> $LOG"

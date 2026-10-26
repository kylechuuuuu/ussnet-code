#!/usr/bin/env bash
# 32 —— 三路误差分解：UNet 基线 / stock nnU-Net 2d / 我们重新设计的模型
# 用的是同一个脚本（30_error_decomposition.py）和同一把尺子，所以数字可以直接并排看。
# 任何一个模型的 PNG 目录还不存在就自动跳过，因此可以随时重复运行。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP3="$(cd "$HERE/.." && pwd)"
USS="$(cd "$STEP3/.." && pwd)"
mkdir -p "$STEP3/logs"
PY="$HOME/anaconda3/envs/zk/bin/python"   # step3 的脚本需要 scipy/skimage，系统 python3 没有

TAGS=()
add_if_exists () {  # add_if_exists <tag> <png dir>
  if [ -d "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ]; then
    TAGS+=("$1:$2")
    echo "[32] + $1  <- $2"
  else
    echo "[32] - skip $1 (not ready: $2)"
  fi
}
add_if_exists unet_73                        "$USS/unet/results/ultrasound_prostate_73"
add_if_exists nnunet_2d_fold0_checkpoint_final "$USS/step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_final_png"
add_if_exists nnunet_2d_fold0_checkpoint_best  "$USS/step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_best_png"
add_if_exists our_resunet_boundary           "$USS/step4_our_new_model_resunet_boundary/predictions/our_resunet_boundary_png"
add_if_exists our_resunet_boundary_best      "$USS/step4_our_new_model_resunet_boundary/predictions/our_resunet_boundary_best_png"
add_if_exists ensemble_ours_and_nnunet       "$USS/step4_our_new_model_resunet_boundary/predictions/ensemble_ours_and_nnunet_png"

LOG="$STEP3/logs/32_three_way_error_decomposition.log"
if [ ${#TAGS[@]} -eq 0 ]; then
  echo "[32] nothing ready yet" | tee -a "$LOG"
  exit 1
fi

"$PY" "$STEP3/scripts/30_error_decomposition.py" --tags "${TAGS[@]}" --split val 2>&1 | tee -a "$LOG"

# 汇总成一张 md 表，方便直接贴进 FINDINGS.md
"$PY" "$STEP3/scripts/33_make_error_summary_table.py" 2>&1 | tee -a "$LOG"
echo "[32] DONE"

#!/usr/bin/env bash
# 37 —— 对"已经出好概率图"的所有模型跑概率层面的诊断（36），自动跳过还没准备好的。
# 需要 nnunet venv 的解释器（scipy / scikit-image / matplotlib 都在里面）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP3="$(cd "$HERE/.." && pwd)"
USS="$(cd "$STEP3/.." && pwd)"
PY="$HOME/anaconda3/envs/zk/bin/python"
mkdir -p "$STEP3/logs"
LOG="$STEP3/logs/37_probability_diagnostics.log"

SRCS=()
add () {  # add <tag> <prob dir>
  if [ -d "$2" ] && { ls "$2"/*.npy >/dev/null 2>&1 || ls "$2"/*.npz >/dev/null 2>&1; }; then
    SRCS+=("--source" "$1:$2")
    echo "[37] + $1  <- $2"
  else
    echo "[37] - skip $1 (not ready: $2)"
  fi
}
add unet_73                          "$STEP3/results/unet_73_probabilities"
add nnunet_2d_fold0_checkpoint_final "$USS/step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_final_nifti"
add nnunet_2d_fold0_checkpoint_best  "$USS/step1_nnunet_baseline/predictions/nnunet_2d_fold0_checkpoint_best_nifti"
add our_resunet_boundary             "$USS/step4_our_new_model_resunet_boundary/predictions/our_resunet_boundary_2d_fold0_checkpoint_final_nifti"
add our_resunet_boundary_best        "$USS/step4_our_new_model_resunet_boundary/predictions/our_resunet_boundary_2d_fold0_checkpoint_best_nifti"

if [ ${#SRCS[@]} -eq 0 ]; then
  echo "[37] nothing ready" | tee -a "$LOG"
  exit 1
fi
"$PY" "$STEP3/scripts/36_probability_diagnostics.py" "${SRCS[@]}" --split val 2>&1 | tee -a "$LOG"
echo "[37] DONE"

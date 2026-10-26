#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 正式训练 nnU-Net baseline（1000 epoch），跑完自动预测 + 导出 PNG + 评估。
#
# 2026-10-02 重启说明：
#   * zk 环境里的 nnunetv2 2.6.2 是别人装好的定制版，trainer 只在
#     site-packages/nnunetv2/training/nnUNetTrainer 里查找，自定义 trainer
#     无法注册 -> 采用方案A：只用 zk 自带 nnUNetTrainer（PlainConvUNet）。
#   * 训练进程必须能在共享服务器上挺过掉线/被杀：这里用 setsid nohup 完全脱离，
#     并且额外挂一个守护循环：万一训练挂了，自动用 --c 续训（最多 RETRY 次）。
#
# 用法（本脚本自己 setsid 挂后台，调用后立即返回）:
#   nohup setsid bash scripts/run_final_baseline.sh > /dev/null 2>&1 &
# 直接等待式调用（前台看输出）:
#   bash scripts/run_final_baseline.sh
# ---------------------------------------------------------------------------
set -uo pipefail

V="/media/72TB_disk/personal/zk/USSnet"
S1="$V/step1_nnunet_baseline"
LOG="$S1/logs/99_final_baseline_driver.log"
TRAINER="nnUNetTrainer"
TRAIN_LOG="$S1/logs/03_train_fold0_${TRAINER}_zk.log"
MAX_RETRY="${MAX_RETRY:-6}"

mkdir -p "$S1/logs"
exec >> "$LOG" 2>&1

say() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

say "=============================================================="
say "正式 baseline 开始 (trainer=$TRAINER)"
say "=============================================================="

cd "$S1" || exit 1
source scripts/env.sh

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}"

# ---- 1. 训练（带自动续训守护）------------------------------------------
attempt=0
consec_kill=0
while :; do
  attempt=$((attempt + 1))
  if [[ $attempt -gt $MAX_RETRY ]]; then
    say "!! 已重试 $MAX_RETRY 次仍未完成，放弃。"
    exit 3
  fi

  RESUME=""
  if [[ -f "${CKPT_DIR}/checkpoint_latest.pth" ]]; then
    RESUME="--c"
    say "第 ${attempt} 次启动：发现 checkpoint_latest.pth，断点续训 (--c)"
  else
    say "第 ${attempt} 次启动：从头训练"
  fi

  # 用 setsid 独立进程组，避免继承本脚本的会话
  setsid "$NNUNET_VENV/bin/nnUNetv2_train" "${DATASET_ID}" "${CONFIG}" "${FOLD}" \
         -tr "${TRAINER}" -device cuda ${RESUME} \
         > >(tee -a "$TRAIN_LOG") 2>&1
  rc=$?
  say "训练进程退出码 = ${rc}"

  if grep -q "Training done" "$TRAIN_LOG"; then
    say "检测到 'Training done' -> 训练完成"
    break
  fi

  # 没完成 -> 判断是不是被外部杀
  if [[ $rc -eq 0 ]]; then
    say "!! 退出码 0 但日志无 'Training done'，异常，放弃。"
    exit 2
  fi
  cur_epoch=$(grep -oP '(?<=^Epoch )\d+' "$TRAIN_LOG" | tail -1)
  say "!! 训练在第 ${cur_epoch:-?} epoch 中断（很可能是被外部杀）。准备续训。"
  consec_kill=$((consec_kill + 1))
  sleep 30
done

# ---- 2. 预测 + 导出 PNG + 评估 -----------------------------------------
for CKPT in checkpoint_final.pth checkpoint_best.pth; do
  if [[ "$CKPT" == "checkpoint_final.pth" ]]; then
    TAG="nnunet_2d_fold0_checkpoint_final"
  else
    TAG="nnunet_2d_fold0_checkpoint_best"
  fi
  [[ -f "${CKPT_DIR}/${CKPT}" ]] || { say "跳过 ${CKPT}（不存在）"; continue; }
  say "=== 预测 ${CKPT} -> ${TAG} ==="
  bash scripts/04_predict_fold0.sh "${FOLD}" "${CKPT}" || say "!! 预测 ${CKPT} 失败"
  bash "$V/step2_compare_unet_vs_nnunet/scripts/11_export_nnunet_png_and_evaluate.sh" \
       "$S1/predictions/nnunet_${CONFIG}_fold${FOLD}_${CKPT%.pth}_nifti" "${TAG}" \
       || say "!! 导出/评估 ${TAG} 失败"
done

say "=============================================================="
say "ALL DONE: baseline 训练 + 预测 + 评估 全部结束"
say "=============================================================="

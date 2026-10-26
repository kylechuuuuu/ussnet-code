#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 用 zk 个人虚拟环境重新完整训练 nnU-Net fold 0，然后预测 + 导出 PNG + 评估。
#
# 背景（2026-10-02）：
#   1. 旧训练在第 673/1000 epoch 被外部杀掉（无 traceback / 无 OOM / 无 core）。
#   2. 旧训练跑出来的是 PlainConvUNet（126.56 M 参数, state_dict 里 0 个 residual
#      block），来源环境与本项目脚本不一致，用户决定废弃重跑。
#   3. 之前用的是本次会话新建的 ~/.venvs/nnunet；用户要求改用个人环境
#      /home/ta/anaconda3/envs/zk。
#
#   ⚠ 重要（为什么这里不 --c 续训）：
#      zk 的 nnunetv2 2.6.2 是别人装好的定制版，只从
#      site-packages/nnunetv2/training/nnUNetTrainer 里找 trainer，
#      不认 nnUNet_extTrainer。自定义 trainer 无法注册，所以本脚本
#      只用 zk 自带的 nnUNetTrainer（方案A），不碰 site-packages。
#
# 用法:
#   bash scripts/restart_nnunet_with_zk.sh                # 完整训练+预测+评估
#   SKIP_EVAL=1 bash scripts/restart_nnunet_with_zk.sh    # 只训练
#   SMOKE=1 bash scripts/restart_nnunet_with_zk.sh        # 5 epoch 冒烟测试，
#                                                         # 输出到独立目录，不污染正式结果
#
# 注意: 2.6.2 的 run_training() 签名里没有 num_epochs（只有 num_gpus /
#       export_validation_probabilities / continue_training / ...），
#       所以短测改用官方自带的 nnUNetTrainerBenchmark_5epochs（num_epochs=5），
#       它的输出目录名天然不同，正式训练结果不会被覆盖。
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP1_ROOT="$(pwd)"
USS_ROOT="$(cd .. && pwd)"
source scripts/env.sh          # DATASET_ID / DATASET_NAME / CONFIG / FOLD / nnUNet_* 路径
                               # env.sh 里 NNUNET_VENV 已指向 ~/anaconda3/envs/zk

ZK_ENV="${ZK_ENV:-${NNUNET_VENV}}"
if [[ ! -x "${ZK_ENV}/bin/nnUNetv2_train" ]]; then
  echo "[restart] ERROR: ${ZK_ENV}/bin/nnUNetv2_train 不存在或不可执行" >&2
  exit 1
fi
CLEAN_PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/\.venvs/nnunet/bin' | paste -sd: -)"
export PATH="${ZK_ENV}/bin:${CLEAN_PATH}"
unset PYTHONHOME PYTHONPATH 2>/dev/null || true
mkdir -p logs

echo "=============================================================="
echo "[restart] USS_ROOT        = ${USS_ROOT}"
echo "[restart] 虚拟环境         = ${ZK_ENV}"
echo "  python        : $("${ZK_ENV}/bin/python" -c 'import sys;print(sys.version.split()[0])')"
echo "  nnunetv2      : $("${ZK_ENV}/bin/python" -m pip show nnunetv2 2>/dev/null | sed -n 's/^Version: //p')"
echo "  torch         : $("${ZK_ENV}/bin/python" -c 'import torch;print(torch.__version__, torch.cuda.get_device_name(0))')"
echo "[restart] DATASET         = ${DATASET_ID} (${DATASET_NAME})  CONFIG=${CONFIG}  FOLD=${FOLD}"
echo "[restart] nnUNet_results  = ${nnUNet_results}"
echo "[restart] trainer         = nnUNetTrainer (zk 自带, 方案A)"
echo "=============================================================="

echo "[restart] nvidia-smi:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

# ---- 训练（直接前台跑，外层用 setsid nohup 挂后台）----------------------
# SMOKE=1 时用官方自带的 5-epoch trainer 做冒烟测试；它的输出目录名不同，
# 不会覆盖正式训练结果。
TRAINER="nnUNetTrainer"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAINER="nnUNetTrainerBenchmark_5epochs"
  echo "[restart] SMOKE=1 -> 用 ${TRAINER} 做冒烟测试（5 epoch, 独立输出目录）"
fi
TRAIN_LOG="logs/03_train_fold${FOLD}_${TRAINER}_zk.log"
echo "[restart] 启动训练 -> ${TRAIN_LOG}"

set +e
"${ZK_ENV}/bin/nnUNetv2_train" "${DATASET_ID}" "${CONFIG}" "${FOLD}" \
      -tr "${TRAINER}" -device cuda > >(tee "${TRAIN_LOG}") 2>&1
TRAIN_RC=$?
set -e
echo "[restart] 训练进程退出码 = ${TRAIN_RC}"

if [[ ${TRAIN_RC} -ne 0 ]]; then
  echo "[restart] ERROR: 训练非正常退出，日志尾部:" >&2
  tail -40 "${TRAIN_LOG}" >&2
  exit ${TRAIN_RC}
fi
if ! grep -q "Training done" "${TRAIN_LOG}"; then
  echo "[restart] ERROR: 日志里没有 'Training done'，可能又被外部杀了。尾部:" >&2
  tail -40 "${TRAIN_LOG}" >&2
  exit 2
fi
echo "[restart] 训练正常结束。"

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[restart] SMOKE 冒烟完成。请检查:"
  echo "  - 网络类型应为 PlainConvUNet（zk 自带 planner 的产物）"
  echo "  - epoch 时间、显存占用是否正常"
  echo "[restart] 下一步: 删掉冒烟输出目录后跑正式训练"
  exit 0
fi

# ---- 预测 + 导出 PNG + 评估（final / best 两个 checkpoint）--------------
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[restart] SKIP_EVAL=1 -> 跳过预测与评估"
  exit 0
fi

for CKPT in checkpoint_final.pth checkpoint_best.pth; do
  if [[ "${CKPT}" == "checkpoint_final.pth" ]]; then
    TAG="nnunet_2d_fold0_checkpoint_final"
  else
    TAG="nnunet_2d_fold0_checkpoint_best"
  fi
  echo "[restart] === 预测 ${CKPT} -> ${TAG} ==="
  bash scripts/04_predict_fold0.sh "${FOLD}" "${CKPT}"
  bash "${USS_ROOT}/step2_compare_unet_vs_nnunet/scripts/11_export_nnunet_png_and_evaluate.sh" \
       "${STEP1_ROOT}/predictions/nnunet_${CONFIG}_fold${FOLD}_${CKPT%.pth}_nifti" "${TAG}"
done

echo "[restart] ALL DONE (train + predict + evaluate)"

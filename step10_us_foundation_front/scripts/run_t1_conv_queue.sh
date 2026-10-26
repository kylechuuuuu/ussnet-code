#!/bin/bash
# T1: does the improved recipe (lr 5e-4) rescue the CONV student's shift robustness?
# The historical conv baseline (runs/l2_student_views_200ep): lr 2e-3, w_distill 1,
# rc3/ra4, views 1,2,3, fold0 -> compound_B 63.02 (in_domain 90.22).
# T1a  = same but lr 5e-4            (recipe lever)
# T1b  = exact historical config     (baseline reproduction guard)
# T1c  = lr 5e-4 + --w-distill 0     (teacher necessity at the good lr)
# usage: run_t1_conv_queue.sh <gpu> <a|b|c> [<trial> ...]
set -u
G="$1"; shift
PY=/home/ta/anaconda3/envs/zk/bin/python
S=/media/72TB_disk/personal/zk/USSnet/step10_us_foundation_front
export CUDA_VISIBLE_DEVICES=$G
export PYTHONHASHSEED=0
cd "$S"
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G 2>/dev/null | tr -d ' ')

for TR in "$@"; do
  case "$TR" in
    a) TAG=conv_gd_lr5e4;        EXTRA="";                 NOTE="T1a conv lr5e-4 (views, dist on)";;
    b) TAG=conv_ctl_lr2e3;       EXTRA="";                 NOTE="T1b conv lr2e-3 (historical config)";;
    c) TAG=conv_gd_lr5e4_nodist; EXTRA="--w-distill 0";    NOTE="T1c conv lr5e-4, no distillation";;
    *) echo "unknown $TR"; continue;;
  esac
  if [ "$TR" = "b" ]; then LR=2e-3; else LR=5e-4; fi
  for i in $(seq 1 240); do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$UUID" || true)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
    if [ "${busy:-1}" -eq 0 ] && [ "${mem:-9999}" -lt 1500 ]; then break; fi
    sleep 60
  done
  echo "===== $TR : $NOTE : train $(date -u +%H:%M:%SZ) ====="
  $PY -u scripts/21_train_student.py --epochs 200 --batch 4 --lr $LR \
    --rc 3 --ra 4 --views 1,2,3 --clean-prob 0.4 --tag $TAG $EXTRA 2>&1 \
    | grep -E 'done|Traceback|Error|w_distill' | tail -4

  echo "== $TR: 18 conditions (fold0) =="
  $PY -u scripts/22_eval_student.py --tag $TAG --device cuda:0 2>&1 | grep -E 'Dice|Traceback|Error' | tail -20
  echo "== $TR: held-out =="
  $PY -u scripts/22_eval_student.py --tag $TAG --device cuda:0 \
    --cond blur_45,speckle_50,compound_C --image-root data/shift_extra \
    --tag-out ${TAG}_heldout 2>&1 | grep -E 'Dice|Traceback|Error' | tail -6
  echo "===== $TR DONE $(date -u +%H:%M:%SZ) ====="
done
echo "[t1 queue GPU $G] ALL DONE $(date -u +%H:%M:%SZ)"

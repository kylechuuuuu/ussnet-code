#!/bin/bash
# Uniform 5-split protocol queue: for each fold, wait for this GPU to be free, then
#   train (warmup 20 + val-selected best checkpoint) -> 18 conditions -> held-out
#   -> masks -> boundary arbitration.
# usage: run_fold_queue.sh <gpu> <fold> [<fold> ...]
set -u
G="$1"; shift
PY=/home/ta/anaconda3/envs/zk/bin/python
S=/media/72TB_disk/personal/zk/USSnet/step10_us_foundation_front
SH=/media/72TB_disk/personal/zk/USSnet/step8_domain_shift/data
export CUDA_VISIBLE_DEVICES=$G
cd "$S"
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G 2>/dev/null | tr -d ' ')

for K in "$@"; do
  for i in $(seq 1 480); do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$UUID" || true)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
    if [ "${busy:-1}" -eq 0 ] && [ "${mem:-9999}" -lt 1500 ]; then break; fi
    sleep 60
  done
  TAG=${PREFIX:-l2vit_u}_fold${K}
  echo "===== fold $K on GPU $G : train (warmup=${WARMUP:-20} lr=${LR:-1e-3}) $(date -u +%H:%M:%SZ) ====="
  $PY -u scripts/27_student_narrow_vit.py --epochs 200 --batch 2 --lr ${LR:-1e-3} --warmup ${WARMUP:-20} \
    --embed 256 --depth 16 --rc 6 --ra 8 --views 1,2,3 --clean-prob 0.4 \
    --sel-every 25 --sel-n 12 --fold $K --tag $TAG 2>&1 | grep -E 'sel ep|done' | tail -10

  CK=ckpt_best.pt; [ -f runs/$TAG/ckpt_best.pt ] || CK=ckpt.pt
  echo "== fold $K: 18 conditions (ckpt=$CK) =="
  $PY -u scripts/27_student_narrow_vit.py --eval-only --fold $K --tag $TAG --ckpt $CK \
    --image-root $SH/shift_fold$K --tta --tag-out $TAG 2>&1 | grep -E 'Dice'
  echo "== fold $K: held-out =="
  $PY -u scripts/27_student_narrow_vit.py --eval-only --fold $K --tag $TAG --ckpt $CK \
    --cond blur_45,speckle_50,compound_C --image-root data/shift_extra_fold$K --tta \
    --tag-out ${TAG}_heldout 2>&1 | grep -E 'Dice'
  echo "== fold $K: masks + arbitration =="
  $PY -u scripts/28_export_student_masks.py --tag $TAG --split val --fold $K --device cuda:0 2>&1 | grep -E 'export\]'
  P=$S/predictions/l2vit_$TAG
  cd /media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze
  $PY tools/63_arbitrate.py --split val --out-dir $P/arb_val --tag ${TAG} \
    --base-png-dir $P/base --inner-png-dir $P/cautious --outer-png-dir $P/amplified 2>&1 \
    | grep -E 'raw|free|band_lrn|veto_lrn|contain_lrn|band_lrn_px'
  cd "$S"
  echo "===== fold $K DONE $(date -u +%H:%M:%SZ) ====="
done
echo "[queue GPU $G] ALL DONE $(date -u +%H:%M:%SZ)"

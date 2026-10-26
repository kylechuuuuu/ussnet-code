#!/bin/bash
# Waits for a GPU to go idle, then runs the full per-fold chain for the 5-split protocol.
# usage: run_fold_chain.sh <fold> <gpu>
set -u
K="$1"; G="$2"
PY=/home/ta/anaconda3/envs/zk/bin/python
S=/media/72TB_disk/personal/zk/USSnet/step10_us_foundation_front
SH=/media/72TB_disk/personal/zk/USSnet/step8_domain_shift/data
export CUDA_VISIBLE_DEVICES=$G
cd "$S"

# wait until the GPU is free (<1500 MiB) and no student training is running
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G 2>/dev/null | tr -d ' ')
for i in $(seq 1 240); do
  # per-GPU check: any compute process holding THIS gpu's uuid?
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$UUID" || true)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  if [ "$busy" -eq 0 ] && [ "${mem:-9999}" -lt 1500 ]; then
    echo "[fold$K] GPU $G free after ${i} checks, starting $(date -u +%H:%M:%SZ)"; break
  fi
  sleep 60
done

echo "== fold $K: train =="
$PY -u scripts/27_student_narrow_vit.py --epochs 200 --batch 2 --lr 1e-3 --warmup 0 \
  --embed 256 --depth 16 --rc 6 --ra 8 --views 1,2,3 --clean-prob 0.4 \
  --fold $K --tag l2vit_fold${K}_rc6 2>&1 | grep -vE '^\s*$' | tail -3

echo "== fold $K: 18 conditions =="
$PY -u scripts/27_student_narrow_vit.py --eval-only --fold $K --tag l2vit_fold${K}_rc6 \
  --image-root $SH/shift_fold$K --tta --tag-out l2vit_fold${K}_rc6 2>&1 | grep -E 'Dice'

echo "== fold $K: held-out conditions =="
$PY -u scripts/27_student_narrow_vit.py --eval-only --fold $K --tag l2vit_fold${K}_rc6 \
  --cond blur_45,speckle_50,compound_C --image-root data/shift_extra_fold$K --tta \
  --tag-out l2vit_fold${K}_rc6_heldout 2>&1 | grep -E 'Dice'

echo "== fold $K: masks + arbitration =="
$PY -u scripts/28_export_student_masks.py --tag l2vit_fold${K}_rc6 --split val --fold $K --device cuda:0 2>&1 | grep -E 'export\]'
P=$S/predictions/l2vit_l2vit_fold${K}_rc6
cd /media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze
$PY tools/63_arbitrate.py --split val --out-dir $P/arb_val --tag l2vitfold${K} \
  --base-png-dir $P/base --inner-png-dir $P/cautious --outer-png-dir $P/amplified 2>&1 | grep -E 'raw|free|band_lrn|veto_lrn|contain_lrn|band_lrn_px'
echo "[fold$K] ALL DONE $(date -u +%H:%M:%SZ)"

#!/bin/bash
# Seed-replication for the DC-free low-rank claim:
#   gDa (raw-1024) vs pca4 (centered PCA-4), seed 1, fold3 + fold4.
# Answers: was fold3's +2.2 Dice / +4.7 bdice a seed fluke or fold-specific?
# usage: run_seed_queue.sh <gpu> <trial> [...]   trials: gf3 pf3 gf4 pf4
set -u
G="$1"; shift
PY=/home/ta/anaconda3/envs/zk/bin/python
S=/media/72TB_disk/personal/zk/USSnet/step10_us_foundation_front
SH=/media/72TB_disk/personal/zk/USSnet/step8_domain_shift/data
export CUDA_VISIBLE_DEVICES=$G
export PYTHONHASHSEED=0
cd "$S"
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G 2>/dev/null | tr -d ' ')

for TR in "$@"; do
  case "$TR" in
    gf3) F=3; TAG=l2vit_gDa_fold3_s1;        VD="";                                          DIM=1024;;
    pf3) F=3; TAG=l2vit_dr_pca4_fold3_s1;    VD="--views-dir results/teacher_views_pca4_fold3"; DIM=4;;
    gf4) F=4; TAG=l2vit_gDa_fold4_s1;        VD="";                                          DIM=1024;;
    pf4) F=4; TAG=l2vit_dr_pca4_fold4_s1;    VD="--views-dir results/teacher_views_pca4_fold4"; DIM=4;;
    *) echo "unknown $TR"; continue;;
  esac
  for i in $(seq 1 240); do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$UUID" || true)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
    if [ "${busy:-1}" -eq 0 ] && [ "${mem:-9999}" -lt 1500 ]; then break; fi
    sleep 60
  done
  echo "===== $TR : fold$F seed1 target=$([ -z "$VD" ] && echo raw1024 || echo pca4) : train $(date -u +%H:%M:%SZ) ====="
  $PY -u scripts/27_student_narrow_vit.py --epochs 200 --batch 2 --lr 5e-4 --warmup 20 \
    --embed 256 --depth 16 --rc 6 --ra 8 --views 1,2,3 --clean-prob 0.4 --seed 1 \
    --sel-every 25 --sel-n 12 --fold $F --tag $TAG $VD --teacher-dim $DIM 2>&1 \
    | grep -E 'sel ep|done|Traceback|Error' | tail -8
  CK=ckpt_best.pt; [ -f runs/$TAG/ckpt_best.pt ] || CK=ckpt.pt
  echo "== $TR: 18 conditions =="
  $PY -u scripts/27_student_narrow_vit.py --eval-only --fold $F --tag $TAG --ckpt $CK \
    --image-root $SH/shift_fold$F --tta --tag-out $TAG 2>&1 | grep -E 'Dice'
  echo "== $TR: held-out =="
  $PY -u scripts/27_student_narrow_vit.py --eval-only --fold $F --tag $TAG --ckpt $CK \
    --cond blur_45,speckle_50,compound_C --image-root data/shift_extra_fold$F --tta \
    --tag-out ${TAG}_heldout 2>&1 | grep -E 'Dice'
  echo "===== $TR DONE $(date -u +%H:%M:%SZ) ====="
done
echo "[seed queue GPU $G] ALL DONE $(date -u +%H:%M:%SZ)"

#!/bin/bash
# Core-content reduction sweep: distill the student from k-dim teacher targets.
#   pca{k}  = PCA projection (data-driven, 32/33 scripts)
#   rand64  = Gaussian random projection (JL control)
#   chan64  = first-64 channels (L0 width-frontier control)
# Recipe identical to gDa/gD: fold3, warmup20 / lr5e-4 / 200ep / rc6 ra8 / views 1,2,3 / clean .4
# usage: run_dr_queue.sh <gpu> <trial> [<trial> ...]
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
    pca4)   SPEC=pca4;   DIM=4;;
    pca16)  SPEC=pca16;  DIM=16;;
    pca64)  SPEC=pca64;  DIM=64;;
    pca256) SPEC=pca256; DIM=256;;
    rand64) SPEC=rand64; DIM=64;;
    chan64) SPEC=chan64; DIM=64;;
    *) echo "unknown trial $TR"; continue;;
  esac
  TAG=l2vit_dr_${SPEC}_fold3
  for i in $(seq 1 240); do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$UUID" || true)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
    if [ "${busy:-1}" -eq 0 ] && [ "${mem:-9999}" -lt 1500 ]; then break; fi
    sleep 60
  done
  echo "===== $TR : teacher target = ${SPEC} (dim $DIM) : train $(date -u +%H:%M:%SZ) ====="
  $PY -u scripts/27_student_narrow_vit.py --epochs 200 --batch 2 --lr 5e-4 --warmup 20 \
    --embed 256 --depth 16 --rc 6 --ra 8 --views 1,2,3 --clean-prob 0.4 \
    --sel-every 25 --sel-n 12 --fold 3 --tag $TAG \
    --views-dir results/teacher_views_${SPEC}_fold3 --teacher-dim $DIM 2>&1 \
    | grep -E 'meta|sel ep|done|Traceback|Error' | tail -10

  CK=ckpt_best.pt; [ -f runs/$TAG/ckpt_best.pt ] || CK=ckpt.pt
  echo "== $TR: 18 conditions (ckpt=$CK) =="
  $PY -u scripts/27_student_narrow_vit.py --eval-only --fold 3 --tag $TAG --ckpt $CK \
    --image-root $SH/shift_fold3 --tta --tag-out $TAG 2>&1 | grep -E 'Dice'
  echo "== $TR: held-out =="
  $PY -u scripts/27_student_narrow_vit.py --eval-only --fold 3 --tag $TAG --ckpt $CK \
    --cond blur_45,speckle_50,compound_C --image-root data/shift_extra_fold3 --tta \
    --tag-out ${TAG}_heldout 2>&1 | grep -E 'Dice'
  echo "== $TR: masks + arbitration =="
  $PY scripts/28_export_student_masks.py --tag $TAG --split val --fold 3 --device cuda:0 2>&1 | grep -E 'export\]'
  P=$S/predictions/l2vit_$TAG
  cd /media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze
  $PY tools/63_arbitrate.py --split val --out-dir $P/arb_val --tag $TAG \
    --base-png-dir $P/base --inner-png-dir $P/cautious --outer-png-dir $P/amplified 2>&1 \
    | grep -E 'raw|free|band_lrn|veto_lrn|contain_lrn|band_lrn_px'
  cd "$S"
  echo "===== $TR DONE $(date -u +%H:%M:%SZ) ====="
done
echo "[dr queue GPU $G] ALL DONE $(date -u +%H:%M:%SZ)"

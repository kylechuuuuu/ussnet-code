#!/bin/bash
# Causal-control queue for "do pretrained-weight distillations solve multi-center?".
# Usage: run_causal_queue.sh <gpu> <trial> [<trial> ...]
#   trials: gda | r1 | r2 | r3 | e5   (see case block; each ~70 min incl. eval)
# All runs: fold 3, recipe = gD (warmup20 / lr5e-4 / 200ep / rc6 ra8 / views 1,2,3 / clean .4)
# Reference anchors already on disk: l2vit_gD_fold3 (un-aligned recipe run, historical)
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
    gda) TAG=l2vit_gDa_fold3;    EXTRA="";                                                       NOTE="aligned anchor (same recipe as gD)";;
    r1)  TAG=l2vit_r1nodist_fold3; EXTRA="--w-distill 0";                                        NOTE="R1: no distillation term";;
    r2)  TAG=l2vit_r2scr_fold3;  EXTRA="--scramble-feats";                                       NOTE="R2: scrambled teacher targets";;
    r3)  TAG=l2vit_r3clean_fold3; EXTRA="--clean-prob 1.0";                                      NOTE="R3: clean-only distillation";;
    e5)  TAG=l2vit_e5ufd_fold3;  EXTRA="--views-dir results/teacher_views_ufd_fold3 --teacher-dim 768"; NOTE="E5: UltraFedFM teacher swap";;
    *) echo "unknown trial $TR"; continue;;
  esac
  for i in $(seq 1 240); do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$UUID" || true)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
    if [ "${busy:-1}" -eq 0 ] && [ "${mem:-9999}" -lt 1500 ]; then break; fi
    sleep 60
  done
  echo "===== $TR : $NOTE : train $(date -u +%H:%M:%SZ) ====="
  $PY -u scripts/27_student_narrow_vit.py --epochs 200 --batch 2 --lr 5e-4 --warmup 20 \
    --embed 256 --depth 16 --rc 6 --ra 8 --views 1,2,3 --clean-prob 0.4 \
    --sel-every 25 --sel-n 12 --fold 3 --tag $TAG $EXTRA 2>&1 | grep -E 'meta|sel ep|done|Traceback|Error' | tail -12

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
echo "[causal queue GPU $G] ALL DONE $(date -u +%H:%M:%SZ)"

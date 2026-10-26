#!/bin/bash
# DR follow-up battery: is the gain from CENTERING, from the teacher's centered
# structure, or from dimensionality?  Plus fold4 replication of pca4 + anchor.
#   mean1024 = centered raw (1024d)        -> centering alone?
#   chanc64  = centered first-64 channels  -> any centered sketch?
#   noise64  = zero-mean iid noise target  -> teacher structure needed?
#   rand4    = centered random projection  -> k=4 without PCA?
#   pca4f4   = pca4 replicated on fold4
#   gdaf4    = raw-1024 anchor on fold4
# usage: run_dr2_queue.sh <gpu> <trial> [...]
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
    mean1024) F=3; SPEC=mean1024; DIM=1024; TAG=l2vit_dr_mean1024_fold3;;
    chanc64)  F=3; SPEC=chanc64;  DIM=64;   TAG=l2vit_dr_chanc64_fold3;;
    noise64)  F=3; SPEC=noise64;  DIM=64;   TAG=l2vit_dr_noise64_fold3;;
    rand4)    F=3; SPEC=rand4;    DIM=4;    TAG=l2vit_dr_rand4_fold3;;
    pca4f4)   F=4; SPEC=pca4;     DIM=4;    TAG=l2vit_dr_pca4_fold4;;
    gdaf4)    F=4; SPEC="";       DIM=1024; TAG=l2vit_gDa_fold4;;
    *) echo "unknown trial $TR"; continue;;
  esac
  if [ -n "$SPEC" ]; then VD="--views-dir results/teacher_views_${SPEC}_fold${F}"; else VD=""; fi
  for i in $(seq 1 480); do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$UUID" || true)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
    if [ "${busy:-1}" -eq 0 ] && [ "${mem:-9999}" -lt 1500 ]; then break; fi
    sleep 60
  done
  echo "===== $TR : fold$F target=${SPEC:-raw1024} dim $DIM : train $(date -u +%H:%M:%SZ) ====="
  $PY -u scripts/27_student_narrow_vit.py --epochs 200 --batch 2 --lr 5e-4 --warmup 20 \
    --embed 256 --depth 16 --rc 6 --ra 8 --views 1,2,3 --clean-prob 0.4 \
    --sel-every 25 --sel-n 12 --fold $F --tag $TAG $VD --teacher-dim $DIM 2>&1 \
    | grep -E 'meta|sel ep|done|Traceback|Error' | tail -10

  CK=ckpt_best.pt; [ -f runs/$TAG/ckpt_best.pt ] || CK=ckpt.pt
  echo "== $TR: 18 conditions (fold$F) =="
  $PY -u scripts/27_student_narrow_vit.py --eval-only --fold $F --tag $TAG --ckpt $CK \
    --image-root $SH/shift_fold$F --tta --tag-out $TAG 2>&1 | grep -E 'Dice'
  echo "== $TR: held-out =="
  $PY -u scripts/27_student_narrow_vit.py --eval-only --fold $F --tag $TAG --ckpt $CK \
    --cond blur_45,speckle_50,compound_C --image-root data/shift_extra_fold$F --tta \
    --tag-out ${TAG}_heldout 2>&1 | grep -E 'Dice'
  echo "== $TR: masks + arbitration =="
  $PY scripts/28_export_student_masks.py --tag $TAG --split val --fold $F --device cuda:0 2>&1 | grep -E 'export\]'
  P=$S/predictions/l2vit_$TAG
  cd /media/72TB_disk/personal/zk/USSnet/step6_sandwich_squeeze
  $PY tools/63_arbitrate.py --split val --out-dir $P/arb_val --tag $TAG \
    --base-png-dir $P/base --inner-png-dir $P/cautious --outer-png-dir $P/amplified 2>&1 \
    | grep -E 'raw|free|contain_lrn'
  cd "$S"
  echo "===== $TR DONE $(date -u +%H:%M:%SZ) ====="
done
echo "[dr2 queue GPU $G] ALL DONE $(date -u +%H:%M:%SZ)"

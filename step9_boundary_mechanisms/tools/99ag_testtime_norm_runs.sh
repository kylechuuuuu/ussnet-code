#!/usr/bin/env bash
# 99ag_testtime_norm_runs.sh -- run the test-time normalization battery.
#
# For each preprocessing variant, re-predict the SAME shipped checkpoint
# (SQ2res-w25 = nnUNetTrainerSandwichResencW25, --heads base, mirroring TTA on)
# on in_domain and compound_B.  No training, no new weights.
#
# waits for the ablation run (99ad) to release the GPU first.
set -u
cd /media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms
REPO=/media/72TB_disk/personal/zk/USSnet
PY=~/anaconda3/envs/zk/bin/python
LOG=$REPO/step9_boundary_mechanisms/logs/99ag_norm.log
MODEL=$REPO/step6_sandwich_squeeze/data/nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerSandwichResencW25__nnUNetPlans__2d
IMG=$REPO/step8_domain_shift/data/shift_norm

while pgrep -f 99ad_ablation_runs.sh > /dev/null; do
  echo "[99ag] waiting for 99ad ablation to finish ..." >> "$LOG"
  sleep 30
done

CONDS=""
for v in flat64 flat128 tgcrow pwl histmatch; do
  CONDS="$CONDS,${v}__in_domain,${v}__compound_B"
done
CONDS=${CONDS#,}

echo "[99ag] start $(date)  conditions=$CONDS" >> "$LOG"
$PY $REPO/step8_domain_shift/tools/81_predict_shifts.py \
  --model-folder "$MODEL" \
  --checkpoint checkpoint_best.pth \
  --heads base \
  --image-root "$IMG" \
  --conditions "$CONDS" \
  --split val \
  --tag tn \
  --device cuda:0 >> "$LOG" 2>&1
echo "[99ag] ALL DONE $(date)" >> "$LOG"

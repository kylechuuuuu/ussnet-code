#!/usr/bin/env bash
# Leave-one-out ablation of compound_B (ingredient attribution), SQ2res-w25 base
# head, mirroring TTA on, val 31 -- exactly the settings behind step8's
# sq2shift:base row (compound_B = 78.99 Dice / 22.1 bdice).
# Waits for the 99ab attribution runs to release the GPU first.
set -u
cd /media/72TB_disk/personal/zk/USSnet/step8_domain_shift || exit 1
PY=$HOME/anaconda3/envs/zk/bin/python
M=../step6_sandwich_squeeze/data/nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerSandwichResencW25__nnUNetPlans__2d
ABL=/media/72TB_disk/personal/zk/USSnet/step8_domain_shift/data/shift_abl
CONDS=abA_fix,abB_noshadow,abB_notgc,abB_nogamma,abB_notspeckle,abB_noblur,abB_tgcsign

while pgrep -f 99ab_attribution_runs.sh > /dev/null; do
  echo "[wait] attribution still running: $(date +%H:%M:%S)"
  sleep 30
done
echo "[start] ablation $(date +%H:%M:%S)"

$PY -u tools/81_predict_shifts.py \
  --model-folder $M --checkpoint checkpoint_best.pth --heads base \
  --image-root $ABL --conditions $CONDS --split val \
  --tag ab --device cuda:0

echo "=== ABLATION DONE"

#!/usr/bin/env bash
set -u
REPO=/media/72TB_disk/personal/zk/USSnet
cd $REPO/step9_boundary_mechanisms
LOG=$REPO/step9_boundary_mechanisms/logs/99al_gain_texture.log
MODEL=$REPO/step6_sandwich_squeeze/data/nnUNet_results/Dataset001_UltrasoundProstate/nnUNetTrainerSandwichResencW25__nnUNetPlans__2d
~/anaconda3/envs/zk/bin/python $REPO/step8_domain_shift/tools/81_predict_shifts.py \
  --model-folder "$MODEL" --checkpoint checkpoint_best.pth --heads base \
  --image-root $REPO/step8_domain_shift/data/shift_abl \
  --conditions abC_texture_only,abD_gain_only --split val --tag ab --device cuda:0 >> "$LOG" 2>&1
echo "[99al] ALL DONE $(date)" >> "$LOG"

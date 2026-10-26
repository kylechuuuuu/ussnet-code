#!/usr/bin/env bash
# Attribution runs: is compound_B's collapse a property of the sandwich, of the
# ResEnc+500ep architecture, or of the training budget?
set -u
cd /media/72TB_disk/personal/zk/USSnet/step8_domain_shift || exit 1
PY=$HOME/anaconda3/envs/zk/bin/python
S4=../step4_our_new_model_resunet_boundary/data/nnUNet_results/Dataset001_UltrasoundProstate
S5=../step5_squeeze_boundary/data/nnUNet_results/Dataset001_UltrasoundProstate
S6=../step6_sandwich_squeeze/data/nnUNet_results/Dataset001_UltrasoundProstate
CONDS=in_domain,compound_A,compound_B

echo "=== [1/4] fpB = ResEncUNet + fingerprint loss, 500ep, NO sandwich"
$PY -u tools/81_predict_shifts.py \
  --model-folder $S4/nnUNetTrainerProstateResUNetFPBoundary__nnUNetPlans__2d \
  --tag fpB --conditions $CONDS --split val --device cuda:0

echo "=== [2/4] fpA = stock UNet + fingerprint loss, 500ep, NO sandwich"
$PY -u tools/81_predict_shifts.py \
  --model-folder $S4/nnUNetTrainerProstateFPBoundary__nnUNetPlans__2d \
  --tag fpA --conditions $CONDS --split val --device cuda:0

echo "=== [3/4] sq2a = stock UNet + SANDWICH (3 heads), 500ep"
$PY -u tools/81_predict_shifts.py \
  --model-folder $S6/nnUNetTrainerSandwich__nnUNetPlans__2d \
  --tag sq2a --heads base --conditions $CONDS --split val --device cuda:0

echo "=== [4/4] s200 = stock UNet, 200ep (budget control)"
$PY -u tools/81_predict_shifts.py \
  --model-folder $S5/nnUNetTrainerProstateStock200__nnUNetPlans__2d \
  --tag s200 --conditions $CONDS --split val --device cuda:0

echo "=== ALL DONE"

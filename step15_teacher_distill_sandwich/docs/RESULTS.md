# step15 · results and protocol

All numbers are fold 0 of the shared prostate-ultrasound split:
**71 train / 31 validation**, 2D nnU-Net, patch 640×1024, batch 4,
200 epochs, PolyLR completes at 200, EMA checkpoint selection, single seed.

Metrics: Dice, HD95, bDice@1 / bDice@4 (boundary Dice at 1 px / 4 px bands),
ASSD, volume difference in %. Scored per case with the repository evaluator
(`step2_compare_unet_vs_nnunet/scripts/eval_lib.py` + `step4/.../57_shape_prior_ceiling.py`).

## Main table

| arm | Dice | HD95 | bDice@1 | bDice@4 | ASSD | vol% |
|---|---:|---:|---:|---:|---:|---:|
| baseline sandwich (200 ep) | 95.94 | 31.79 | **32.79** | 66.65 | 5.96 | +0.83 |
| **EXP2 teacher distillation** | **96.05** | **28.57** | 32.58 | 66.38 | **5.63** | **+0.55** |
| Δ vs baseline | +0.11 | −3.23 | −0.21 | −0.27 | −0.33 | −0.28 |

Paired per-case statistics (n = 31, Wilcoxon signed-rank):

| metric | Δ mean (sd) | median Δ | p | cases improved |
|---|---:|---:|---:|---:|
| Dice | +0.11 (0.87) | −0.01 | 0.96 | 15/31 |
| HD95 | −3.23 (10.72) | −1.27 | 0.10 | 18/31 |
| bDice@1 | −0.21 (4.37) | −0.16 | 0.71 | 14/31 |
| ASSD | −0.33 (1.51) | −0.06 | 0.65 | 17/31 |

**Nothing is statistically significant.** The HD95 mean gain is carried by a
few outlier slices; the median gain is 1.3 px.

## Context: the other arms run the same night

| arm | Dice | HD95 | bDice@1 | bDice@4 | note |
|---|---:|---:|---:|---:|---|
| baseline | 95.94 | 31.79 | 32.79 | 66.65 | step-6 sandwich |
| EXP1 frozen prior (PCA8+proto+edge, 10 input channels) | 95.80 | 31.47 | 31.52 | 65.60 | no gain |
| EXP1b proto+edge only (2 channels) | 96.02 | 31.03 | 32.02 | 67.02 | ≈ baseline |
| **EXP2 distillation (this folder)** | **96.05** | **28.57** | 32.58 | 66.38 | best training arm |
| reference: dual-encoder concat at stage 4 | 95.64 | 30.90 | 31.57 | 64.98 | worse than baseline |

Post-hoc **BCC** (`p' = p_base + 0.25·(p_amp − p_cau)`, no retraining) applied
on the distilled model gives Dice 96.06 / HD95 28.04 / bDice@1 32.46 — again a
boundary-only, sub-0.1-Dice effect. BCC code is not part of this folder.

Negative control (standing for "do not hard-snap boundaries"): a band random
walker using the same teacher features drops Dice to 94.87 and bDice@1 to 12.5.

## Interpretation

- The dataset is saturated at ~95.9 Dice; the interesting movement is in
  boundary metrics, and it is small.
- Distilling frozen features improves HD95/ASSD and volume bias, i.e. it makes
  the *worst* boundary cases slightly better, without moving the median Dice.
- bDice@1/4 do not improve, which is consistent with the teacher's stride-16
  features being too coarse for 1–2 px boundary corrections.
- A student network is unnecessary here: the only trainable model is the
  sandwich; the teacher adds 0.9 M adapter parameters and no inference-time
  re-training.

## Caveats

1. n = 31, single fold, single seed, p > 0.05 everywhere.
2. Teacher and student share the same foundation model family; the "teacher"
   is not an independent source of supervision.
3. The teacher is applied to augmented batches, so the target is augmentation-
   dependent; this was not ablated.
4. HD95 on 31 cases is sensitive to a handful of slices; report medians too.

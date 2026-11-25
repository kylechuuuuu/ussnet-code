# step15 · Main model: sandwich + frozen-teacher distillation

This is the current **main segmentation model** of the project:

> **stock 2D sandwich U-Net** + **feature-level distillation from a frozen
> ultrasound foundation model** (UltraFedFM ViT-B/16), with **no student
> network** and **no test-time BCC by default**.

The teacher never predicts masks. It supplies dense stride-16 features that
modulate the sandwich encoder (zero-initialised FiLM) and an auxiliary PCA-16
target (cosine loss). The only trainable network is the sandwich itself.

```
                 frozen UltraFedFM ViT-B/16  (stride 16, 768-d, no grad)
                              │
                    ┌─────────┴──────────┐
                    │                    │
            1×1 proj → γ,β         PCA-16 target
                    │                    │
   image ──► sandwich encoder stage 4 ──► h' = h·(1+tanh γ) + β
                    │                    │
                    │              1×1 aux head ──► cosine loss (w = 0.05)
                    ▼
             stock decoder + 6-ch sandwich head ──► L_sandwich
```

## Results

### In-domain (prostate, fold 0, 71/31, 200 ep)

| arm | Dice | HD95 | bDice@1 | bDice@4 | ASSD | vol% |
|---|---:|---:|---:|---:|---:|---:|
| baseline sandwich | 95.94 | 31.79 | **32.79** | 66.65 | 5.96 | +0.83 |
| **+ teacher distillation** | **96.05** | **28.57** | 32.58 | 66.38 | **5.63** | **+0.55** |

### Cross-dataset quick benchmarks (single seed, 100 epochs)

Same protocol for both arms: 2D nnU-Net, patch auto-planned, 100 epochs,
class-stratified random split, test = validation fold.

| benchmark | train / test | model | Dice | HD95 | ASSD | bDice@1 | bDice@4 | vol% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DDTI (nodule) | 75 / 25 | nnU-Net | **69.91** | 58.72 | 18.02 | 11.09 | 30.64 | +82.5 |
| | | **ours** | 69.88 | **57.37** | **17.46** | **13.14** | **33.81** | **+71.4** |
| BUSI (breast lesion) | 75 / 25 | nnU-Net | 71.62 | 108.80 | 33.65 | 14.09 | 39.47 | +103.9 |
| | | **ours** | **76.83** | **73.17** | **24.34** | **15.20** | **40.47** | **−10.4** |
| TG3K (thyroid gland) | 60 / 20 | nnU-Net | 85.59 | 27.18 | 5.11 | 45.91 | 75.81 | +14.5 |
| | | **ours** | **87.77** | **20.38** | **4.07** | **47.65** | **77.67** | **+9.6** |

Per-case win rates against nnU-Net: DDTI 16/25, BUSI 12/25, TG3K 14/20.
The BUSI gain is outlier-driven (median ΔDice = 0.00; a few catastrophic
over-segmentations are repaired); TG3K is the most consistent result
(six metrics better, median ΔDice +0.14).

## BCC is optional, and off by default

`p' = p_base + α·(p_amplified − p_cautious)` (α = 0.25) is kept as an optional
post-processing tool in `tools/bcc.py`. It is **not** part of the main model:

| benchmark | base Dice | +BCC Dice | base HD95 | +BCC HD95 | verdict |
|---|---:|---:|---:|---:|---|
| prostate | 95.94 | 95.94 | 33.36 | 32.09 | HD95/boundary only |
| BUSI | 76.83 | 77.15 | 73.17 | 75.39 | small Dice gain |
| DDTI | 69.88 | 69.83 | 57.37 | 57.80 | slightly negative |
| TG3K | 87.77 | 87.70 | 20.38 | 20.42 | slightly negative |

It expands the boundary and therefore helps only when the baseline massively
over-segments; on well-fitted cases it costs bDice and volume. Do not enable it
globally.

## What is trainable

| module | params | note |
|---|---:|---|
| sandwich U-Net (stock) | ~46 M | the only segmentation model |
| `proj` 768→C (C = 512) | 0.39 M | teacher → stage-4 width |
| `film_g`, `film_b` C→C | 0.52 M | zero-init ⇒ training starts as baseline |
| `aux_head` C→16 | 0.008 M | predicts teacher PCA-16, training-only loss |

The teacher (86 M) stays frozen, in `eval()` mode, under `torch.no_grad()`.

## Run

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export US_FM_CHECKPOINT=/path/to/UltraFedFM_checkpoint.pth
source scripts/env_step15.sh

# 1) fit the frozen PCA-16 basis + prototypes on the fold-0 TRAIN split
python scripts/01_fit_prior_stats.py --dataset Dataset001_UltrasoundProstate --fold 0

# 2) make nnunetv2 2.6.2 resolve -tr nnUNetTrainerSandwichDistill
bash scripts/70_install_trainer_shim.sh

# 3) train
bash scripts/71_train_distill.sh 0 200

# 4) three heads + optional BCC (not used by the main model)
python scripts/72_predict_three_heads.py --dataset Dataset001_UltrasoundProstate \
    --trainer nnUNetTrainerSandwichDistill --fold 0 --out heads/run
python tools/bcc.py --heads heads/run/base --cautious heads/run/cautious \
    --amplified heads/run/amplified --out heads/run/bcc
```

## Files

| path | contents |
|---|---|
| `trainer/us_distill.py` | frozen teacher + `DistillUNet` (FiLM + aux head) |
| `trainer/nnUNetTrainerSandwichDistill.py` | trainer; adds the cosine term to the stock `train_step` |
| `trainer/nnUNetTrainerSandwich.py` | step-6 sandwich trainer (unchanged, required base) |
| `trainer/sandwich_{net,config,losses}.py` | step-6 dependencies (unchanged) |
| `scripts/01_fit_prior_stats.py` | fits PCA basis + prototypes on the train split |
| `scripts/70_install_trainer_shim.sh` | registers the trainer with nnunetv2 2.6.2 |
| `scripts/71_train_distill.sh` | launch script |
| `scripts/72_predict_three_heads.py` | predict base/cautious/amplified (+ `.npz` probs) |
| `tools/bcc.py` | optional band-consistent calibration (off by default) |
| `tools/eval_pred_dir.py` | score a prediction directory against a fold's GT |
| `docs/RESULTS.md` | full protocol, all arms, paired statistics, caveats |
| `metrics/*.json` | result JSONs behind the tables (no data) |

## What this is not

- Not a student: no second segmentation network, no student logits, no student
  checkpoint, no offline teacher cache.
- Not the step-10 line: `step10_us_foundation_front` trains a 13.33 M narrow-ViT
  student from a 841.7 M teacher; this step distils features *into* the
  sandwich decoder.
- Not the dual-encoder arm: concatenating teacher features at stage 4
  (`nnUNetTrainerSandwichDualEnc`) scored 95.64 Dice, below the baseline.

## Provenance

`trainer/sandwich_*.py` and `trainer/nnUNetTrainerSandwich.py` are byte-identical
copies of `step6_sandwich_squeeze/trainer/*` and are included so the folder is
self-contained.

# step15 · Teacher-feature distillation into the sandwich model (no student)

Code for the feature-level distillation arm that was run on the shared
prostate-ultrasound split (fold 0, 71 train / 31 val, 2D, 200 epochs).

The point of this step is **not** to train a student network. The only
trainable model is the step-6 sandwich U-Net. A frozen ultrasound foundation
model (UltraFedFM MAE ViT-B/16) is used **online** as a teacher that supplies
dense high-dimensional features; the sandwich learns to absorb them through a
zero-initialised FiLM modulation and an auxiliary cosine loss.

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

Nothing is concatenated, no second network is trained, no teacher cache is
written to disk.

## What is trainable

| module | params | note |
|---|---:|---|
| sandwich U-Net (stock) | ~30 M | the only segmentation model |
| `proj` 768→C (C = 512) | 0.39 M | teacher → stage-4 width |
| `film_g`, `film_b` C→C | 0.52 M | zero-init ⇒ training starts as baseline |
| `aux_head` C→16 | 0.008 M | predicts teacher PCA-16, training-only loss |

The teacher (86 M) stays frozen, in `eval()` mode, under `torch.no_grad()`.

## Results (fold 0, n = 31, 200 epochs)

| arm | Dice | HD95 | bDice@1 | bDice@4 | ASSD | vol% |
|---|---:|---:|---:|---:|---:|---:|
| baseline sandwich | 95.94 | 31.79 | **32.79** | 66.65 | 5.96 | +0.83 |
| **+ teacher distillation** | **96.05** | **28.57** | 32.58 | 66.38 | **5.63** | **+0.55** |

Paired Wilcoxon (per case, n=31): Dice Δ +0.11 (p = 0.96),
HD95 Δ −3.23 (p = 0.10), ASSD Δ −0.33. Differences are **not significant**;
the HD95 gain is driven by a few outlier cases (median Δ −1.27 px).
Treat this as "boundary is steadier, Dice is saturated", not as a win.

Full protocol, the ablations (frozen prior, proto-only, BCC post-processing,
band random-walker negative control) and the caveats are in `docs/RESULTS.md`.

## Requirements

- Python 3.11, `torch 2.7.1+cu118`, `timm`, `nnunetv2==2.6.2`
- UltraFedFM checkpoint (`UltraFedFM_checkpoint.pth`) — **not included**
- The prostate ultrasound dataset in nnU-Net v2 layout — **not included**
- A fold-0 split file produced by nnU-Net (`splits_final.json`)

Weights, data, and the fitted prior statistics are deliberately absent from
this folder. `.gitignore` blocks them.

## Run

```bash
# 0) environment
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export US_FM_CHECKPOINT=/path/to/UltraFedFM_checkpoint.pth
source scripts/env_step15.sh

# 1) fit the frozen PCA-16 basis + foreground/background prototypes
#    (fold-0 TRAIN split only; writes prior_stats.npz)
python scripts/01_fit_prior_stats.py --dataset Dataset001_UltrasoundProstate --fold 0

# 2) make nnU-Net 2.6.2 able to resolve -tr nnUNetTrainerSandwichDistill
bash scripts/70_install_trainer_shim.sh

# 3) train
bash scripts/71_train_distill.sh 0 200      # gpu 0, 200 epochs
```

The built-in end-of-training validation writes `validation/`; score it with the
repository's `eval_val.py` (step 2/6 tools) to reproduce the table above.

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
| `scripts/env_step15.sh` | path/config environment |
| `docs/RESULTS.md` | protocol, all arms, paired statistics, caveats |
| `metrics/*.json` | the two result JSONs behind the table (no data) |

## What this is not

- Not a student: there is no second segmentation network, no student logits,
  no student checkpoint, no offline teacher cache.
- Not the step-10 distillation line: `step10_us_foundation_front` trains a
  13.33 M narrow-ViT student from a 841.7 M teacher; this step distils
  features *into* the sandwich decoder instead.
- Not the step-15 dual-encoder arm: that one concatenates teacher features at
  stage 4 (`nnUNetTrainerSandwichDualEnc`), which scored 95.64 Dice here.

## Provenance

The `trainer/sandwich_*.py` and `trainer/nnUNetTrainerSandwich.py` files are
byte-identical copies of `step6_sandwich_squeeze/trainer/*` (same MD5) and are
included so this folder is self-contained. Citations and design notes for the
sandwich objective live in that step.

# USSnet

Research code for **prostate ultrasound segmentation**: a 13.33 M-parameter narrow-ViT
student distilled from an 841.7 M ultrasound foundation model, trained with **DC-free
low-rank teacher features** and decoded through a **"sandwich" 6-channel boundary head +
boundary-evidence arbitration**. Evaluated against nnU-Net and CNN baselines on repeated
71/31 splits, with 20 acquisition-condition shifts and 3 held-out conditions.

This repository is a **code-only export**: no raw data, no model weights, no teacher
caches, no prediction masks. It contains scripts, configs, reports and small result tables.

## Layout

| Path | Contents |
|---|---|
| `step1_nnunet_baseline/` | nnU-Net reproduction on the shared split |
| `step2_compare_unet_vs_nnunet/` | Unified evaluator (`eval_lib.py`) and metric tables |
| `step3_failure_analysis/` | Error decomposition, boundary-failure anatomy |
| `step4_our_new_model_resunet_boundary/` | ResEnc + boundary objectives (start of the large-model line) |
| `step5_squeeze_boundary/` | Early boundary-compression experiments |
| `step6_sandwich_squeeze/` | **Core:** 6-channel sandwich head + arbitration |
| `step7_band_arbitration/` | Band / arbitration mechanism analysis |
| `step8_domain_shift/` | 17 pre-registered acquisition operators + 3 held-out conditions |
| `step9_boundary_mechanisms/` | Gating/fallback, two-model band, TTA, final delivery |
| `step10_us_foundation_front/` | Teacher caching, DC-free low-rank distillation, narrow-ViT student |
| `step11_model_summary/` | Paper-level model and claim summary |
| `step12_lowrank_decomp/` | Low-rank refiner (negative result) and no-TTA comparison |
| `unet/` | Original U-Net baseline (code and config only) |

## Start here

- `step11_model_summary/README.md` and `EVIDENCE.md` — model overview and evidence table
- `step12_lowrank_decomp/REPORT.md` — no-TTA delivery-line comparison
- `CODE_README.md` — full Chinese write-up: verified claims, negative results, exact numbers

## Environment

Python 3.11, PyTorch 2.7.1+cu118, CUDA 11.8. See `requirements_code.txt`.

```bash
uv venv --python 3.11
uv pip install -r requirements_code.txt
```

The teacher branch additionally needs the internal `sam3` (UltraSAM3 ViT-Det) and `sam2`
packages, which are not on PyPI; install them from the original UltraSAM3 environment.
The nnU-Net baseline needs `nnunetv2==2.6.2`.

## Running

Data and weights are **not** included. You need the prostate ultrasound dataset, the
UltraSAM3 / UltraFedFM checkpoints, and a teacher cache. Weight fetcher:
`step10_us_foundation_front/scripts/00_fetch_weights.sh`.

```bash
# 1) teacher view cache (requires UltraSAM3)
python step10_us_foundation_front/scripts/23_cache_teacher_views.py --fold 3

# 2) PCA + DC-free low-rank targets
python step10_us_foundation_front/scripts/32_pca_teacher.py --fold 3
python step10_us_foundation_front/scripts/33_make_reduced_caches.py --fold 3 \
  --methods pca4,pca16,pca64,pca256,rand64,chan64,chanc64,mean1024,noise64,rand4

# 3) narrow-ViT student (k=4, mean-removed)
python step10_us_foundation_front/scripts/27_student_narrow_vit.py \
  --fold 3 --teacher-dim 4 --views-dir results/teacher_views_pca4_fold3 --epochs 200

# 4) boundary arbitration (large-model line, mask mode)
python step6_sandwich_squeeze/tools/63_arbitrate.py \
  --base-png-dir step6_sandwich_squeeze/predictions/sq2resencw25_checkpoint_best_arb_val/raw \
  --split val --out-dir /tmp/arb --tag sq2_val
```

Batch queues: `run_dr_queue.sh`, `run_dr2_queue.sh`, `run_causal_queue.sh`,
`run_seed_queue.sh` under `step10_us_foundation_front/scripts/`.

> **Paths:** scripts still contain absolute paths from the original cluster
> (`/media/72TB_disk/personal/zk/USSnet`). Search and replace `USS` / `ROOT` / `GT_DIR`
> before running elsewhere.

## Reading the numbers

`CODE_README.md` §6 lists the caveats that matter: the five "folds" are repeated random
71/31 splits (not 5-fold CV), the L0 probe is a frozen-feature 5-NN (not comparable to
segmentation Dice), the student was not run with sliding-window inference or TTA, and the
delivery-line `bdice` gain comes mostly from arbitration/post-processing rather than from
the model. `CODE_README.md` §2 lists the approaches that did **not** work.

## License

No license file is included yet, so the default "all rights reserved" applies. Contact the
author before reuse.

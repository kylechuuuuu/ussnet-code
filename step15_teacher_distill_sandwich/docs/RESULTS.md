# step15 · results and protocol

The main model is **sandwich + frozen-teacher distillation**, evaluated with the
same protocol as the nnU-Net baseline on every benchmark below.

## Protocols

| benchmark | split | images | epochs | notes |
|---|---|---:|---:|---|
| prostate (in-domain) | official fold 0 | 71 / 31 | 200 | 2D TRUS, 1 mm |
| DDTI | class-stratified random | 75 / 25 | 100 | thyroid nodule, PNG→NIfTI |
| BUSI | class-stratified random | 75 / 25 | 100 | breast lesion |
| TG3K | random (no labels) | 60 / 20 | 100 | thyroid gland |

All arms: 2D nnU-Net, patch size auto-planned, batch size auto, PolyLR completes
at the configured epoch count, EMA checkpoint selection, single seed. "test" is
the validation fold used for checkpoint selection, for both arms.

## In-domain: prostate, fold 0 (n = 31)

| arm | Dice | HD95 | bDice@1 | bDice@4 | ASSD | vol% |
|---|---:|---:|---:|---:|---:|---:|
| baseline sandwich | 95.94 | 31.79 | **32.79** | 66.65 | 5.96 | +0.83 |
| **+ teacher distillation** | **96.05** | **28.57** | 32.58 | 66.38 | **5.63** | **+0.55** |
| Δ | +0.11 | −3.23 | −0.21 | −0.27 | −0.33 | −0.28 |

Paired Wilcoxon (n = 31): Dice p = 0.96, HD95 p = 0.10. The HD95 gain is driven
by a few outlier slices (median ΔHD95 = −1.27 px).

## Cross-dataset quick benchmarks

| benchmark | model | Dice | HD95 | ASSD | bDice@1 | bDice@4 | vol% |
|---|---|---:|---:|---:|---:|---:|---:|
| DDTI (n=25) | nnU-Net | **69.91** | 58.72 | 18.02 | 11.09 | 30.64 | +82.5 |
| | **ours** | 69.88 | **57.37** | **17.46** | **13.14** | **33.81** | **+71.4** |
| BUSI (n=25) | nnU-Net | 71.62 | 108.80 | 33.65 | 14.09 | 39.47 | +103.9 |
| | **ours** | **76.83** | **73.17** | **24.34** | **15.20** | **40.47** | **−10.4** |
| TG3K (n=20) | nnU-Net | 85.59 | 27.18 | 5.11 | 45.91 | 75.81 | +14.5 |
| | **ours** | **87.77** | **20.38** | **4.07** | **47.65** | **77.67** | **+9.6** |

Per-case (ours − nnU-Net):

| benchmark | mean ΔDice | median ΔDice | cases better |
|---|---:|---:|---:|
| DDTI | −0.03 | +0.88 | 16/25 |
| BUSI | +5.21 | 0.00 | 12/25 |
| TG3K | +2.18 | +0.14 | 14/20 |

Reading: TG3K is the most consistent result (all six metrics better, 14/20
cases). BUSI's mean gain is outlier-driven: a few baseline cases over-segment
catastrophically (HD95 > 100, volume +100%) and the sandwich model repairs them,
while the median case is unchanged.

## Optional BCC (off by default)

`p' = p_base + α·(p_amplified − p_cautious)`, α = 0.25, computed on the three
sandwich heads (`tools/bcc.py`). It expands the boundary and is therefore only
useful when the baseline over-segments.

| benchmark | base Dice | +BCC Dice | base HD95 | +BCC HD95 | verdict |
|---|---:|---:|---:|---:|---|
| prostate | 95.94 | 95.94 | 33.36 | 32.09 | boundary/HD95 only |
| BUSI | 76.83 | 77.15 | 73.17 | 75.39 | small Dice gain |
| DDTI | 69.88 | 69.83 | 57.37 | 57.80 | slightly negative |
| TG3K | 87.77 | 87.70 | 20.38 | 20.42 | slightly negative |

Conclusion: **do not enable BCC globally.** If it is kept at all, gate it by an
abnormal predicted volume (e.g. `V_pred > 1.5 × V_train_median`) so it only acts
on catastrophic over-segmentation.

## Caveats

1. n = 20–31 per benchmark, single fold, single seed; differences below ~0.5
   Dice are not meaningful.
2. The cross-dataset runs use small training sets (60–75 images) and 100 epochs;
   they are sanity checks, not final paper numbers.
3. Test = validation fold for checkpoint selection, for both arms equally.
4. Teacher and model share the same foundation-model family; the teacher is not
   an independent source of supervision.
5. All cross-dataset numbers use a single τ/α configuration; no per-dataset
   tuning was done.

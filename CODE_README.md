# USSnet — 超声前列腺分割：边界三明治 × DC-free 低秩蒸馏

> 代码包（2026-10-26）。本包只含**代码、配置、报告、小型结果表**；不含原始数据、模型权重、teacher 缓存、预测掩膜和大图。
> 原始项目根目录：`/media/72TB_disk/personal/zk/USSnet`。代码里仍有该绝对路径，迁移到新机器时需改 `USS`/`ROOT`/`GT_DIR` 等常量。

---

## 0. 一句话

一个 **13.33 M 的窄 ViT 学生**（窗口注意力 + RoPE，63× 压缩自 841.7 M 超声基础模型），
训练时用**去均值低秩教师特征**做蒸馏，推理时用**三明治 6 通道头 + 边界证据仲裁**把边界推准，
并用 **GT-free 软度统计**做监测/回退；大模型交付线（SQ2res-w25 + 仲裁 + 后处理）在**无 TTA** 下
四项指标均略优于 nnU-Net。

---

## 1. 已验证的创新点（可以作为论文主张）

### 1.1 DC-free 低秩蒸馏：可迁移的是“去均值后的低秩结构”，不是全维特征

- **全维 cosine 蒸馏 ≈ 空转**：token 与全局均值的平均余弦 **0.54**，DC 能量占比 **26.8%**；
  raw-1024 蒸馏与“无教师”在 fold3 上几乎相同（gDa 90.15/89.83 vs R1 89.82/89.51，Dice/compound_B）。
- **去均值解锁**：`(X − μ) @ W_k`，k=4/16/64/256（PCA）、随机投影、中心化通道子集都可用；
  fold3 s0：PCA-4 **92.39 / 20.34 / 91.81**，raw-1024 90.15/15.63/89.83，mean1024 91.58/18.95/91.12。
- **低秩是交付方式，不是唯一关键**：k≥16 时 PCA / 随机 / 通道子集等价；k=4 时 PCA 明显优于 random-4
  （bdice +4.34，p<0.001，单折）。**“4 维就够”只有 fold3 单种子证据，不能当稳健结论。**
- **跨折/种子**：2 折 × 2 种子四对平均 **in +1.04 Dice / bd1 +2.37 / cB +0.86**；
  方向 4/4、3/4、3/4 为正，但 n=4、p≈0.10–0.15；fold4 s0 的 raw 增益完全消失。
  **诚实口径**：小到中等、方向一致的增益；fold3-s0 的 +2.2/+4.7 是上界，不是典型值。

### 1.2 多中心“平坦性”由架构决定，不由教师知识决定

- 6 臂因果电池（fold3，同配方）：删教师（R1）、打乱内容（R2）、只蒸干净图（R3）、换 UltraFedFM（E5）
  全部落在 89.5–90.7 的噪声带内；唯一移动数字的是优化配方（lr 1e-3→5e-4，+5 Dice）。
- 卷积学生两种配方 **compound_B Δ≈−26**；窄 ViT 学生所有配置 **Δ≈−0.5**；
  20 个采集条件折内落差 ≤1.7 Dice。
- **结论**：教师提供精度/边界质量，**不提供多中心鲁棒性**；鲁棒性是架构/预训练域属性。

### 1.3 三明治边界结构：编辑窗口 + 仲裁，不是不确定性估计

- 一个 backbone → 6 通道头：`base / cautious=erode3 / amplified=dilate4`，嵌套由构造保证；
  区间 `band = amplified \ cautious` 是**编辑窗口**，不是 UQ。
- **UQ 已被三类独立构造证否**：三头 cover 0.679 vs 同宽 0.7135（Δ−0.034）；双模型 cover 0.084–0.110；
  视图抖动 Δ−0.223；区间宽度与误差 ρ=+0.074。
- **编辑增益不变性**：δ = **+3.7~4.6 bdice** 在 9 个训练臂上为正；5/5 折改善；
  E1 条件维 **60/60 个“臂×条件”δ>0**（3 臂 × 20 条件）。
- **GT-free 监测/回退**：`conf_margin` 对 gross failure AUROC **0.989–0.990**；`ent_loc` 逐例 ρ=+0.452（31/31）；
  compound_B **78.99 → 93.42**，in-domain **0/31 误报**。

### 1.4 无 TTA 交付线（主表口径）

统一后处理：`median3 → binary opening 5×5 → largest CC → fill holes`；仲裁参数预登记（frac 0.7 / maxstep 4 / sigma 1.5 / ±16 px）。

| 模型 | Dice | bdice@1 | HD95 | ASSD |
|---|---|---|---|---|
| nnU-Net raw | 96.106 | 35.162 | 30.948 | 5.605 |
| nnU-Net + free + post | 96.103 | 36.721 | 28.817 | 5.392 |
| SQ2res-w25 raw | 96.381 | 35.330 | 29.667 | 5.289 |
| **SQ2res-w25 + veto + post** | **96.388** | **37.057** | **27.735** | **5.090** |

- vs nnU-Net raw：Dice **+0.282**（Wilcoxon p=0.034）、bdice **+1.896**（p=0.041）、HD95 −3.213、ASSD −0.514。
- vs **nnU-Net + 同样仲裁 + 后处理**（最强基线）：Dice **+0.285**（p=0.043）、bdice +0.337（ns）、
  HD95 −1.082（ns）、ASSD −0.302（ns）。
- **诚实读法**：全套流程四项都略优于 nnU-Net；但 bdice 的 +1.9 主要来自仲裁/后处理，
  模型本身的 raw 增益只有 Dice +0.28 / bdice +0.17。**不要把这 +1.9 全部归给模型。**
- **TTA 版本（bdice 37.30）不作为主表**：其他模型没有用 TTA，口径不一致。

### 1.5 压缩/部署

- 13.33 M 学生（63×）in-domain **92.4**、compound_B **91.8**，20 条件落差 ≤1.7；
  已接近 UltraSAM3 冻结特征 5-NN 的可分性上限（92.8/91.3）。
- 学生 in-domain 仍比 nnU-Net 低约 4 点：**压缩 + 多中心 ≠ 更高精度**；论文应把精度与鲁棒性分开声明。

---

## 2. 未验证 / 负结果（不要重复投入）

| 方向 | 结果 |
|---|---|
| 学习式低秩边界 refiner（step12） | 5 折 CV：Dice **−0.052**、bdice **−1.074**、HD95 +0.043、ASSD +0.062；不迁移 |
| 三头区间当 UQ | cover 0.679 < 同宽 0.7135；尾部覆盖 14.6% |
| 双模型内/外圈区间 | cover 0.084–0.110，Δ −0.107~−0.145 |
| 测试时归一化 | flat64: 78.99→53.52 |
| 距离加权边界损失 | −1.105 bdice |
| 宽窗吸附 / 区间宽度修正 | ±40px 窗 17.6（−17.7）；24 变体全负 |
| 全维 cosine 蒸馏 | ≈ 无教师（DC 主导） |
| 多教师共识/融合 | CKA 族内 0.72；换教师无差（E5） |
| ImageNet 前端 | compound_B 55.26、27/31 <80 |
| “4 维最优 / 低秩分解网络 beat nnU-Net” | **不成立**；当前无此证据 |

---

## 3. 代码地图

```
USSnet/
├── README.md                         项目总览（step1…step4 的原始说明）
├── step1_nnunet_baseline/            nnU-Net 复现（同划分 71/31）
├── step2_compare_unet_vs_nnunet/     统一评测器 eval_lib.py + 指标
├── step3_failure_analysis/           失败原因 / 边界误差解剖
├── step4_our_new_model_resunet_boundary/  ResEnc + 边界目标（大模型线起点）
├── step5_squeeze_boundary/           边界压缩/三明治早期实验
├── step6_sandwich_squeeze/           三明治 6 通道头 + 仲裁（核心）
│   ├── trainer/sandwich_net.py      6 通道头 / tie_seg_heads / 嵌套
│   └── tools/63_arbitrate.py        边界证据仲裁（free/band/veto）
├── step7_band_arbitration/           区间/仲裁机制分析
├── step8_domain_shift/               17 个预登记采集算子 + 3 留出条件
├── step9_boundary_mechanisms/        门控/回退、双模型、TTA、最终交付
├── step10_us_foundation_front/       教师缓存 + DC-free 低秩蒸馏 + 窄 ViT 学生
│   └── scripts/27_student_narrow_vit.py, 32_pca_teacher.py, 33_make_reduced_caches.py
├── step11_model_summary/             论文级模型/创新点总结（先读）
├── step12_lowrank_decomp/            低秩 refiner 负结果 + 无 TTA 对比
└── unet/                             原始 U-Net 基线（仅代码/配置）
```

**建议先读**：`step11_model_summary/README.md`、`step11_model_summary/EVIDENCE.md`、
`step12_lowrank_decomp/REPORT.md`。

---

## 4. 复现（关键命令）

> 数据与权重不在包内。需要 `UltrasoundProstate/`、UltraSAM3/UltraFedFM 权重和 teacher 缓存。
> 权重下载脚本：`step10_us_foundation_front/scripts/00_fetch_weights.sh`。

```bash
# 1) 教师视图缓存（需 UltraSAM3）
python step10_us_foundation_front/scripts/23_cache_teacher_views.py --fold 3

# 2) PCA + 低秩目标缓存（DC-free）
python step10_us_foundation_front/scripts/32_pca_teacher.py --fold 3
python step10_us_foundation_front/scripts/33_make_reduced_caches.py --fold 3 \
  --methods pca4,pca16,pca64,pca256,rand64,chan64,chanc64,mean1024,noise64,rand4

# 3) 窄 ViT 学生（k=4 去均值低秩）
python step10_us_foundation_front/scripts/27_student_narrow_vit.py \
  --fold 3 --teacher-dim 4 --views-dir results/teacher_views_pca4_fold3 --epochs 200

# 4) 边界仲裁（大模型线；mask 模式）
python step6_sandwich_squeeze/tools/63_arbitrate.py \
  --base-png-dir step6_sandwich_squeeze/predictions/sq2resencw25_checkpoint_best_arb_val/raw \
  --split val --out-dir /tmp/arb --tag sq2_val
```

批量队列见 `step10_us_foundation_front/scripts/run_dr_queue.sh`、`run_dr2_queue.sh`、
`run_causal_queue.sh`、`run_seed_queue.sh`。

---

## 5. 环境

- Python 3.11，PyTorch 2.7.1+cu118，CUDA 11.8，8× RTX 4090。
- 依赖：`torch`, `torchvision`, `timm`, `transformers`, `monai`, `scipy`, `scikit-image`,
  `opencv-python`, `pandas`, `numpy`, `Pillow`, `nibabel`, `nnunetv2`（baseline）。
- 代码还依赖内部 `sam3`（UltraSAM3 的 ViT-Det 主干）和 `sam2` 包；不在 pip 里，需按原项目安装。
- 所有路径目前是绝对路径（`/media/72TB_disk/personal/zk/USSnet`），迁移后请全局替换。

---

## 6. 引用口径提醒

1. 5 个“折”是 **repeated random 71/31 splits**，不是 5-fold CV。
2. L0 探针是冻结特征 5-NN，不能与分割 Dice 直接比。
3. 学生 in-domain 与 nnU-Net 不可直接比（无滑窗、无 1000 ep、TTA 状态不同）。
4. 降维扫描是 fold3 单种子；fold4/seed1 已补，但“4 维最优”仍不成立。
5. 仲裁参数预登记，未在 val 上调；后处理是确定性的、无 GT。
6. TTA 版本存在（bdice 37.30），但主表不使用，因为基线未用 TTA。

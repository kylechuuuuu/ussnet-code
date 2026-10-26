# 模型结构（训练 / 推理 / 部署）

> 本文描述模型的组成、数据流、超参与接口约束。数字来自 step10 学生管线与 step6 三明治管线。

---

## 1. 总览图

```
[训练期｜离线]
 图像 (71 张, fold 划分)
   │
   ├─► 17 采集算子族 → 视图采样 (clean / texture+gain / texture / gain / heavy_blur / compound_full)
   │        │
   │        ├─► 教师：UltraSAM3 24-block trunk（841.7M，冻结，1008×1008）
   │        │       └─ 特征 (1024 × h × 72)，h≤54（按图像有效行）
   │        │       └─ 去均值 → PCA/随机投影/通道子集 → 目标 k ∈ {4 … 1024}
   │        │
   │        └─► 学生输入（同一扰动的图，z-score，1008×1008 画布）
   │
   └─► 学生：窄 ViT 13.33M
           ├─ patch14, embed256, depth16, window24, global blocks (3,7,11,15)
           ├─ RoPE + tiled abs-pos
           ├─ 解码器 → stride 2
           ├─ 6 通道 1×1 分割头（base / cautious / amplified）
           └─ 训练期 1×1 投影头（256→k，仅用于余弦蒸馏，推理丢弃）

[推理期｜部署]
 单张图像 → 学生 1 次前向（三个头共享 trunk）→ 3 个 mask
   ├─ base: 直接输出
   ├─ cautious/amplified: 由 tie_seg_heads 绑定的 logit 偏置给出
   ├─ 区间 = amplified \ cautious（集合运算保证嵌套）
   ├─ 边界仲裁（±16px 窗内的图像脊证据）→ 最终边界
   └─ GT-free 监测（conf_margin / ent_loc）→ 编辑开关 / 回退 / 复核
```

---

## 2. 组件清单

| 组件 | 规格 | 是否部署 | 备注 |
|---|---|---|---|
| 教师 | UltraSAM3（SAM3 ViT-Det trunk，32 层取 24 层，宽 1024，patch 14）| ✗ 仅离线 | 841.7 M / 10 GB；深度前沿：24/32 层免费（compound_B 91.02 vs 91.31） |
| 备选教师 | UltraFedFM（MAE ViT-B/16，111.9M）、UltraSam（SAM1 ViT-B，93.7M） | ✗ | 用于教师族对照（E5）；FetalCLIP 为非域负对照 |
| 视图缓存 | 71 图 × 6 视图，`image.npy`（供参考）+ `feat.npy`（蒸馏目标） | ✗ | 视图种子写入 `_meta.json`，训练端按种子**精确重生成**同一实现 |
| 学生 | 窄 ViT 13.33 M（patch14, embed256, depth16, window24, global 3/7/11/15, RoPE） | ✓ | 63× 压缩 |
| 分割头 | 6 通道 1×1（base/cautious/amplified），`tie_seg_heads()` 绑定 | ✓ | 硬约束：网络中不得出现第二个 6 通道 1×1 卷积 |
| 投影头 | 256→k 1×1（训练期） | ✗ | 参数量：k=4 时 1 k；k=1024 时 262 k |
| 仲裁器 | 图像脊证据 + 法向搜索（±16px，步长 0.5，σ=1.5，frac=0.7，maxstep=4） | ✓（可关闭） | 参数预登记于训练集探针，不在 val 上调 |
| 监测器 | conf_margin（0.5 等值面平均 \|2p−1\|）、ent_loc（轮廓 ±2px 熵） | ✓ | 全 GT-free |
| 门控 | conf_margin < τ（τ=0.6234，域内最小值）→ 回退/池化 | ✓（可选） | 域内 0/31 误报；compound_B 16/31 触发 |

---

## 3. 训练细节

| 项 | 值 |
|---|---|
| 数据 | 102 例单中心 TRUS 前列腺；fold 划分，71 训练 / 31 验证 |
| 输入 | 1008×1008 画布（宽度对齐、下方补零、全局 z-score） |
| 监督目标 | base=GT；cautious=erode(GT, 3px)；amplified=dilate(GT, 4px)；半径按 step4 边界指纹角向调制（弱证据扇区 ×[0.5,1.5]） |
| 损失 | DC+CE（三头）+ 嵌套惩罚（cautious≤base≤amplified）+ 深监督（stride 4/8，w=0.5）+ 余弦特征蒸馏（w=1） |
| 蒸馏目标 | 教师特征去均值后 k 维投影（k∈{4,…,1024}）；**PCA / 随机投影 / 中心化通道子集等效** |
| 优化 | AdamW，lr 5e-4，warmup 20 ep，余弦退火至 0，batch 2，200 ep |
| 训练成本 | 单卡 ~45–50 min / 200 ep（13.33 M，1008 画布） |
| 选择 | 干净验证集 12 例（无 TTA）选 best 快照；评测用 `ckpt_best`，掩膜导出用 `ckpt.pt`（项目约定） |
| 随机性 | 视图种子由 `_meta.json` 固定；`PYTHONHASHSEED=0`；单折单种子（复现性见 LIMITATIONS） |

---

## 4. 推理细节

1. **三态头**：一次前向得到 6 通道 logits；`base=[0:2]`、`cautious=[2:4]`、`amplified=[4:6]`；三个头共享 100% 特征，仅最后一层偏置不同。
2. **嵌套**：推理端用集合运算硬保证 `cautious ⊆ base ⊆ amplified`（不依赖损失）。
3. **区间**：`band = amplified \ cautious`。**它不是 UQ 估计**（宽度与误差 ρ=0.07；见 `INNOVATION_2`），而是**编辑窗口**。
4. **仲裁**：对 base 轮廓的每个法向点，在 ±16px 内沿法向找图像脊（σ=1.5 高斯模糊后的梯度响应），取 `frac×max` 以上、最靠近零的峰；`free`（无约束）/`band_lrn`（限制在学到的区间）/`veto_lrn`（概率+证据联合）等变体；参数冻结。
5. **监测/回退**：`conf_margin < 0.6234` 时回退到鲁棒臂（stock 骨干或池化）；`ent_loc` 给出逐点风险带，用于复核/定向标注。
6. **TTA（可选）**：镜像翻转 + 外观视图（jpeg_30/jpeg_15）概率平均；交付配方 `final_tta_art` = 3 视图平均 + median3→open2→fill_holes。

---

## 5. 接口约束（改动时必须遵守）

1. **6 通道 1×1 唯一出口**：`tie_seg_heads()` 要求存在 `out_channels==6, kernel=(1,1)` 的卷积；新前端/新解码器不得引入第二个。
2. **区间半径 = 残差尺度**：`r_c/r_a` 必须按学生实测残差重算（学生残差大于大模型；当前 rc6/ra8 对应 contain 0.51–0.64）。
3. **仲裁参数预登记**：frac/maxstep/sigma/窗口必须在训练集探针上冻结，不在验证集调。
4. **教师不进部署**：任何形式的 842M/112M 权重、缓存、特征都只存在于离线管线。
5. **视图种子**：训练端必须从缓存的 `_meta.json` 读取种子重生成视图；禁止用进程随机 hash（历史 bug，2026-10-19 已修）。

---

## 6. 与项目其它模块的关系

| 模块 | 作用 | 路径 |
|---|---|---|
| 三明治训练器 | 6 通道头 + tie_seg_heads + 嵌套 | `step6_sandwich_squeeze/trainer/sandwich_net.py` |
| 仲裁工具 | 边界证据重定位（free/band/veto/contain） | `step6_sandwich_squeeze/tools/63_arbitrate.py` |
| 算子族 | 17 预登记采集条件 + 3 留出 | `step8_domain_shift/` |
| 门控工具 | conf_margin 回退/池化 | `step9_boundary_mechanisms/tools/99ai/99aj` |
| 学生训练 | 窄 ViT + 视图蒸馏 | `step10_us_foundation_front/scripts/27_student_narrow_vit.py` |
| 教师缓存 | UltraSAM3 24 层视图缓存 | `step10/scripts/23_cache_teacher_views.py` |
| 降维缓存 | PCA/随机/通道/去均值/噪声 | `step10/scripts/32_pca_teacher.py`、`33_make_reduced_caches.py` |
| 因果对照 | R1/R2/R3/E5 六臂 | `step10/scripts/run_causal_queue.sh`、`results/CAUSAL_RESULTS.md` |
| 降维扫描 | pca4…pca256/rand64/chan64 + 机制对照 | `step10/scripts/run_dr_queue.sh`、`run_dr2_queue.sh` |

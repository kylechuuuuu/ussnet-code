#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document arm E (3-class zone parameterisation) in DESIGN_NEXT.md and README.md."""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DN = os.path.join(ROOT, "DESIGN_NEXT.md")
RM = os.path.join(ROOT, "..", "README.md")

H_SECTION = u"""### H.（2026-10-06 新增）3 类区域参数化 = arm E：把 §3.F 真正跑起来

§3.F「同时预测 127/255 两个区」一直只是列在清单上，因为看起来需要新标注。**其实不需要**：
原始 mask 的三值 {0,127,255} 里，127 和 255 **一直共存于全部 102 张**
（`tools/71_zone_split_stats.py` 实测：102/102 同时含两者，0 张只含其一，无其他取值）。
127 占并集的面积比 mean 0.488 / median 0.504（min 0.173 / max 0.741），
127 区面积占比 mean 0.0452、255 区 mean 0.0521 —— 内部界面是**真结构**，不是标注笔误。
§2.2 说它"不是错误集中区"，所以它的期望收益是**正则 + 多一个监督信号**，
这也正是它作为"新信息"值得一试的理由（前十五个实验都是在重加权同一个信号）。

做法（`tools/72_build_zone3_dataset.py` + `scripts/{58,59,60}_*.sh`）：

- 建 **step4 本地**的 `data/nnUNet_raw/Dataset002_UltrasoundProstate3/`：
  `imagesTr` 软链到 Dataset001，`labelsTr` 为 0/1/2（127→1、255→2），
  `splits_final.json` 原样复制（fold 0 仍是 71/31，与其余所有 arm 同一划分）。
  **对全部 102 例断言**「重建并集 == Dataset001 的前景」，并逐例 cross-check `fg_pixels`。
  数据根刻意放在 step4 本地、`nnUNet_results` 仍共享，绝不动只读的 step1 目录。
- `nnUNetv2_plan_and_preprocess -d 2 -c 2d`：planner 选出的 2d 配置与 Dataset001
  **完全一致**（patch 640×1024、batch 4、features (32,64,128,256,512,512,512,512)、
  strides ((1,1),(2,2)×7)、n_conv_per_stage (2)×8，预处理好后 170 MB）。
  所以 arm E 相对 arm A 只差**输出参数化**这一件事，归因干净。
- 损失侧唯一改动：`fp_boundary_losses.py` 里两处
  `prob = torch.softmax(net_output, dim=1)[:, 1:2]` 改成 `fg_prob(net_output)`
  ——2 类时两者**同值**（CPU smoke test 里 `max|diff| = 0`，故 arm A/B/C/D 的可复现性不受影响），
  3 类时返回 `1 - P(bg)` 即并集。`fg_from_target` 本来就取 `target > 0.5`，
  3 类下天然等于并集，不用改（nnU-Net 的 target 是整数类别图，已核对 `compound_losses.py`）。
- 推理侧：nnU-Net 存的是 argmax 的 {0,1,2}，而 `05_export_nifti_to_png.py` 结尾是
  `np.where(arr > 0, 255, 0)` —— **并集免费**，于是可以直接与其余 arm 用同一个
  `eval_lib.py` 比较，不需要任何特殊处理。
- 自检：`tools/73_smoke_zone3_loss.py`（CPU，无数据无 GPU）全部通过：
  2 类等价性、3 类并集、2/3 类前反向有限、3 类每个像素都有梯度（8192/8192）。

状态：链式脚本 `scripts/60_chain_armE_after_C.sh` 于 2026-10-06 09:44:10 UTC 启动，
**等 arm C 整条链（训练+预测+评测）报 `arm C ALL DONE` 后**自动占用空的 GPU1 跑
500 epochs，再自动预测/导出/评测。不现在开跑是刻意的：arm C 正在 GPU1 上，
并行只会把 arm C 的 epoch 时间翻倍、拖慢主交付物。

"""

ZERO = u"## 4. 已被证据否决的方向（不要再花时间）"

OLD_LAUNCH = u"bash scripts/51_train_fp.sh D 0 500              # arm D（ImageNet 预训练 ResNet34 编码器 + 指纹损失）\n"
NEW_LAUNCH = OLD_LAUNCH + u"bash scripts/58_train_zone3.sh 1 500             # arm E（3 类区域参数化 + 指纹损失，Dataset002；需先建数据集）\n"

OLD_DROW = u"| **新模型（指纹驱动：预训练编码器 arm D = ResNet34）** | 训练中（500 epochs / GPU0，2026-10-06 09:27 UTC 起；GPU0 与他人共享，≈25 h） | | | | | | 同上 |\n"
NEW_DROW = OLD_DROW + (
    u"| **新模型（指纹驱动：3 类区域参数化 arm E）** | 排队中（等 arm C 释放 GPU1 后自动开跑 500 epochs，"
    u"2026-10-06 09:44 UTC 起等待） | | | | | | 见 `step4_our_new_model_resunet_boundary/DESIGN_NEXT.md` §3.H |\n")

OLD_N = u"三个 arm，各 500 epochs，GPU0/GPU1 并行：**A** = 原架构 + 指纹损失（隔离损失本身的作用）；\n**B** = ResEnc（`n_blocks_per_stage=(1,2,2,3,3,3,3,3)`，比官方 ResEnc 更小，避免把提升归因于参数变多）\n+ 指纹损失 + 方向加权。设计与证据见 `step4_our_new_model_resunet_boundary/DESIGN_NEXT.md`，\n实现与复现步骤见 `IMPLEMENTATION.md`。\n"
NEW_N = (u"五个 arm，各 500 epochs，GPU0/GPU1 并行或接力：**A** = 原架构 + 指纹损失（隔离损失本身的作用）；\n"
         u"**B** = ResEnc（`n_blocks_per_stage=(1,2,2,3,3,3,3,3)`，比官方 ResEnc 更小，避免把提升归因于参数变多）\n"
         u"+ 指纹损失 + 方向加权；**C / D** = ImageNet 预训练 ResNet18 / ResNet34 编码器（只换编码器，"
         u"patch/batch/归一化/损失/深监督全不变，所以差异可归因到「编码器先验」）；\n"
         u"**E** = 3 类区域参数化（127/255 分开，并集与二值前景逐像素相同，故可直接与其余 arm 同表比较）。\n"
         u"设计与证据见 `step4_our_new_model_resunet_boundary/DESIGN_NEXT.md`，\n"
         u"实现与复现步骤见 `IMPLEMENTATION.md`。\n")


def patch(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, "expected exactly 1 occurrence in %s, found %d for: %r" % (
            os.path.basename(path), n, old[:70])
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8").write(s)
    print("[armE-docs] patched %s" % path)


patch(DN, [(ZERO, H_SECTION + ZERO)])
patch(RM, [(OLD_LAUNCH, NEW_LAUNCH), (OLD_DROW, NEW_DROW), (OLD_N, NEW_N)])

for p, key in ((DN, u"### H."), (RM, u"arm E")):
    s = io.open(p, encoding="utf-8").read()
    print("[armE-docs] %s -> %d lines, contains %r: %s" % (
        os.path.basename(p), s.count("\n") + 1, key, key in s))

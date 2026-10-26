#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document arm C in DESIGN_NEXT.md and USSnet/README.md.  Idempotent."""
import io
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(TOOLS)
USS = os.path.dirname(STEP4)

DN = os.path.join(STEP4, "DESIGN_NEXT.md")
RM = os.path.join(USS, "README.md")

SECTION_G = '''### G.（2026-10-06 新增）预训练编码器臂：唯一还没被证否的"新信息"来源

§5.1–§5.11 的十一个实验证明：**重新加权 / 后处理同一个信号，收益上限就是 +0.1**。
要越过去只能引入当前模型没有的信息。在 71 张可训练图、无新增标注、
GPU 仅 0/1 号的约束下，唯一能拿到的"新信息"是**别的数据上训出来的视觉先验**：
ImageNet 预训练编码器（本机 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` 已在，
**不需要联网**；`timm(pretrained=True)` 会去 HuggingFace 并挂死，禁用）。

实现原则：**只换编码器，其余（patch / batch / 归一化 / 损失 / 深监督 / 推理）全部不变**，
这样与 arm A/B、nnU-Net 的差异可以干净归因到"编码器先验"这一件事上。

- 编码器 `PreEncEncoder`：resnet18 主干，`conv1` 单通道化且 **stride 强制 1**、去掉 maxpool，
  所有 `BatchNorm2d` 就地换成 `InstanceNorm2d`；主干之后的 4 个 stage 用
  `StackedConvBlocks` 一直下采样到 stride 128。
- 用 4 个 1×1 `proj` 把主干通道**对齐到指纹给的**
  `features_per_stage=[32,64,128,256,512,512,512,512]`；
  解码器**直接复用 nnU-Net 的 `UNetDecoder`**（它对编码器是鸭子类型，只读
  `output_channels/strides/kernel_sizes/conv_op/norm_op/nonlin/...`），
  于是 7 档深监督顺序和 `mod.decoder.deep_supervision` 开关免费拿到。
- `freeze_upto=2`（stem/layer1/layer2 冻结、前向 `no_grad`）+ `layer3/layer4/proj` 用 0.1× lr:
  实测 **55.2 s/epoch、5.6 GB 峰值、55.82 M 可训练**（不冻结是 93.2 s / 10.0 GB）。
- **必须自己做 LR 调度**：nnU-Net 的 `PolyLRScheduler.step()` 是**全局**的
  （忽略每组的起始 lr），低 lr 组会被一步提到全局值；arm C 用 `_PolyLRPerGroup` 逐组保底。

状态：2026-10-06 09:13 UTC 起在 GPU1 训练 500 epochs（fold 0），
冒烟测试数字与结果见 `results/ARM_C_PREENC.md`。

'''

src = io.open(DN, encoding="utf-8").read()
if "### G.（2026-10-06 新增）" not in src:
    anchor = "## 4. 已被证据否决的方向（不要再花时间）"
    assert src.count(anchor) == 1, "DESIGN_NEXT anchor not found exactly once"
    src = src.replace(anchor, SECTION_G + anchor)
    io.open(DN, "w", encoding="utf-8").write(src)
    print("[patch] DESIGN_NEXT.md += 3.G")
else:
    print("[patch] DESIGN_NEXT.md already has 3.G")

src = io.open(RM, encoding="utf-8").read()
orig = src
reps = []

# (a) the run recipe
reps.append((
    'bash scripts/51_train_fp.sh B 1 500              # arm B（ResEnc + 指纹损失 + 方向加权）',
    'bash scripts/51_train_fp.sh B 1 500              # arm B（ResEnc + 指纹损失 + 方向加权）\n'
    'bash scripts/51_train_fp.sh C 1 500              # arm C（ImageNet 预训练 ResNet18 编码器 + 指纹损失）'))

# (b) the results table gets an arm-C row
reps.append((
    '| 新模型 + nnU-Net（概率集成） | 已排除（不作为方向） |',
    '| **新模型（指纹驱动：ImageNet 预训练编码器 = arm C）** | 训练中（500 epochs / GPU1，2026-10-06 09:13 UTC 起） | | | | | | 见 `step4_our_new_model_resunet_boundary/results/ARM_C_PREENC.md` |\n'
    '| 新模型 + nnU-Net（概率集成） | 已排除（不作为方向） |'))

# (c) how many arms
reps.append((
    '两个 arm，各 500 epochs，GPU0/GPU1 并行：',
    '三个 arm，各 500 epochs，GPU0/GPU1 并行：'))

for old, new in reps:
    assert src.count(old) == 1, "README anchor not found exactly once: %r" % old[:60]
    src = src.replace(old, new)

if src != orig:
    io.open(RM, "w", encoding="utf-8").write(src)
    print("[patch] README.md updated (3 edits)")
else:
    print("[patch] README.md already up to date")

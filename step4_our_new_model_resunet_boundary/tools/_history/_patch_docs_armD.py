#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document arm D next to arm C (DESIGN_NEXT.md section G + USSnet/README.md).
Idempotent."""
import io
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(TOOLS)
USS = os.path.dirname(STEP4)


def sub(path, old, new, tag):
    src = io.open(path, encoding="utf-8").read()
    if new in src:
        print("[docs] %s: already done" % tag)
        return
    assert src.count(old) == 1, "[docs] %s: anchor not unique (%d)" % (
        tag, src.count(old))
    io.open(path, "w", encoding="utf-8").write(src.replace(old, new))
    print("[docs] %s: patched" % tag)


# ---------------------------------------------------------------- DESIGN_NEXT
DN = os.path.join(STEP4, "DESIGN_NEXT.md")
sub(DN,
    "### G.（2026-10-06 新增）预训练编码器臂",
    "### G.（2026-10-06 新增）预训练编码器臂（arm C = ResNet18 / arm D = ResNet34）",
    "DESIGN_NEXT heading")

ARM_D_BLOCK = """- **arm D = resnet34，同一套代码**：`BACKBONES` 表里加一行即可，
  `freeze_upto=2` 时 66.60 M 参数 / 65.26 M 可训练。两个坑值得记：
  ① `resnet34-333f7ec4.pth` 是旧的 `.tar` 格式，torch >= 2.6 把 `torch.load` 的
  `weights_only` 默认值改成 `True`，直接抛
  `RuntimeError: Cannot use 'weights_only=True' with files saved in the legacy .tar format`，
  必须 try/except 回退到 `weights_only=False`（本机缓存里受信任的 torchvision IMAGE**NET**1K 权重）；
  ② **nnU-Net 用 trainer 的类名决定结果目录**，两个 arm 共用一个类名会让 arm D 写进
  arm C 的 `...__nnUNetPlans__2d/fold_0/` 并覆盖它的 checkpoint。所以 arm D 单独一个类
  `nnUNetTrainerProstateFPPreEncR34`，**只覆盖 `ARM_TAG`/`ARM_DESC`，绝对不能加 `__init__`**
  （nnU-Net 从 `locals()` 重建 `my_init_kwargs`，多一个参数就 KeyError）。

状态：**arm C**（resnet18）2026-10-06 09:13:57 UTC 起在 GPU1 训练 500 epochs
（≈55 s/epoch，ETA ≈16:50 UTC）；**arm D**（resnet34）09:27:16 UTC 起在 GPU0 训练
500 epochs（GPU0 与他人共享，≈179 s/epoch，ETA ≈次日 10:30 UTC）。两条链都由
`scripts/57_watch_then_evaluate.sh <C|D>` 驱动：只等**自己**那条训练结束
（等待模式是 `nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TR}[ ]-device`，
尾部锚定的 `[ ]-device` 才不会让 arm C 等到 arm D）→ 预测两个 checkpoint → 导 PNG →
共享 `eval_lib.py` 打分 → 重建六模型 `comparison_fp.md` → 概率集成。
"""
sub(DN,
    "状态：2026-10-06 09:13 UTC 起在 GPU1 训练 500 epochs（fold 0），\n"
    "冒烟测试数字与结果见 `results/ARM_C_PREENC.md`。\n",
    ARM_D_BLOCK, "DESIGN_NEXT status")

# ---------------------------------------------------------------------- README
RM = os.path.join(USS, "README.md")
sub(RM,
    "bash scripts/51_train_fp.sh C 1 500              # arm C（ImageNet 预训练 ResNet18 编码器 + 指纹损失）\n",
    "bash scripts/51_train_fp.sh C 1 500              # arm C（ImageNet 预训练 ResNet18 编码器 + 指纹损失）\n"
    "bash scripts/51_train_fp.sh D 0 500              # arm D（ImageNet 预训练 ResNet34 编码器 + 指纹损失）\n",
    "README launch lines")

lines = io.open(RM, encoding="utf-8").read().split("\n")
new_lines, hit = [], 0
for ln in lines:
    if "ImageNet 预训练编码器 = arm C" in ln:
        hit += 1
        new_lines.append(
            "| **新模型（指纹驱动：预训练编码器 arm C = ResNet18）** | 训练中（500 epochs / "
            "GPU1，2026-10-06 09:13 UTC 起，≈7.6 h） | | | | | | 见 "
            "`step4_our_new_model_resunet_boundary/results/ARM_C_PREENC.md` |")
        new_lines.append(
            "| **新模型（指纹驱动：预训练编码器 arm D = ResNet34）** | 训练中（500 epochs / "
            "GPU0，2026-10-06 09:27 UTC 起；GPU0 与他人共享，≈25 h） | | | | | | 同上 |")
    else:
        new_lines.append(ln)
if hit:
    io.open(RM, "w", encoding="utf-8").write("\n".join(new_lines))
    print("[docs] README results table: %d row(s) replaced by arm C + arm D" % hit)
else:
    print("[docs] README results table: already done")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
34 —— 从 nnU-Net 的 training_log_*.txt 里解析训练曲线，画成图。
（不要读 tee 出来的 stdout 日志：python 的 stdout 是块缓冲的，会滞后很多。）

用法（可多组 --log/--label；需要 matplotlib，请用 $HOME/anaconda3/envs/zk/bin/python 运行）：
  python3 scripts/34_plog_training_curves.py \
      --log <nnunet_log> --label "nnU-Net 2d" \
      --log <our_log>    --label "我们的模型" \
      --out ../step3_failure_analysis/figures/training_curves.png

画两张图：左 = EMA pseudo Dice vs epoch；右 = train/val loss vs epoch。
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体：容器里多半没有 CJK 字体，因此默认改成英文标签，避免一堆缺字警告 + 方块。
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _cjk_ok():
    try:
        from matplotlib import font_manager
        avail = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return False
    return bool(avail & {"Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
                         "Source Han Sans SC", "Noto Sans CJK JP"})


CJK = _cjk_ok()
# 英文/中文两套文案
TL = {"x_epoch": "epoch" if CJK else "epoch",
      "y_dice": "pseudo Dice",
      "t_dice": "\u9a8c\u8bc1\u96c6 pseudo Dice\uff08\u7ec6\u7ebf=\u9010 epoch\uff0c\u7c97\u7ebf=EMA\uff09" if CJK else
                "validation pseudo Dice (thin = per epoch, thick = EMA)",
      "y_loss": "loss",
      "t_loss": "\u8bad\u7ec3/\u9a8c\u8bc1\u635f\u5931" if CJK else "train/val loss",
      "train": "train" if CJK else "train",
      "val": "val" if CJK else "val"}

EMA_ALPHA = 0.9  # nnUNetTrainer 里的 EMA 系数

RE_EPOCH = re.compile(r"Epoch (\d+)\s*$")
RE_TIME = re.compile(r"Epoch time:\s*([\d.]+)")
RE_TRAIN = re.compile(r"train_loss\s+(-?[\d.eE+]+)")
RE_VAL = re.compile(r"val_loss\s+(-?[\d.eE+]+)")
RE_PSEUDO = re.compile(r"Pseudo dice \[([\d.]+)\]")
RE_EMA = re.compile(r"New best EMA pseudo Dice:\s*([\d.]+)")
RE_LR = re.compile(r"Current learning rate:\s*([\d.eE-]+)")


def parse(path):
    rows = []
    cur = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = RE_EPOCH.search(line.rstrip())
            if m:
                if cur is not None:
                    rows.append(cur)
                cur = {"epoch": int(m.group(1))}
                continue
            if cur is None:
                continue
            for key, rx, cast in (("time", RE_TIME, float),
                                  ("train_loss", RE_TRAIN, float),
                                  ("val_loss", RE_VAL, float),
                                  ("pseudo_dice", RE_PSEUDO, float),
                                  ("lr", RE_LR, float)):
                m = rx.search(line)
                if m:
                    try:
                        cur[key] = cast(m.group(1))
                    except ValueError:
                        pass
            m = RE_EMA.search(line)
            if m:
                cur.setdefault("ema_logged", []).append(float(m.group(1)))
    if cur is not None:
        rows.append(cur)
    rows = [r for r in rows if "pseudo_dice" in r]
    # 用与 nnUNetTrainer 相同的 alpha 重算 EMA（并用日志里的 EMA 值做校验）
    ema = None
    check = []
    for r in rows:
        v = r["pseudo_dice"]
        ema = v if ema is None else EMA_ALPHA * ema + (1 - EMA_ALPHA) * v
        r["ema"] = ema
        if "ema_logged" in r:
            check.append((r["epoch"], r["ema_logged"][-1], round(ema, 4)))
    return rows, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    assert len(args.log) == len(args.label), "--log 和 --label 数量必须一致"

    series = []
    for p, lb in zip(args.log, args.label):
        if not os.path.exists(p):
            print("[34] skip (missing): %s" % p)
            continue
        rows, check = parse(p)
        if not rows:
            print("[34] skip (no epochs parsed): %s" % p)
            continue
        series.append((lb, rows))
        if check:
            bad = [c for c in check if abs(c[1] - c[2]) > 2e-3]
            print("[34] %s: %d epochs, EMA 重算校验 %d/%d 一致%s"
                  % (lb, len(rows), len(check) - len(bad), len(check),
                     "" if not bad else "  不一致样例: %s" % bad[:3]))
        else:
            print("[34] %s: %d epochs" % (lb, len(rows)))
    if not series:
        print("[34] nothing to plot")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for i, (lb, rows) in enumerate(series):
        c = colors[i % len(colors)]
        ep = [r["epoch"] for r in rows]
        axes[0].plot(ep, [r["pseudo_dice"] for r in rows], color=c, alpha=0.28, lw=1)
        axes[0].plot(ep, [r["ema"] for r in rows], color=c, lw=2, label=lb)
        if "train_loss" in rows[0]:
            axes[1].plot(ep, [r.get("train_loss", float("nan")) for r in rows],
                         color=c, lw=1.6, label="%s train" % lb)
        if "val_loss" in rows[0]:
            axes[1].plot(ep, [r.get("val_loss", float("nan")) for r in rows],
                         color=c, lw=1.6, ls="--", label="%s val" % lb)

    axes[0].set_xlabel(TL["x_epoch"]); axes[0].set_ylabel(TL["y_dice"])
    axes[0].set_title(TL["t_dice"])
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
    axes[1].set_xlabel(TL["x_epoch"]); axes[1].set_ylabel(TL["y_loss"])
    axes[1].set_title(TL["t_loss"]); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    title = args.title or ("\u8bad\u7ec3\u66f2\u7ebf" if CJK else "training curves")
    if not CJK:
        title = "".join(ch if ord(ch) < 128 else "" for ch in title).strip() or "training curves"
    if not CJK and any(ord(ch) > 127 for ch in title):
        print("[34] WARNING: CJK label dropped because no CJK font is installed")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("[34] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

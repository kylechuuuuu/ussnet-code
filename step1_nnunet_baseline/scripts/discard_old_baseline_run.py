#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
丢弃旧的 baseline 训练结果并归档其日志。

背景（2026-10-02）：旧的 fold_0 训练跑出的是 PlainConvUNet（126.56 M 参数，
state_dict 里 0 个 residual block），并且来源环境与本项目脚本不一致，
用户决定（见对话）废弃重跑。这里做的是：
  1. 把旧的 fold_0 结果目录改名归档（不直接删，先留证据）
  2. 把被外部杀掉的训练日志改名归档
之后由 restart 脚本重新完整训练 1000 epoch。
"""
import os
import shutil
import sys
import time

V = "/media/72TB_disk/personal/zk/USSnet"
RES = os.path.join(V, "step1_nnunet_baseline/data/nnUNet_results/Dataset001_UltrasoundProstate")
OLD_RUN = os.path.join(RES, "nnUNetTrainer__nnUNetPlans__2d")
LOGS = os.path.join(V, "step1_nnunet_baseline/logs")
STAMP = time.strftime("%Y%m%d_%H%M%S")

def guard(path):
    """绝对路径安全检查：必须落在预期的 step1 目录内。"""
    ap = os.path.abspath(path)
    must = os.path.abspath(os.path.join(V, "step1_nnunet_baseline"))
    if not ap.startswith(must + os.sep):
        raise SystemExit("[guard] 拒绝操作越界路径: %s" % ap)
    return ap

# ---- 1. 归档旧结果目录 ---------------------------------------------------
if os.path.isdir(OLD_RUN):
    dst = guard(OLD_RUN + "_DISCARDED_plainconv_" + STAMP)
    print("[discard] 归档旧结果: %s" % OLD_RUN)
    print("       -> %s" % dst)
    shutil.move(OLD_RUN, dst)
    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fs in os.walk(dst) for f in fs)
    print("[discard] 已归档, 大小 %.1f MB" % (sz / 1e6))
else:
    print("[discard] %s 不存在, 跳过" % OLD_RUN)

# ---- 2. 归档旧的训练日志 -------------------------------------------------
for fn in ["03_train_fold0_nnUNetTrainer.log", "03_train_wrapper.log"]:
    p = os.path.join(LOGS, fn)
    if os.path.isfile(p):
        dst = guard(p + ".killed_ep673_" + STAMP)
        shutil.move(p, dst)
        print("[discard] %s -> %s" % (fn, os.path.basename(dst)))

print("\n[discard] 完成。下一步用 restart 脚本重新完整训练。")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4 自定义 trainer (nnUNetTrainerProstateResUNetBoundary) 对 nnunetv2 2.6.2
的兼容性检查。只读检查，不修改任何文件。

用法:
  ~/anaconda3/envs/zk/bin/python compat_check_262.py
"""
import importlib, inspect, os, sys, traceback

_RAW = "/media/72TB_disk/personal/zk/USSnet/step1_nnunet_baseline/data/nnUNet_raw"
_PRE = "/media/72TB_disk/personal/zk/USSnet/step1_nnunet_baseline/data/nnUNet_preprocessed"
_RES = "/media/72TB_disk/personal/zk/USSnet/step1_nnunet_baseline/data/nnUNet_results"
os.environ.setdefault("nnUNet_raw", _RAW)
os.environ.setdefault("nnUNet_preprocessed", _PRE)
os.environ.setdefault("nnUNet_results", _RES)

def hr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)

hr("1. 版本")
try:
    from importlib.metadata import version
    print("pip nnunetv2      =", version("nnunetv2"))
    print("pip dna           =", version("dynamic-network-architectures"))
    print("pip batchgenv2    =", version("batchgeneratorsv2"))
except Exception as e:
    print("pip 查询失败:", e)
import nnunetv2
print("nnunetv2 __file__ =", nnunetv2.__file__)
import dynamic_network_architectures as dna
print("dna __file__      =", dna.__file__)
try:
    import torch
    print("torch             =", torch.__version__, torch.cuda.is_available())
except Exception as e:
    print("torch 失败:", e)

hr("2. nnUNetTrainer 关键方法")
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as T
for name in ["initialize", "build_network_architecture", "_build_loss", "on_train_epoch_start",
             "get_dataloaders", "get_training_transforms", "get_validation_transforms",
             "set_deep_supervision_enabled", "train_step", "validation_step",
             "run_training", "maybe_load_checkpoint", "print_to_log_file", "_do_i_compile",
             "_set_batch_size_and_oversample", "save_checkpoint", "load_checkpoint"]:
    print("  %-34s %s" % (name, hasattr(T, name)))
print("  TRAINER FILE =", T.__module__, inspect.getsourcefile(T))

hr("3. build_network_architecture 签名 (2.6.2)")
print("  base :", inspect.signature(T.build_network_architecture))
try:
    src = inspect.getsource(T.initialize)
    print("  initialize 源码中的关键行:")
    for ln in src.split("\n"):
        if "build_network_architecture" in ln or "self.loss" in ln or "compile" in ln:
            print("     " + ln.rstrip())
except Exception as e:
    print("  initialize 失败:", e)

hr("4. _build_loss 源码 (2.6.2)")
try:
    print(inspect.getsource(T._build_loss))
except Exception as e:
    print("  失败:", e)

hr("5. __init__ 源码 (2.6.2)")
try:
    print(inspect.getsource(T.__init__))
except Exception as e:
    print("  失败:", e)

hr("6. 外部 trainer 查找机制 nnUNet_extTrainer")
for mod_name in ["nnunetv2.run.run_training", "nnunetv2.utilities.find_class_by_trainer_name"]:
    try:
        m = importlib.import_module(mod_name)
        s = inspect.getsource(m)
        hits = [ln.rstrip() for ln in s.split("\n") if "extTrainer" in ln]
        print("  %s:" % mod_name)
        for h in (hits if hits else ["   （无 extTrainer 关键词）"]):
            print("     " + h)
    except Exception as e:
        print("  %s 导入失败: %r" % (mod_name, e))

hr("7. ResidualEncoderUNet 构造参数")
try:
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
    print("  ResidualEncoderUNet:", inspect.signature(ResidualEncoderUNet.__init__))
    print("  PlainConvUNet      :", inspect.signature(PlainConvUNet.__init__))
except Exception:
    traceback.print_exc()

hr("8. get_network_from_plans 签名")
try:
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    print("  ", inspect.signature(get_network_from_plans))
except Exception:
    traceback.print_exc()

hr("9. 我们的 trainer 能否被导入并识别")
sys.path.insert(0, "/media/72TB_disk/personal/zk/USSnet/step4_our_new_model_resunet_boundary/trainer")
try:
    import nnUNetTrainerProstateResUNetBoundary as M
    print("  模块导入 OK:", M.__file__)
    print("  类:", [n for n in dir(M) if n.startswith("nnUNetTrainer")])
    from nnunetv2.utilities.find_class_by_trainer_name import find_class_by_trainer_name
    cls = find_class_by_trainer_name("nnUNetTrainerProstateResUNetBoundary", None)
    print("  find_class_by_trainer_name ->", cls)
except Exception:
    traceback.print_exc()

hr("10. 实例化 + my_init_kwargs")
try:
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from batchgenerators.utilities.file_and_folder_operations import load_json
    plans = load_json(_PRE + "/Dataset001_UltrasoundProstate/nnUNetPlans.json")
    dj = load_json(_PRE + "/Dataset001_UltrasoundProstate/dataset.json")
    pm = PlansManager(plans); cm = pm.get_configuration("2d")
    tr = M.nnUNetTrainerProstateResUNetBoundary(pm, cm, 0, dj, device=torch.device("cpu"))
    print("  实例化 OK; my_init_kwargs =", tr.my_init_kwargs)
    print("  enable_deep_supervision =", tr.enable_deep_supervision)
except Exception:
    traceback.print_exc()

hr("11. 网络能否按 2.6.2 的签名构建")
try:
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from batchgenerators.utilities.file_and_folder_operations import load_json
    pm = PlansManager(load_json(_PRE + "/Dataset001_UltrasoundProstate/nnUNetPlans.json"))
    cm = pm.get_configuration("2d")
    net = M.nnUNetTrainerProstateResUNetBoundary.build_network_architecture(pm, cm, 1, 2, True)
    print("  网络构建 OK:", type(net))
except Exception:
    traceback.print_exc()

hr("12. 2.6.2 的 planner 是否支持 ResEnc")
try:
    from nnunetv2.experiment_planning.experiment_planners.resencUNet_planner import ResEncUNetPlanner
    print("  ResEncUNetPlanner OK")
    print("  get_* :", [n for n in dir(ResEncUNetPlanner) if n.startswith("get_")])
except Exception:
    traceback.print_exc()

hr("13. 自定义 trainer 实例化后 _build_loss 能否真的跑起来")
try:
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from batchgenerators.utilities.file_and_folder_operations import load_json
    pm = PlansManager(load_json(_PRE + "/Dataset001_UltrasoundProstate/nnUNetPlans.json"))
    cm = pm.get_configuration("2d")
    tr = M.nnUNetTrainerProstateResUNetBoundary(pm, cm, 0, dj, device=torch.device("cpu"))
    loss = tr._build_loss()
    print("  _build_loss OK:", type(loss))
    x = torch.randn(1, 2, 64, 64)
    y = (torch.rand(1, 1, 64, 64) > 0.5).to(torch.int16)
    out = loss(x, y)
    print("  loss 前向 OK:", float(out))
    tr.loss = loss
    tr.loss.set_epoch(50)
    tr.on_train_epoch_start.__wrapped__ if hasattr(tr.on_train_epoch_start, "__wrapped__") else None
    print("  set_epoch OK")
except Exception:
    traceback.print_exc()

print("\n===== compat_check_262 DONE =====")

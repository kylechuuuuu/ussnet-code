#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bring the superseded `nnUNetTrainerProstateResUNetBoundary` back into a
runnable state: its `build_network_architecture` was written for the nnU-Net
>= 2.8 call form (plans_manager, configuration_manager, ...) while the zk env is
nnU-Net 2.6.2, whose `nnUNetTrainer.initialize` passes the architecture over as

    (architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
     num_input_channels, num_output_channels, enable_deep_supervision)

Calling the old version under 2.6.2 raises TypeError before training starts.
The body is rewritten to the exact form already proven by arm B
(`nnUNetTrainerProstateResUNetFPBoundary`, which trains and predicts fine).
Idempotent."""
import io
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(TOOLS)
P = os.path.join(STEP4, "trainer", "nnUNetTrainerProstateResUNetBoundary.py")

OLD = '''    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Same fingerprint plans, residual encoder instead of plain conv encoder."""
        arch_kwargs = dict(configuration_manager.network_arch_init_kwargs)
        n_stages = int(arch_kwargs["n_stages"])
        if "n_blocks_per_stage" in arch_kwargs:
            pass                                             # plans already residual
        else:
            arch_kwargs.pop("n_conv_per_stage", None)        # not accepted by ResEnc
            arch_kwargs["n_blocks_per_stage"] = residual_blocks_per_stage(n_stages)

        network = get_network_from_plans(
            RESENC_UNET_CLASS,
            arch_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return network
'''

NEW = '''    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Same fingerprint plans, residual encoder instead of plain conv encoder.

        NOTE (2026-10-06): written for the nnU-Net **2.6.2** call form, which is
        what this project's zk env ships -- `nnUNetTrainer.initialize` calls
        `build_network_architecture(class_name, init_kwargs, req_import,
        num_input_channels, num_output_channels, enable_deep_supervision)`.
        nnU-Net >= 2.8 passes `plans_manager, configuration_manager, ...`
        instead; a trainer written for one form raises TypeError on the other.
        Keep this signature byte-for-byte in sync with
        `nnUNetTrainerProstateResUNetFPBoundary.build_network_architecture`.
        """
        arch_kwargs = dict(arch_init_kwargs)
        n_stages = int(arch_kwargs["n_stages"])
        if "n_blocks_per_stage" in arch_kwargs:
            pass                                             # plans already residual
        else:
            arch_kwargs.pop("n_conv_per_stage", None)        # not accepted by ResEnc
            arch_kwargs["n_blocks_per_stage"] = residual_blocks_per_stage(n_stages)

        return get_network_from_plans(
            RESENC_UNET_CLASS,
            arch_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
'''

BANNER_OLD = '"""\nstep4_our_new_model_resunet_boundary/trainer/nnUNetTrainerProstateResUNetBoundary.py\n'
BANNER_NEW = ('"""\n'
              'step4_our_new_model_resunet_boundary/trainer/'
              'nnUNetTrainerProstateResUNetBoundary.py\n'
              '\n'
              '.. deprecated:: 2026-10-06\n'
              '   SUPERSEDED.  This was the first design (ResEnc + a hand-tuned\n'
              '   boundary-weighted BCE, `step4/DESIGN.md`).  It is not registered in\n'
              '   `scripts/53_install_trainer_shim.sh` and is not launched by any\n'
              '   current script; the experiments that replaced it are the\n'
              '   fingerprint-loss arms A/B/C/D driven by `scripts/51_train_fp.sh`.\n'
              '   It is kept only so the DESIGN.md-era code stays readable.  Its\n'
              '   `build_network_architecture` has been ported to the nnU-Net 2.6.2\n'
              '   signature so that it is runnable rather than instantly broken.\n')

src = io.open(P, encoding="utf-8").read()
orig = src
if ".. deprecated:: 2026-10-06" not in src:
    assert src.count(BANNER_OLD) == 1, "docstring banner anchor not unique"
    src = src.replace(BANNER_OLD, BANNER_NEW)
    print("[patch] added the deprecation banner")
else:
    print("[patch] banner already present")
if OLD in src:
    assert src.count(OLD) == 1
    src = src.replace(OLD, NEW)
    print("[patch] ported build_network_architecture to the 2.6.2 signature")
elif "architecture_class_name: str" in src:
    print("[patch] build_network_architecture already ported")
else:
    raise SystemExit("[patch] ERROR: neither the old nor the new method was found")

if src != orig:
    io.open(P, "w", encoding="utf-8").write(src)

import ast
ast.parse(io.open(P, encoding="utf-8").read())
print("[patch] AST OK -> %s" % P)

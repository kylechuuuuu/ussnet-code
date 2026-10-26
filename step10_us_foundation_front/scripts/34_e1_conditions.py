#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_us_foundation_front/scripts/34_e1_conditions.py

E1: condition-wise arbitration delta table  delta(c) = bdice(free) - bdice(raw).

For each arm x condition (17 in-domain/operator conditions + 3 held-out):
  1) export the three head masks ON THAT CONDITION's images
     (28_export_student_masks.py --image-root ... --out-dir ...)
  2) run boundary arbitration with the CONDITION IMAGE as evidence
     (63_arbitrate.py --image-root ... --names-file data/fold3_val_names.txt)
  3) collect raw / free / contain per (arm, condition)

Outputs:
  results/E1_delta_conditions.csv
  results/E1_DELTA_CONDITIONS.md
  logs/e1_conditions.log (when launched with nohup)

usage: python scripts/34_e1_conditions.py --gpu 3 --workers 8 \
          --arms l2vit_gDa_fold3,l2vit_dr_pca4_fold3,l2vit_r1nodist_fold3
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
USS = "/media/72TB_disk/personal/zk/USSnet"
STEP6 = os.path.join(USS, "step6_sandwich_squeeze")
SH = os.path.join(USS, "step8_domain_shift", "data")
PY = sys.executable
NAMES = os.path.join(ROOT, "data", "fold3_val_names.txt")

CONDS17 = ["in_domain", "affine_gain_m25", "affine_gain_p25", "blur_20", "blur_35",
           "compound_A", "compound_B", "gamma_060", "gamma_080", "gamma_125",
           "jpeg_15", "jpeg_30", "shadow", "speckle_15", "speckle_35", "tgc_m60", "tgc_p60"]
HELD = ["blur_45", "speckle_50", "compound_C"]


def image_root(cond):
    if cond == "in_domain":
        return os.path.join(USS, "UltrasoundProstate", "images")
    if cond in HELD:
        return os.path.join(ROOT, "data", "shift_extra_fold3", cond, "images")
    return os.path.join(SH, "shift_fold3", cond, "images")


def wait_gpu(gpu):
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(gpu)],
                          capture_output=True, text=True).stdout.strip()
    for i in range(240):
        busy = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-i", str(gpu)], capture_output=True, text=True).stdout.strip()
        if busy.count(uuid) == 0 and mem and int(mem) < 1500:
            print("[e1] gpu %d free" % gpu, flush=True)
            return True
        time.sleep(60)
    return False


def export(arm, cond, gpu):
    out = os.path.join(ROOT, "predictions", "e1", arm, cond)
    if os.path.isdir(os.path.join(out, "base")) and len(os.listdir(os.path.join(out, "base"))) == 31:
        return out
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [PY, "-u", os.path.join(ROOT, "scripts", "28_export_student_masks.py"),
           "--tag", arm, "--split", "val", "--fold", "3", "--device", "cuda:0",
           "--image-root", image_root(cond), "--out-dir", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("export failed %s/%s:\n%s\n%s" % (arm, cond, r.stdout[-800:], r.stderr[-800:]))
    return out


def arbitrate(arm, cond, out):
    res = os.path.join(STEP6, "results", "arbitrate_e1_%s_%s.json" % (arm, cond))
    if os.path.exists(res):
        return res
    cmd = [PY, "-u", os.path.join(STEP6, "tools", "63_arbitrate.py"),
           "--split", "val", "--names-file", NAMES,
           "--image-root", image_root(cond),
           "--out-dir", os.path.join(out, "arb"),
           "--tag", "e1_%s_%s" % (arm, cond),
           "--base-png-dir", os.path.join(out, "base"),
           "--inner-png-dir", os.path.join(out, "cautious"),
           "--outer-png-dir", os.path.join(out, "amplified")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(res):
        raise RuntimeError("arbitrate failed %s/%s:\n%s\n%s" % (arm, cond, r.stdout[-800:], r.stderr[-800:]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", default="l2vit_gDa_fold3,l2vit_dr_pca4_fold3,l2vit_r1nodist_fold3")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    conds = CONDS17 + HELD
    print("[e1] arms=%s | conditions=%d | gpu=%d" % (arms, len(conds), args.gpu), flush=True)

    if not wait_gpu(args.gpu):
        raise SystemExit("[e1] gpu %d never became free" % args.gpu)

    # 1) mask exports (sequential on the GPU; model load per condition)
    t0 = time.time()
    for ai, arm in enumerate(arms):
        for ci, cond in enumerate(conds):
            export(arm, cond, args.gpu)
            print("[e1] export %d/%d %s/%s (%.1f min)" % (
                ai * len(conds) + ci + 1, len(arms) * len(conds), arm, cond, (time.time() - t0) / 60), flush=True)

    # 2) arbitration (parallel CPU workers)
    jobs = [(arm, cond) for arm in arms for cond in conds]
    done = [0]

    def work(job):
        arm, cond = job
        out = os.path.join(ROOT, "predictions", "e1", arm, cond)
        arbitrate(arm, cond, out)
        done[0] += 1
        print("[e1] arb %d/%d %s/%s" % (done[0], len(jobs), arm, cond), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))

    # 3) summary
    rows = []
    for arm in arms:
        for cond in conds:
            p = os.path.join(STEP6, "results", "arbitrate_e1_%s_%s.json" % (arm, cond))
            d = json.load(open(p))["summary"]
            m = d["mean"]
            rows.append({"arm": arm, "condition": cond, "n": d["n"],
                         "raw": m["raw"], "free": m["free"], "delta": m["free"] - m["raw"],
                         "contain": m.get("contain_lrn")})
    out_csv = os.path.join(ROOT, "results", "E1_delta_conditions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "condition", "n", "raw", "free", "delta", "contain"])
        w.writeheader()
        w.writerows(rows)
    out_md = os.path.join(ROOT, "results", "E1_DELTA_CONDITIONS.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# E1: condition-wise arbitration delta (fold3, val 31)\n\n")
        fh.write("`delta = free - raw` (bdice@1px); arbitration params frozen (frac 0.7 / maxstep 4 / sigma 1.5 / ±16px).\n\n")
        for arm in arms:
            sub = [r for r in rows if r["arm"] == arm]
            pos = sum(1 for r in sub if r["delta"] > 0)
            fh.write("## %s\n\n| condition | n | raw | free | delta | contain |\n|---|---|---|---|---|---|\n" % arm)
            for r in sub:
                fh.write("| %s | %d | %.2f | %.2f | %+.2f | %.3f |\n" % (
                    r["condition"], r["n"], r["raw"], r["free"], r["delta"],
                    r["contain"] if r["contain"] is not None else float("nan")))
            fh.write("\n**summary: delta>0 in %d/%d conditions; median delta %+.2f; min %+.2f; median contain %.3f**\n\n" % (
                pos, len(sub),
                sorted(r["delta"] for r in sub)[len(sub) // 2],
                min(r["delta"] for r in sub),
                sorted(r["contain"] for r in sub if r["contain"] is not None)[max(0, (len(sub) - 1) // 2)]))
    print("[e1] wrote", out_csv, "and", out_md, flush=True)


if __name__ == "__main__":
    sys.exit(main())

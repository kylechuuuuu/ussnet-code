import io
p = "/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/REPORT_TWO_MODEL_AND_DOMAIN_SHIFT.md"
s = io.open(p, encoding="utf-8").read()
old = "成本实测：resenc 三明治 2d ≈ **122 s/epoch**，stock ctl ≈ 49 s/epoch，60ep 单臂 ≈ 2 h。"
new = ("成本实测（各臂自己的训练日志）：**resenc 三明治 2d ≈ 47 s/epoch**（SQ2res-w25 跑 500ep 用了 6.5 h）；"
       "同一骨干**再加上 step9 的容差带损失 ≈ 122 s/epoch**（即带损失让单 epoch 慢 2.6×，该臂已判负）；"
       "stock 骨干 ≈ 49 s/epoch ⇒ 60ep 的单臂增广消融 ≈ 50 min，可承受。")
assert old in s, "cost line not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")

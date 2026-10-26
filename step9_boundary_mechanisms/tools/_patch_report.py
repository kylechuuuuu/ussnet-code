import io
p = "/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/REPORT_TWO_MODEL_AND_DOMAIN_SHIFT.md"
s = io.open(p, encoding="utf-8").read()

old = """| 行 | `r_err`（两模型带符号误差的 Pearson） | `r_AB`（穿越位置本身） | `same_dir` |
|---|---|---|---|
| `pair_sq2_stock` | **+0.848** | +0.079 | — |
| `pair_sq2_fpB` | **+0.900** | +0.071 | — |
| `pair_sq2_fpA` | **+0.861** | +0.078 | — |
| `pair_sq2_sq2a` | **+0.842** | +0.065 | — |
| `pair_stock_fpA` | +0.877 | +0.509 | — |
| `pair_stock_sq2a` | +0.855 | +0.450 | — |"""
new = """| 行 | `r_err`（两模型带符号误差的 Pearson） | `r_AB`（穿越位置本身） | `same_dir`（同侧比例） | `both_gt2px`（两者同时 >2px） |
|---|---|---|---|---|
| `pair_sq2_stock` | **+0.848** | +0.079 | 0.893 | 0.443 |
| `pair_sq2_fpB` | **+0.900** | +0.071 | 0.916 | 0.452 |
| `pair_sq2_fpA` | **+0.861** | +0.078 | 0.904 | 0.446 |
| `pair_sq2_sq2a` | **+0.842** | +0.065 | 0.890 | 0.444 |
| `pair_stock_fpA` | +0.877 | +0.509 | 0.900 | 0.451 |
| `pair_stock_sq2a` | +0.855 | +0.450 | 0.895 | 0.444 |"""
assert old in s
s = s.replace(old, new)

old2 = """3. **根本原因（r_err = 0.84~0.90）**"""
new2 = """3. **两个模型在 89%~92% 的法向点上位于真值的同一侧，在 44% 的点上两者同时偏离 >2px** ⇒ 「一个模型框内圈、一个模型框外圈」在绝大多数点上退化成「两个模型框同一侧」，区间里没有「内/外」可言。
4. **根本原因（r_err = 0.84~0.90）**"""
assert old2 in s
s = s.replace(old2, new2)

# renumber the following two items (4. -> 5., 5. -> 6.)
s = s.replace("""4. **唯一的正面发现**""", """5. **唯一的正面发现**""")
s = s.replace("""5. 注：`arch3h` 的 width""", """6. 注：`arch3h` 的 width""")
io.open(p, "w", encoding="utf-8").write(s)
print("patched", len(s))

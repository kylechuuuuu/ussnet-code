import io
p = '/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/tools/99ai_robust_fallback.py'
s = io.open(p, encoding='utf-8').read()
old = '''    out = {"per_cond": P, "pool": POOL, "gate_sweep": sweep}'''
new = '''    per = {c: {n: {k: float(v) for k, v in d.items() if k not in ("sq", "st", "fp")}
               for n, d in P[c].items()} for c in P}
    out = {"per_cond": per, "pool": POOL, "gate_sweep": sweep}'''
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched 99ai json OK')

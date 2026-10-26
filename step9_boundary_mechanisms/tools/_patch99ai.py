import io
p = '/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/tools/99ai_robust_fallback.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace('        P[cond]["_pool"] = pool', '        POOL[cond] = pool')
s = s.replace('''    out = {"per_cond": {c: {k: v for k, v in P[c].items() if k != "_pool"} for c in P},
           "pool": {c: P[c]["_pool"] for c in P}, "gate_sweep": sweep}''',
              '''    out = {"per_cond": P, "pool": POOL, "gate_sweep": sweep}''')
s = s.replace('''    P = {}
    for cond in ["in_domain", "compound_B"]:''',
              '''    P, POOL = {}, {}
    for cond in ["in_domain", "compound_B"]:''')
assert 'POOL[cond] = pool' in s and 'POOL = {}' in s
io.open(p, 'w', encoding='utf-8').write(s)
print('patched 99ai OK')

import io
p = '/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/tools/99ah_analyze_norm.py'
s = io.open(p, encoding='utf-8').read()
old = '''                b = load("tn_%s" % v, st, cond)'''
new = '''                b = load("tn", st, "%s__%s" % (v, cond))'''
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched 99ah dir naming OK')

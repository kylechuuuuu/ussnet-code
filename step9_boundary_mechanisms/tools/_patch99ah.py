import io
p = '/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/tools/99ah_analyze_norm.py'
s = io.open(p, encoding='utf-8').read()

old = '''_spec = importlib.util.spec_from_file_location("s57", os.path.join(
    USS, "step5_squeeze_boundary", "tools", "57_boundary_lib.py"))
s57 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s57)
_m82 = importlib.util.spec_from_file_location("m82", os.path.join(S8, "tools", "82_clinical_metrics.py"))
m82 = importlib.util.module_from_spec(_m82)
_m82.loader.exec_module(m82)'''
new = '''sys.path.insert(0, os.path.join(USS, "step4_our_new_model_resunet_boundary", "tools"))
import importlib  # noqa: E402
m82 = importlib.import_module("82_clinical_metrics")
s57 = importlib.import_module("57_shape_prior_ceiling")

MASK_DIR = os.path.join(USS, "UltrasoundProstate", "masks")'''
assert old in s, 'old import block not found'
s = s.replace(old, new)

old2 = '''    gt = {}
    for n in names:
        gt[n] = s57.load_gt(os.path.splitext(n)[0])'''
new2 = '''    gt = {n: eval_lib.load_binary(os.path.join(MASK_DIR, n)) for n in names}'''
assert old2 in s, 'old gt block not found'
s = s.replace(old2, new2)

io.open(p, 'w', encoding='utf-8').write(s)
print('patched 99ah OK')

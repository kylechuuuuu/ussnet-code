import io
p = '/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/tools/99ah_analyze_norm.py'
s = io.open(p, encoding='utf-8').read()

old = '''def load(tag, stem, cond):
    d = os.path.join(PRED, "%s_%s_base_nifti" % (tag, cond))
    fp = os.path.join(d, stem + ".npz")
    if not os.path.exists(fp):
        return None
    return m82.read_npz_prob(fp)'''
new = '''def load_fg(fp):
    """nnU-Net saves probabilities with shape (2, 1, H, W): squeeze the
    singleton, then take channel 1 (foreground).  82_clinical_metrics'
    read_npz_prob takes a[0] on 4-D input, i.e. the BACKGROUND channel."""
    with np.load(fp) as d:
        p = np.asarray(d["probabilities"], dtype=np.float32)
    while p.ndim > 3 and p.shape[1] == 1:
        p = p[:, 0]
    if p.ndim == 3 and p.shape[0] == 2:
        p = p[1]
    return p


def load(tag, stem, cond):
    d = os.path.join(PRED, "%s_%s_base_nifti" % (tag, cond))
    fp = os.path.join(d, stem + ".npz")
    if not os.path.exists(fp):
        return None
    return load_fg(fp)'''
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched 99ah load_fg OK')

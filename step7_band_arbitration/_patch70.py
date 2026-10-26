import io, sys
p = '/media/72TB_disk/personal/zk/USSnet/step7_band_arbitration/tools/70_dp_arbitrate.py'
s = io.open(p, encoding='utf-8').read()
old = "        tot = dp[:, None, :] - cost[None, :, :]             # (s, prev, cur)\n        k = np.argmax(tot, axis=2)                          # (s, cur)\n        bk[i] = k.astype(np.int16)\n        dp = tot[np.arange(K)[:, None], k, np.arange(K)[None, :]] + Wm[i][None, :]"
new = "        tot = dp[:, :, None] - cost[None, :, :]              # (s, prev, cur)\n        k = np.argmax(tot, axis=1)                           # (s, cur) best prev\n        bk[i] = k.astype(np.int16)\n        dp = np.take_along_axis(tot, k[:, None, :], axis=1)[:, 0, :] + Wm[i][None, :]"
if old not in s:
    print('PATTERN NOT FOUND'); sys.exit(1)
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new))
print('patched ok')

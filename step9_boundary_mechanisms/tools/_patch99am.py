import io, re
p = "/media/72TB_disk/personal/zk/USSnet/step9_boundary_mechanisms/tools/99am_twomodel_band.py"
s = io.open(p, encoding="utf-8").read()

old = '''            st = band_stats(-np.clip(ti, 0.0, float(arb.OFFS[-1]))[okc],
                            np.clip(to, 0.0, float(arb.OFFS[-1]))[okc], tgt)
            st["nan_frac"] = float((~okc).mean())
            push("arch3h", st)
            per_case.append({"name": stem, "row": "arch3h", **st})'''
new = '''            lo3 = -np.clip(ti, 0.0, float(arb.OFFS[-1]))[okc]
            hi3 = np.clip(to, 0.0, float(arb.OFFS[-1]))[okc]
            st = band_stats(lo3, hi3, tgt)
            st["nan_frac"] = float((~okc).mean())
            st["rho_w_err"] = float(sps.spearmanr(hi3 - lo3, np.abs(tgt)).statistic)
            push("arch3h", st)
            pushc("arch3h", "rho_w_err", st["rho_w_err"])
            per_case.append({"name": stem, "row": "arch3h", **st})'''
assert old in s, "arch3h block not found"
s = s.replace(old, new)

old2 = '''    for row in sorted(corr):
        print("    %-20s r_AB %+.3f   r_err %+.3f   rho(w,|err|) %+.3f" %
              (row, np.mean(corr[row]["r_AB"]), np.mean(corr[row]["r_err"]),
               np.mean(corr[row]["rho_w_err"])))
        out[row].update({k + "_percase": float(np.mean(v)) for k, v in corr[row].items()})'''
new2 = '''    for row in sorted(corr):
        g = lambda k: np.mean(corr[row][k]) if k in corr[row] else float("nan")
        print("    %-20s r_AB %+.3f   r_err %+.3f   rho(w,|err|) %+.3f" %
              (row, g("r_AB"), g("r_err"), g("rho_w_err")))
        out[row].update({k + "_percase": float(np.mean(v)) for k, v in corr[row].items()})'''
assert old2 in s, "print block not found"
s = s.replace(old2, new2)
io.open(p, "w", encoding="utf-8").write(s)
print("patched")

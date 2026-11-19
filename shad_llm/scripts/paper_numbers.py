"""Offline numbers for the paper (no API calls): rerun noise, label efficiency of the readout, Sonnet judge, length
correlations of judge vs humans.  python scripts/paper_numbers.py results/claude_hs2 > results/paper_<ds>.txt"""
import os, sys
import numpy as np, pandas as pd
sys.argv, ARGS = sys.argv[:1] + [sys.argv[1], "x"], sys.argv[1:]
exec(open(os.path.join(os.path.dirname(__file__), "claude_opt.py")).read().split("# ---------------------------------------------------------------- main loop")[0])
J = [x + "_j" for x in A]
ld = lambda n, s: frame(pd.read_csv(f"{R}/out/{n}_{s}.csv"), s)
tr = pd.concat([ld("v0", "fit"), ld("v0", "sel")], ignore_index=True)
te = ld("v0", "test")
H = te[A].to_numpy(float)
fmt = lambda v: " ".join(f"{x:6.3f}" for x in v) + f" | {np.nanmean(v):.3f}"
print(f"=== {os.path.basename(R)}  attrs {A}  train n={len(tr)} test n={len(te)}")
print("raw            ", fmt(per_task(H, te[J].to_numpy(float))))
print("readout        ", fmt(per_task(H, readout_scores(tr, te))))
# rerun noise: same prompt judged twice on test
if os.path.exists(f"{R}/out/v0r_test.csv"):
    tr_ = te.merge(frame(pd.read_csv(f"{R}/out/v0r_test.csv"), "test")[["id"] + J], on="id", suffixes=("", "_r"))
    Hr = tr_[A].to_numpy(float)
    a_ = per_task(Hr, tr_[J].to_numpy(float)); b_ = per_task(Hr, tr_[[c + "_r" for c in J]].to_numpy(float))
    rr = te[["id"]].merge(frame(pd.read_csv(f"{R}/out/v0r_test.csv"), "test"), on="id")
    print("rerun raw      ", fmt(b_), f"  |Δmean| vs first run {abs(np.nanmean(a_) - np.nanmean(b_)):.3f}")
    print("rerun readout  ", fmt(per_task(rr[A].to_numpy(float), readout_scores(tr, rr))))
    agree = np.mean([(tr_[c] == tr_[c + "_r"]).mean() for c in J])
    print(f"rerun exact score agreement {agree:.3f}")
# label efficiency
rng = np.random.default_rng(0)
for n in [25, 50, 100, 200, 400, 800]:
    if n > len(tr): break
    m = []
    for _ in range(10):
        try:
            m.append(np.nanmean(per_task(H, readout_scores(tr.iloc[rng.permutation(len(tr))[:n]].reset_index(drop=True), te))))
        except ValueError:  # a fold with a constant attribute (tiny samples)
            pass
    print(f"labels {n:4d}: readout mean {np.mean(m):.3f} ± {np.std(m):.3f}")
# length correlation (Spearman with z) on test
print("rho(human, len)", " ".join(f"{sp(te[x], te.z):+.2f}" for x in A))
print("rho(judge, len)", " ".join(f"{sp(te[x + '_j'], te.z):+.2f}" for x in A))
if os.path.exists(f"{R}/out/v0S_test.csv"):
    trS = pd.concat([ld("v0S", "fit"), ld("v0S", "sel")], ignore_index=True); teS = ld("v0S", "test")
    print("SONNET raw     ", fmt(per_task(teS[A].to_numpy(float), teS[J].to_numpy(float))))
    print("SONNET readout ", fmt(per_task(teS[A].to_numpy(float), readout_scores(trS, teS))))
    print("SONNET rho(judge,len)", " ".join(f"{sp(teS[x + '_j'], teS.z):+.2f}" for x in A))

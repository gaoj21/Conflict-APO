"""Final HS2 test numbers: raw judge vs judge + per-attribute length head (beta fitted on train = fit + sel, applied to
test), with a paired bootstrap over test rows for the difference to v0 raw.

  python scripts/claude_hs2_final.py
"""
import numpy as np, pandas as pd

D, A = "results/claude_hs2", ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
g = pd.read_csv(f"{D}/gold.csv")
load = lambda v, sp: g[g.split == sp].merge(pd.read_csv(f"{D}/out/{v}_{sp}.csv"), on="id", suffixes=("", "_j"))
GRID, PRIOR = np.linspace(-2, 2, 81), 0.02


def preds(v):
    tr, te = pd.concat([load(v, "fit"), load(v, "sel")]), load(v, "test")
    mu, sd = tr.z.mean(), tr.z.std()
    out = {}
    for a in A:
        zt = (tr.z - mu) / sd
        b = GRID[int(np.argmax([tr[a].corr(tr[a + "_j"] + x * zt, method="spearman") - PRIOR * x * x for x in GRID]))]
        out[a] = (te[a + "_j"].to_numpy(), (te[a + "_j"] + b * (te.z - mu) / sd).to_numpy(), b)
    return te, out


rk = lambda x: pd.Series(x).rank().to_numpy()
sp = lambda x, y: np.corrcoef(rk(x), rk(y))[0, 1]
te0, P0 = preds("v0"); te2, P2 = preds("v2")
assert (te0.id.to_numpy() == te2.id.to_numpy()).all()
arms = {"v0": {a: P0[a][0] for a in A}, "v0+head": {a: P0[a][1] for a in A},
        "v2": {a: P2[a][0] for a in A}, "v2+head": {a: P2[a][1] for a in A}}
rng = np.random.default_rng(0)
B = [rng.integers(0, len(te0), len(te0)) for _ in range(1000)]
print(f"{'test n=' + str(len(te0)):10s}" + "".join(f"{a[:11]:>12s}" for a in A) + f"{'mean':>8s}   mean diff vs v0 [95% CI]")
for k, p in arms.items():
    r = [sp(te0[a], p[a]) for a in A]
    line = f"{k:10s}" + "".join(f"{x:12.3f}" for x in r) + f"{np.mean(r):8.3f}"
    if k != "v0":
        d = [np.mean([sp(te0[a].to_numpy()[i], p[a][i]) - sp(te0[a].to_numpy()[i], arms["v0"][a][i]) for a in A]) for i in B]
        line += f"   {np.mean(r) - np.mean([sp(te0[a], arms['v0'][a]) for a in A]):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
    print(line)
print("head betas v0:", {a[:5]: P0[a][2] for a in A}, " v2:", {a[:5]: P2[a][2] for a in A})
print("judge~len vs human~len (test):", {a[:5]: f"{te2[a + '_j'].corr(te2.z, method='spearman'):+.2f}/{te2[a].corr(te2.z, method='spearman'):+.2f}" for a in A})

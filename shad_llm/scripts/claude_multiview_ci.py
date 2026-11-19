"""Paired bootstrap (test items) for scripts/claude_multiview.py: archive views vs self-ensemble at equal calls, and vs
the single final prompt.  python scripts/claude_multiview_ci.py results/claude_hs2r s2c1 s2c0 s1c2"""
import os, sys
import numpy as np, pandas as pd
ROOT, FIN, ADD = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.argv = sys.argv[:1] + [ROOT, "--select-only"]
src = open("scripts/claude_multiview.py").read().rsplit('if __name__ == "__main__":', 1)[0]
exec(src.replace("ARGV = sys.argv[1:]", "ARGV = [ROOT]"))
V = {n: view(n, ("fit", "sel", "test")) for n in [FIN] + ADD + [f"{FIN}_r1", f"{FIN}_r2"]}
def preds(views):
    tr = pd.concat([table(views, "fit"), table(views, "sel")], ignore_index=True); te = table(views, "test")
    return te.id.to_numpy(), te[A].to_numpy(float), decoder(tr, len(views))(te)
sets = {"single": [FIN], "arch2": [FIN] + ADD[:1], "arch3": [FIN] + ADD[:2], "self2": [FIN, f"{FIN}_r1"], "self3": [FIN, f"{FIN}_r1", f"{FIN}_r2"]}
P = {k: preds([V[n] for n in v]) for k, v in sets.items()}
ids = sorted(set.intersection(*[set(p[0]) for p in P.values()]))
al = {k: (pd.Index(p[0]).get_indexer(ids)) for k, p in P.items()}
H = P["single"][1][al["single"]]
rng = np.random.default_rng(0); B = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
def m(k, b=None):
    pr = P[k][2][al[k]]; b = np.arange(len(ids)) if b is None else b
    return np.nanmean(per_task(H[b], pr[b]))
for a_, b_ in [("arch2", "self2"), ("arch3", "self3"), ("arch3", "single"), ("self3", "single")]:
    d = [m(a_, b) - m(b_, b) for b in B]
    print(f"{os.path.basename(ROOT)} {a_} - {b_}: {m(a_) - m(b_):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] (n={len(ids)})")

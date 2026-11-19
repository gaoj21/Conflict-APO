"""Routing with empirical-Bayes shrinkage toward the run's final prompt (offline, cached judgments only).

Per attribute k and archive version v: d_vk = CV Spearman advantage of v's readout over the final prompt's readout
(5-fold within-fold CV on fit + sel pooled, paired), se_vk = its paired-bootstrap SE. Advantages are shrunk toward
0 with a normal prior whose variance is estimated from all (v, k): tau2 = max(var(d) - mean(se^2), 0),
d*_vk = d_vk * tau2 / (tau2 + se_vk^2). Route attribute k to argmax_v d*_vk (the final prompt itself has d* = 0).
No threshold: with noisy advantages tau2 -> 0 and every attribute stays with the final prompt.

  python scripts/claude_route_shrink.py results/claude_se
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

ARGV = sys.argv[1:]
SEL_ONLY = "--sel" in ARGV
sys.argv = sys.argv[:1] + [ARGV[0]]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_portfolio.py")
exec(open(__file__).read().split('if __name__ == "__main__":')[0]
     .replace('sys.argv[:1] + [ARGV[0], "portfolio"] + ARGV[1:]', 'sys.argv[:1] + [ARGV[0], "route"]'))
TEXT = f"{R}/opt/text"
SPLITS = ["sel"] if SEL_ONLY else ["fit", "sel"]  # --sel: route on sel only (the optimizer never saw sel items)
FOLDS, B = 5, 200


def load(name, split):
    f = f"{R}/out/v0_{split}.csv" if name == "v0" else f"{TEXT}/out/{name}_{split}.csv"
    return frame(pd.read_csv(f), split)


def oof(name, ids):
    """Out-of-fold readout predictions of one version on the pooled fit + sel rows (aligned to ids)."""
    d = pd.concat([load(name, s_) for s_ in SPLITS], ignore_index=True).set_index("id").loc[ids].reset_index()
    fold = np.arange(len(d)) % FOLDS
    P = np.zeros((len(d), K))
    for f in range(FOLDS):
        mt = {"fit": d[fold != f].reset_index(drop=True), "sel": d[fold == f].reset_index(drop=True)}
        P[fold == f] = decoder(table([mt], "fit"), 1)(table([mt], "sel"))
    return P, d[A].to_numpy(float), fold


def rs(x, y):
    return np.corrcoef(rankdata(x), rankdata(y))[0, 1]


names = ["v0"] + sorted({os.path.basename(f)[:-8] for f in glob.glob(f"{TEXT}/out/s*c*_sel.csv")})
fin = open(f"{TEXT}/final.txt").read().strip()
ids = None
for n in names:
    d = pd.concat([load(n, s_) for s_ in SPLITS])
    ids = set(d.id) if ids is None else ids & set(d.id)
ids = sorted(ids)
OOF = {n: oof(n, ids) for n in names}
H, fold = OOF[fin][1], OOF[fin][2]
rng = np.random.default_rng(0)
boot = [np.concatenate([rng.choice(np.flatnonzero(fold == f), (fold == f).sum()) for f in range(FOLDS)]) for _ in range(B)]


def cv(P, k, idx=None):
    idx = np.arange(len(H)) if idx is None else idx
    return np.nanmean([rs(H[idx][fold[idx] == f, k], P[idx][fold[idx] == f, k]) for f in range(FOLDS)])


D, SE = {}, {}
for n in names:
    if n == fin:
        continue
    D[n] = np.array([cv(OOF[n][0], k) - cv(OOF[fin][0], k) for k in range(K)])
    SE[n] = np.array([np.std([cv(OOF[n][0], k, b) - cv(OOF[fin][0], k, b) for b in boot]) for k in range(K)])
d = np.array(list(D.values())); se = np.array(list(SE.values()))
tau2 = max(np.var(d) - np.mean(se ** 2), 0.0)
shr = d * tau2 / (tau2 + se ** 2)
vs = list(D)
route = [vs[int(np.argmax(shr[:, k]))] if shr[:, k].max() > 0 else fin for k in range(K)]
print(f"=== {os.path.basename(R)}: final {fin}; tau = {np.sqrt(tau2):.4f}, typical se = {np.sqrt(np.mean(se ** 2)):.4f}")
for k in range(K):
    i = int(np.argmax(d[:, k]))
    print(f"  {A[k]:12s} raw best {vs[i]} d={d[i, k]:+.3f} (se {se[i, k]:.3f})  ->  route {route[k]}"
          f" (shrunk d* {shr[:, k].max():+.4f})")
# test: attribute k from its routed version's readout (fit + sel -> test), if that version was judged on test
def test_readout(n):
    base = f"{R}/out" if n == "v0" else f"{TEXT}/out"
    if not os.path.exists(f"{base}/{n}_test.csv"):
        return None
    mt = {"fit": pd.concat([load(n, "fit"), load(n, "sel")], ignore_index=True), "sel": frame(pd.read_csv(f"{base}/{n}_test.csv"), "test")}
    T_ = table([mt], "sel")
    return per_task(T_[A].to_numpy(float), decoder(table([mt], "fit"), 1)(T_))
need = [n for n in dict.fromkeys(route) if test_readout(n) is None]
if need:
    print("  not judged on test yet:", need)
else:
    r = np.array([test_readout(n)[k] for k, n in enumerate(route)])
    rf = test_readout(fin)
    print("  test  final " + " ".join(f"{x:.3f}" for x in rf) + f" | {np.nanmean(rf):.3f}")
    print("  test  routed" + " ".join(f" {x:.3f}" for x in r) + f" | {np.nanmean(r):.3f}   ({len(set(route))} calls)")

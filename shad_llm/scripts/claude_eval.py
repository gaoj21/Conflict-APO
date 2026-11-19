"""Test-set evaluation of a judge version for any results/claude_<ds> root: raw scores, per-task length head,
linear readout (all judge scores + length -> each human attribute, ridge, fitted on train = fit + sel), its rank-2
truncation; paired bootstrap CI vs raw; SVD structure of judge vs human scores; rerun noise if <ver>r exists.

  python scripts/claude_eval.py results/claude_br v0 [v0S ...]
"""
import os, sys
import numpy as np, pandas as pd

R, VERS = sys.argv[1], sys.argv[2:]
A = open(f"{R}/attrs.txt").read().split()
g = pd.read_csv(f"{R}/gold.csv")
GRID, RIDGE = np.linspace(-2, 2, 81), 10.0
std = lambda X: (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
sp = lambda x, y: pd.Series(np.asarray(x, float)).corr(pd.Series(np.asarray(y, float)), method="spearman")


def ld(v, sps):
    return g[g.split.isin(sps)].merge(pd.concat([pd.read_csv(f"{R}/out/{v}_{s}.csv") for s in sps]), on="id",
                                       suffixes=("", "_j")).dropna(subset=[a + "_j" for a in A])


def readouts(tr, te):
    J = lambda d: d[[a + "_j" for a in A]].to_numpy(float)
    mu, sd = tr.z.mean(), tr.z.std()
    P = {"raw": J(te)}
    hd = []
    for a in A:
        zt = (tr.z - mu) / sd
        b = GRID[int(np.argmax([sp(tr[a], tr[a + "_j"] + x * zt) - 0.02 * x * x for x in GRID]))]
        hd.append(te[a + "_j"] + b * (te.z - mu) / sd)
    P["length head"] = np.c_[hd].T
    m_, s_ = J(tr).mean(0), np.where(J(tr).std(0) > 0, J(tr).std(0), 1)
    Xtr, Xte = np.c_[(J(tr) - m_) / s_, (tr.z - mu) / sd], np.c_[(J(te) - m_) / s_, (te.z - mu) / sd]
    W = np.linalg.solve(Xtr.T @ Xtr + RIDGE * np.eye(Xtr.shape[1]), Xtr.T @ std(tr[A].to_numpy(float)))
    P["linear readout"] = Xte @ W
    _, _, Vt = np.linalg.svd(Xtr @ W, full_matrices=False)
    P["rank-2 readout"] = Xte @ (W @ Vt[:2].T @ Vt[:2])
    P["anchored readout"] = anchored(Xtr, tr[A].to_numpy(float), Xte)
    # back to the human scale and rounded to its integer levels (ties again where the readout is unsure)
    Htr = tr[A].to_numpy(float)
    lo, hi = Htr.min(0), Htr.max(0)
    for k in ("linear readout", "anchored readout"):
        P[k + " (rounded)"] = np.clip(np.round(P[k] * Htr.std(0) + Htr.mean(0)), lo, hi)
    return P


LAMS = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1e5]


def anchored(Xtr, Htr, Xte, folds=5):
    """Per task: ridge from all judge scores + length, shrunk toward the task's own score (w0 = own-score slope)
    instead of toward 0; the shrinkage strength is chosen by K-fold CV Spearman on train. Very strong shrinkage
    gives back the raw score's ranking, so the readout departs from it only as far as the data support."""
    Ys = std(Htr)
    fold = np.arange(len(Xtr)) % folds
    out = np.zeros((len(Xte), Ys.shape[1]))
    for k in range(Ys.shape[1]):
        def fit(X, y, lam):
            w0 = np.zeros(X.shape[1]); w0[k] = (X[:, k] @ y) / (X[:, k] @ X[:, k])
            return w0 if lam is None else np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y + lam * w0)
        cv = []
        for lam in LAMS + [None]:  # None = the raw score alone (ties kept)
            # within-fold Spearman, averaged: pooled folds would break the raw score's ties via per-fold slopes
            cv.append(np.nanmean([sp(Ys[fold == f, k], Xtr[fold == f] @ fit(Xtr[fold != f], Ys[fold != f, k], lam))
                                  for f in range(folds)]))
        out[:, k] = Xte @ fit(Xtr, Ys[:, k], (LAMS + [None])[int(np.nanargmax(cv))])
    return out


rng = np.random.default_rng(0)
for v in VERS:
    tr, te = ld(v, ["fit", "sel"]), ld(v, ["test"])
    H = te[A].to_numpy(float)
    B = [rng.integers(0, len(te), len(te)) for _ in range(1000)]
    P = readouts(tr, te)
    print(f"=== {os.path.basename(R)} {v}  test n={len(te)}  (Spearman: {' '.join(a[:6] for a in A)} | mean)")
    for k, p in P.items():
        r = [sp(H[:, j], p[:, j]) for j in range(len(A))]
        line = f"  {k:15s}" + " ".join(f"{x:6.3f}" for x in r) + f" | {np.mean(r):.3f}"
        if k != "raw":
            d = [np.mean([sp(H[i, j], p[i, j]) - sp(H[i, j], P['raw'][i, j]) for j in range(len(A))]) for i in B]
            line += f"  {np.mean(r) - np.mean([sp(H[:, j], P['raw'][:, j]) for j in range(len(A))]):+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]"
        print(line)
    if os.path.exists(f"{R}/out/{v}r_test.csv"):
        te2 = ld(v + "r", ["test"])
        print("  rerun raw      " + " ".join(f"{sp(te2[a], te2[a + '_j']):6.3f}" for a in A))
    print("  judge~len      " + " ".join(f"{sp(te[a + '_j'], te.z):+6.2f}" for a in A) + "   human~len " +
          " ".join(f"{sp(te[a], te.z):+.2f}" for a in A))
    for name, X in (("human", H), ("judge", P["raw"])):
        U, s, Vt = np.linalg.svd(std(X), full_matrices=False)
        sg = [np.sign(Vt[k].sum()) or 1 for k in range(2)]
        print(f"  SVD {name:5s} var {np.round(s ** 2 / np.sum(s ** 2), 2)}  comp1 {np.round(Vt[0] * sg[0], 2)} (len {sp(U[:, 0] * sg[0], te.z):+.2f})"
              f"  comp2 {np.round(Vt[1] * sg[1], 2)} (len {sp(U[:, 1] * sg[1], te.z):+.2f})")

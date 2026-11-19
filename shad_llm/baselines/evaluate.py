"""Test Spearman of every method's judge, raw and through the per-attribute readout (the decoder), plus the two
post-hoc baselines that use the same labels: own-score calibration (isotonic) and length debiasing (own score + log
length, the readout restricted to the attribute's own score).

The readout is the one in scripts/claude_opt.py: per attribute, ridge on [all judge scores, z] shrunk toward the
attribute's own-score slope, lambda (or the raw score) chosen by 5-fold within-fold CV Spearman; fitted on fit + sel.
  .venv/bin/python baselines/evaluate.py [ds ...]
"""
import glob, os, sys

import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gem import DS, OUT  # noqa: E402

LAMS = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1e5]
sp = lambda x, y: pd.Series(np.asarray(x, float)).corr(pd.Series(np.asarray(y, float)), method="spearman")


def anchored_fit(X, Y, folds=5):
    """Same as scripts/claude_opt.py: per attribute k, ridge shrunk toward the own-score slope; option 'raw'."""
    fold = np.arange(len(X)) % folds
    Ws = []
    for k in range(Y.shape[1]):
        y = Y[:, k]

        def fit(Xa, ya, lam):
            w0 = np.zeros(Xa.shape[1]); w0[k] = (Xa[:, k] @ ya) / max(Xa[:, k] @ Xa[:, k], 1e-9)
            if lam is None:
                return w0
            return np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya + lam * w0)

        opts = LAMS + [None]
        cv = [np.nanmean([sp(y[fold == f], X[fold == f] @ fit(X[fold != f], y[fold != f], lam)) for f in range(folds)])
              for lam in opts]
        Ws.append(fit(X, y, opts[int(np.nanargmax(cv))]))
    W = np.array(Ws).T
    return lambda Xn: Xn @ W


def load(ds, d):
    S = {s: pd.read_csv(f"{d}/scores_{s}.csv") for s in ("fit", "sel", "test")}
    tr = pd.concat([S["fit"], S["sel"]], ignore_index=True)
    return tr, S["test"]


def scores(ds, d):
    tr, te = load(ds, d)
    A = ds.A
    mu = tr[A].mean()
    Xtr, Xte = tr[A].fillna(mu).to_numpy(float), te[A].fillna(mu).to_numpy(float)
    m, s = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1)
    ztr, zte = ds.gold.loc[tr.id, "z"].to_numpy(float), ds.gold.loc[te.id, "z"].to_numpy(float)
    zm, zs = ztr.mean(), ztr.std()
    Ftr = np.c_[(Xtr - m) / s, (ztr - zm) / zs]; Fte = np.c_[(Xte - m) / s, (zte - zm) / zs]
    Ytr = ds.human(tr.id); Ytr = (Ytr - Ytr.mean(0)) / Ytr.std(0)
    H = ds.human(te.id)
    raw = np.array([sp(H[:, k], Xte[:, k]) for k in range(len(A))])
    ro = anchored_fit(Ftr, Ytr)(Fte)
    readout = np.array([sp(H[:, k], ro[:, k]) for k in range(len(A))])
    iso, lenk = [], []
    for k in range(len(A)):
        f = IsotonicRegression(out_of_bounds="clip").fit(Xtr[:, k], Ytr[:, k])
        iso.append(sp(H[:, k], f.predict(Xte[:, k])))
        g = anchored_fit(Ftr[:, [k, -1]], Ytr[:, [k]])
        lenk.append(sp(H[:, k], g(Fte[:, [k, -1]])[:, 0]))
    return dict(raw=raw, readout=readout, isotonic=np.array(iso), own_len=np.array(lenk))


def main(names):
    rows = []
    for n in names:
        ds = DS(n)
        for d in sorted(glob.glob(f"{OUT}/{n}/*/seed*")):
            if not all(os.path.exists(f"{d}/scores_{s}.csv") for s in ("fit", "sel", "test")):
                continue
            r = scores(ds, d)
            method, seed = d.split("/")[-2], d.split("/")[-1]
            rows.append(dict(ds=n, method=method, seed=seed, **{k: float(np.nanmean(v)) for k, v in r.items()}))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no finished runs"); return
    df.to_csv(f"{OUT}/results.csv", index=False)
    g = df.groupby(["ds", "method"])
    t = g[["raw", "isotonic", "own_len", "readout"]].mean().round(3)
    t["sd_raw"], t["sd_readout"], t["n"] = g.raw.std().round(3), g.readout.std().round(3), g.size()
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(t)


if __name__ == "__main__":
    main(sys.argv[1:] or ["hs2", "hanna", "ell", "merlin", "se"])

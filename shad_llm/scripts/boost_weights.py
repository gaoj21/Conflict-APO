"""Boosting weights for APO: how wrong the current ShAD judge (over the scores of the archive's prompts) is on each
labelled item, from 5-fold cross-fitted predictions on fit + sel. w_i = mean over attributes of |standardized
residual|, normalized to mean 1 (smooth, no threshold).
  .venv/bin/python scripts/boost_weights.py <ds> <out.csv> <method_dir> [<method_dir> ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
from gem import DS  # noqa: E402
from evaluate import anchored_fit  # noqa: E402

ds, out, dirs = DS(sys.argv[1]), sys.argv[2], sys.argv[3:]
S = [pd.concat([pd.read_csv(f"{d}/scores_{s}.csv") for s in ("fit", "sel")]).set_index("id")[ds.A] for d in dirs]
ids = S[0].index
X = np.hstack([s.loc[ids].to_numpy(float) for s in S]); X = np.where(np.isnan(X), np.nanmean(X, 0), X)
z = ds.gold.loc[ids, "z"].to_numpy(float)
Y = ds.human(ids); Y = (Y - Y.mean(0)) / Y.std(0)
fold = np.random.default_rng(0).permutation(len(ids)) % 5
P = np.zeros_like(Y)
for f in range(5):
    tr, te = fold != f, fold == f
    m, s = X[tr].mean(0), np.where(X[tr].std(0) > 0, X[tr].std(0), 1)
    F = np.c_[(X - m) / s, (z - z[tr].mean()) / z[tr].std()]
    P[te] = anchored_fit(F[tr], Y[tr])(F[te])
w = np.abs(Y - P).mean(1); w = w / w.mean()
pd.DataFrame(dict(id=ids, w=w)).to_csv(out, index=False)
print(f"{len(w)} weights, mean 1, sd {w.std():.2f}, max {w.max():.2f}")

"""ShAD teacher labels for distillation: 5-fold cross-fitted ShAD predictions (human scale, clipped) on fit + sel,
from the scores of a judge run. APO then optimizes the prompt toward these instead of the human scores
(GEM_TEACHER=<csv> replaces the gold scores of these items; test keeps the human scores).
--mix a: soft labels a * teacher + (1 - a) * human (a = 1: pure teacher).
  .venv/bin/python scripts/make_teacher.py <ds> <method_dir> <out.csv> [--mix 1]"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
from gem import DS  # noqa: E402
from evaluate import anchored_fit  # noqa: E402

ds, d, out = DS(sys.argv[1]), sys.argv[2], sys.argv[3]
mix = float(sys.argv[sys.argv.index("--mix") + 1]) if "--mix" in sys.argv else 1.0
S = pd.concat([pd.read_csv(f"{d}/scores_{s}.csv") for s in ("fit", "sel")]).set_index("id")[ds.A]
ids = S.index; X = S.to_numpy(float); X = np.where(np.isnan(X), np.nanmean(X, 0), X)
z = ds.gold.loc[ids, "z"].to_numpy(float)
Y = ds.human(ids); mu, sd = Y.mean(0), Y.std(0)
fold = np.random.default_rng(0).permutation(len(ids)) % 5
P = np.zeros_like(Y)
for f in range(5):
    tr, te = fold != f, fold == f
    m, s = X[tr].mean(0), np.where(X[tr].std(0) > 0, X[tr].std(0), 1)
    F = np.c_[(X - m) / s, (z - z[tr].mean()) / z[tr].std()]
    P[te] = anchored_fit(F[tr], (Y[tr] - mu) / sd)(F[te]) * sd + mu
P = np.clip(mix * P + (1 - mix) * Y, ds.LO, ds.HI)
pd.DataFrame(P, index=ids, columns=ds.A).reset_index().to_csv(out, index=False)
print(f"{len(ids)} teacher rows; corr with human per attr:",
      np.round([np.corrcoef(P[:, k], Y[:, k])[0, 1] for k in range(len(ds.A))], 2))

"""How the optimized prompt differs from v0, for APO on human vs content labels (test items):
  len-move: per attribute, Spearman between the change of the judge's score (final - v0) and log length z
            (how much the edit moved the judge along the length axis);
  d raw / d ShAD: per-attribute change of test Spearman x100 vs v0, raw and after ShAD;
  conflicts: significant (> 2 paired-bootstrap SE) ups and downs among the attributes.
  .venv/bin/python scripts/content_conflicts.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
from gem import DS, OUT  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402


def load(ds, m):
    d = f"{OUT}/{ds.name}/{m}/seed0"
    S = {s: pd.read_csv(f"{d}/scores_{s}.csv") for s in ("fit", "sel", "test")}
    return pd.concat([S["fit"], S["sel"]], ignore_index=True), S["test"]


def shad_pred(ds, tr, te):
    A = ds.A
    Xtr, Xte = tr[A].to_numpy(float), te[A].to_numpy(float)
    mu = np.nanmean(Xtr, 0); Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    m, s = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1)
    z = ds.gold["z"]; ztr, zte = z.loc[tr.id].to_numpy(float), z.loc[te.id].to_numpy(float)
    F = lambda X, zz: np.c_[(X - m) / s, (zz - ztr.mean()) / ztr.std()]
    Y = ds.human(tr.id)
    return anchored_fit(F(Xtr, ztr), (Y - Y.mean(0)) / Y.std(0))(F(Xte, zte)), Xte


def boot_d(H, P0, P1, B=300, seed=0):
    rng = np.random.default_rng(seed); n = len(H); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        out.append([sp(H[i, k], P1[i, k]) - sp(H[i, k], P0[i, k]) for k in range(H.shape[1])])
    return np.nanstd(np.array(out), 0)


rows = []
for n in ["hs2", "ell", "hanna"]:
    ds = DS(n)
    tr0, te0 = load(ds, "v0"); P0, X0 = shad_pred(ds, tr0, te0)
    for base in ("dspy_mipro", "textgrad_ccc"):
        for lab, m in (("human", base), ("content", base + "_content")):
            tr1, te1 = load(ds, m)
            te1 = te1.set_index("id").loc[te0.id].reset_index(); tr1 = tr1.set_index("id").loc[tr0.id].reset_index()
            P1, X1 = shad_pred(ds, tr1, te1)
            H = ds.human(te0.id); z = ds.gold.loc[te0.id, "z"].to_numpy(float)
            mv = [sp(X1[:, k] - X0[:, k], z) for k in range(len(ds.A))]
            dr = np.array([sp(H[:, k], X1[:, k]) - sp(H[:, k], X0[:, k]) for k in range(len(ds.A))])
            dd = np.array([sp(H[:, k], P1[:, k]) - sp(H[:, k], P0[:, k]) for k in range(len(ds.A))])
            se_r, se_d = boot_d(H, X0, X1), boot_d(H, P0, P1)
            conf = lambda d, se: int(((d > 2 * se).any() and (d < -2 * se).any()))
            rows.append(dict(ds=n, method=base, labels=lab, len_move=np.nanmean(np.abs(mv)),
                             len_move_max=np.nanmax(np.abs(mv)),
                             sig_up_raw=int((dr > 2 * se_r).sum()), sig_down_raw=int((dr < -2 * se_r).sum()),
                             conflict_raw=conf(dr, se_r),
                             sig_up_shad=int((dd > 2 * se_d).sum()), sig_down_shad=int((dd < -2 * se_d).sum()),
                             conflict_shad=conf(dd, se_d)))
df = pd.DataFrame(rows)
with pd.option_context("display.width", 200):
    print(df.round(3).to_string(index=False))
    print(df.groupby("labels")[["len_move", "sig_down_raw", "conflict_raw", "sig_down_shad", "conflict_shad"]].mean().round(3))

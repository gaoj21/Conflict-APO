"""Test Spearman x100 of a feature-APO judge: raw attribute scores, ShAD over attributes + length (as in the main
table), and ShAD over attributes + checklist answers + length. Decoders fitted on fit + sel.
  .venv/bin/python scripts/eval_features.py <ds> [method]"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
from gem import DS, OUT  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ds = DS(sys.argv[1]); m = sys.argv[2] if len(sys.argv) > 2 else "features_J12"
d = f"{OUT}/{ds.name}/{m}/seed0"
S = {s: pd.read_csv(f"{d}/scores_{s}.csv") for s in ("fit", "sel", "test")}
tr, te = pd.concat([S["fit"], S["sel"]], ignore_index=True), S["test"]
A = ds.A; extra = [c for c in te.columns if c not in A + ["id", "in_tok", "out_tok", "reason", "error"]]
print("checklist items:", len(extra), "| unparsed test rows:", int(te[A + extra].isna().any(axis=1).sum()))
H = ds.human(te.id); Y = ds.human(tr.id); Y = (Y - Y.mean(0)) / Y.std(0)
z = ds.gold["z"]; ztr, zte = z.loc[tr.id].to_numpy(float), z.loc[te.id].to_numpy(float)

def shad(cols):
    Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isfinite(mu), mu, 0)
    Xtr, Xte = np.where(np.isnan(Xtr), mu, Xtr), np.where(np.isnan(Xte), mu, Xte)
    m_, s_ = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1)
    F = lambda X, zz: np.c_[(X - m_) / s_, (zz - ztr.mean()) / ztr.std()]
    P = anchored_fit(F(Xtr, ztr), Y)(F(Xte, zte))
    return np.array([sp(H[:, k], P[:, k]) for k in range(len(A))])

raw = np.array([sp(H[:, k], te[A[k]].to_numpy(float)) for k in range(len(A))])
r1, r2 = shad(A), shad(A + extra)
print(pd.DataFrame({"raw": raw, "ShAD (attrs+len)": r1, "ShAD (attrs+checklist+len)": r2}, index=A).mul(100).round(1).to_string())
print(f"mean: raw {100*np.nanmean(raw):.1f} | ShAD attrs {100*np.nanmean(r1):.1f} | ShAD +checklist {100*np.nanmean(r2):.1f}")

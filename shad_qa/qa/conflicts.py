"""Claim 1 of the QA pilot: no single behavior is best for every category (a fixed prompt creates conflicts).
Per category: accuracy of every behavior; vs the behavior best on average, a category is up / down when the paired
difference exceeds two bootstrap SE.
  .venv/bin/python qa/conflicts.py [train|test]
"""
import os, sys

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
split = sys.argv[1] if len(sys.argv) > 1 else "train"
split = split if "/" in split else f"mmlupro/{split}"
R = pd.read_csv(f"{ROOT}/results/gemma_qa/{split}/runs.csv")
acc = R.pivot_table(index="category", columns="action", values="correct", aggfunc="mean") * 100
tok = R.groupby("action").out_tok.mean()
best = acc.mean().idxmax()
W = R.pivot_table(index=["category", "qid"], columns="action", values="correct")
rng = np.random.default_rng(0)
rows = []
for c, g in W.groupby(level=0):
    for act in acc.columns:
        if act == best:
            continue
        d = (g[act] - g[best]).to_numpy()
        se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        rows.append(dict(category=c, action=act, diff=100 * d.mean(), sig=abs(d.mean()) > 2 * se))
S = pd.DataFrame(rows)
print(f"== {split}: accuracy x100 per category (best on average: {best})")
print(acc.round(1).to_string())
print("\nmean      " + "  ".join(f"{a} {acc[a].mean():.1f}" for a in acc.columns))
print("out tokens " + "  ".join(f"{a} {tok[a]:.0f}" for a in acc.columns))
print(f"\nbest behavior per category: {acc.idxmax(axis=1).value_counts().to_dict()}")
print(f"per-category best (oracle over behaviors): {acc.max(axis=1).mean():.1f} vs fixed {best}: {acc[best].mean():.1f}")
for act in S.action.unique():
    s = S[S.action == act]
    up, dn = s[(s["diff"] > 0) & s.sig].category.tolist(), s[(s["diff"] < 0) & s.sig].category.tolist()
    print(f"{act} vs {best}: significantly better in {up or '-'}; worse in {dn or '-'}")

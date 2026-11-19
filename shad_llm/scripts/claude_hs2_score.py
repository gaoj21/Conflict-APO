"""Score Claude-judge CSVs on HelpSteer2SafeG: per-attribute Spearman (primary), judge vs human rank correlation with
response length (the spurious axis), and partial Spearman given length.

  python scripts/claude_hs2_score.py v1 [v0 ...] [--split sel]
"""
import sys
import numpy as np, pandas as pd

D, ATTR = "results/claude_hs2", ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
args = [x for x in sys.argv[1:] if not x.startswith("--")]
split = next((x.split("=")[1] for x in sys.argv if x.startswith("--split=")), "sel")
gold = pd.read_csv(f"{D}/gold.csv")
gold = gold[gold.split == split]


def partial(a, b, z):  # Spearman of a and b controlling for z
    r = lambda x: pd.Series(x).rank().to_numpy()
    a, b, z = r(a), r(b), r(z)
    res = lambda x: x - np.polyval(np.polyfit(z, x, 1), z)
    return np.corrcoef(res(a), res(b))[0, 1]


print(f"{'':6s} " + " ".join(f"{a[:6]:>18s}" for a in ATTR) + "    (rho | judge~len human~len | partial|len)")
for v in args:
    m = gold.merge(pd.read_csv(f"{D}/out/{v}_{split}.csv"), on="id", suffixes=("", "_j"))
    cells = []
    for a in ATTR:
        d = m[[a, f"{a}_j", "z"]].dropna()
        rho = d[a].corr(d[f"{a}_j"], method="spearman")
        jl, hl = d[f"{a}_j"].corr(d.z, method="spearman"), d[a].corr(d.z, method="spearman")
        cells.append(f"{rho:.3f}|{jl:+.2f}{hl:+.2f}|{partial(d[a], d[f'{a}_j'], d.z):.2f}")
    print(f"{v:6s} " + " ".join(f"{c:>18s}" for c in cells) + f"   n={len(m)}")

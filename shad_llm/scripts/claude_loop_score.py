"""Score Claude-judge CSVs against gold: per-kind accuracy, P(pick longer), axis / content vs a reference version.

  python scripts/claude_loop_score.py v1 [--ref v0] [--split fit,sel] [--fails 8]
"""
import argparse, glob
import numpy as np, pandas as pd

D = "results/claude_loop"
ap = argparse.ArgumentParser()
ap.add_argument("ver"); ap.add_argument("--ref"); ap.add_argument("--split", default="fit,sel"); ap.add_argument("--fails", type=int, default=0)
a = ap.parse_args()
gold = pd.read_csv(f"{D}/gold.csv")


def load(v):
    fs = [f for f in glob.glob(f"{D}/out/{v}_*.csv")]
    o = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in fs]).drop_duplicates("id")
    m = gold.merge(o, on="id", how="inner")
    m["verdict"] = pd.to_numeric(m.verdict, errors="coerce")
    m["ok"] = m.verdict == m.preference
    m["picks_longer"] = m.verdict == np.where(m.z > 0, 1, 2)
    return m


def summ(m):
    L, S = m.ok[m.kind == "longer"].mean(), m.ok[m.kind == "shorter"].mean()
    return dict(n=int(m.verdict.notna().sum()), L=L, S=S, mean=(L + S) / 2, p_longer=m.picks_longer[m.verdict.notna()].mean())


M = load(a.ver); R = load(a.ref) if a.ref else None
for sp in a.split.split(","):
    m = M[(M.split == sp) & M.id.isin(M.id[M.verdict.notna()])]  # only judged rows
    if m.verdict.notna().sum() == 0:
        continue
    s = summ(m)
    line = f"{a.ver} {sp:5s} n={s['n']:3d}/{len(m)}  L {s['L']:.3f}  S {s['S']:.3f}  mean {s['mean']:.3f}  P(longer) {s['p_longer']:.3f}"
    if R is not None:
        r = R[(R.split == sp) & R.id.isin(m.id[m.verdict.notna()])]
        if r.verdict.notna().sum():
            q = summ(r); dL, dS = s["L"] - q["L"], s["S"] - q["S"]
            line += f"  | vs {a.ref}: axis {100 * (dL - dS) / 2:+.1f}  content {100 * (dL + dS) / 2:+.1f}"
    print(line)
if a.fails:  # length-matched failures on fit, weight exp(-z^2/2h^2)
    m = M[(M.split == "fit") & M.verdict.notna() & ~M.ok]
    h = 0.5 * gold.z.std()
    pick = m.sample(min(a.fails, len(m)), weights=np.exp(-m.z ** 2 / (2 * h * h)), random_state=0)
    print(pick[["id", "kind", "w1", "w2", "preference", "verdict", "reason"]].to_string(index=False))

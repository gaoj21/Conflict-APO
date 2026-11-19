"""Offline: ShAD over the scores of several prompts (v0 + a method's final prompt) vs. the final prompt alone."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
from gem import DS
from evaluate import anchored_fit, sp
ROOTS = {"gemma": "results/gemma", "luna": "wgc_gpt/results/gpt56luna", "nano": "wgc_gpt/results/gpt5nano"}
METH = ["fewshot", "opro", "textgrad_ccc", "textgrad_single", "dspy_gepa", "dspy_mipro"]

def load(root, ds, m):
    d = f"{root}/{ds.name}/{m}/seed0"
    S = [pd.read_csv(f"{d}/scores_{s}.csv").set_index("id")[ds.A] for s in ("fit", "sel", "test")]
    return pd.concat(S[:2]), S[2]

def shad(ds, parts):
    """parts: list of (train_df, test_df) of judge scores; the anchor is the LAST part's own score."""
    tr_ids, te_ids = parts[-1][0].index, parts[-1][1].index
    Xtr = np.hstack([p[0].loc[tr_ids].to_numpy(float) for p in parts]); Xte = np.hstack([p[1].loc[te_ids].to_numpy(float) for p in parts])
    mu = np.nanmean(Xtr, 0); Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    m, s = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1)
    z = ds.gold["z"]; ztr, zte = z.loc[tr_ids].to_numpy(float), z.loc[te_ids].to_numpy(float)
    K = len(ds.A); last = slice(len(parts) * K - K, len(parts) * K)
    # put the anchored (own) scores first so anchored_fit's e_k anchor is the final prompt's own score
    order = list(range(last.start, last.stop)) + [i for i in range(len(parts) * K) if not last.start <= i < last.stop]
    F = lambda X, zz: np.c_[((X - m) / s)[:, order], (zz - ztr.mean()) / ztr.std()]
    Y = ds.human(tr_ids); Y = (Y - Y.mean(0)) / Y.std(0)
    P = anchored_fit(F(Xtr, ztr), Y)(F(Xte, zte)); H = ds.human(te_ids)
    return 100 * np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)])

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
for j, r in ROOTS.items():
    r = f"{root_dir}/{r}"; rows = []
    for n in ["hs2", "hanna", "ell", "merlin"]:
        ds = DS(n); v0 = load(r, ds, "v0")
        for m in METH:
            try: f = load(r, ds, m)
            except FileNotFoundError: continue
            rows.append(dict(ds=n, m=m, final=shad(ds, [f]), stack=shad(ds, [v0, f])))
        rows.append(dict(ds=n, m="v0", final=shad(ds, [v0]), stack=np.nan))
    df = pd.DataFrame(rows)
    t = df.pivot_table(index="m", columns="ds", values=["final", "stack"]).round(1)
    print(f"===== {j}"); print(df.groupby("m")[["final", "stack"]].mean().round(1).assign(gain=lambda x: (x["stack"] - x["final"]).round(1)))

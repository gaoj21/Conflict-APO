"""Test Spearman (x100, mean over attributes) of ShAD over the scores of one or several prompts, gpt-6-luna runs.
Rows: v0, v0 + its re-judging (noise control), TextGrad CCC, v0 + TextGrad, archive APO (ShAD over all its prompts).
Every decoder is fitted on fit + sel and anchored to the first listed prompt's own score.
  .venv/bin/python scripts/eval_archive.py [root]
"""
import glob, json, os, sys

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
from gem import DS, ROOT  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/lanyi_gpt-6-luna"


def load(ds, files):
    S = [pd.read_csv(files[s]).set_index("id")[ds.A] for s in ("fit", "sel", "test")]
    return pd.concat(S[:2]), S[2]


def run_files(ds, method):
    d = f"{OUT}/{ds.name}/{method}/seed0"
    return {s: f"{d}/scores_{s}.csv" for s in ("fit", "sel", "test")}


def evaluate(ds, parts):
    """parts: list of (train, test) score frames; returns (raw of first part, ShAD) test Spearman x100."""
    tr_ids, te_ids = parts[0][0].index, parts[0][1].index
    Xtr = np.hstack([p[0].loc[tr_ids].to_numpy(float) for p in parts])
    Xte = np.hstack([p[1].loc[te_ids].to_numpy(float) for p in parts])
    mu = np.nanmean(Xtr, 0); Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    m, s = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1)
    z = ds.gold["z"]; ztr, zte = z.loc[tr_ids].to_numpy(float), z.loc[te_ids].to_numpy(float)
    F = lambda X, zz: np.c_[(X - m) / s, (zz - ztr.mean()) / ztr.std()]
    Y = ds.human(tr_ids); Y = (Y - Y.mean(0)) / Y.std(0)
    P = anchored_fit(F(Xtr, ztr), Y)(F(Xte, zte)); H = ds.human(te_ids); K = len(ds.A)
    raw = 100 * np.nanmean([sp(H[:, k], Xte[:, k]) for k in range(K)])
    return raw, 100 * np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)])


rows = []
for n in ["hs2", "hanna", "ell", "merlin"]:
    ds = DS(n)
    have = lambda f: all(os.path.exists(x) for x in f.values())
    v0f = run_files(ds, "v0")
    if not have(v0f):
        continue
    v0 = load(ds, v0f)
    cases = {"v0 (1 call)": [v0]}
    for r in ("v0rerun1", "v0rerun2"):
        if have(run_files(ds, r)):
            cases[f"v0 + {r} (noise control)"] = [v0, load(ds, run_files(ds, r))]
    if all(have(run_files(ds, r)) for r in ("v0rerun1", "v0rerun2")):
        cases["v0 x3 (noise control, 3 calls)"] = [v0] + [load(ds, run_files(ds, r)) for r in ("v0rerun1", "v0rerun2")]
    for m in ("textgrad_ccc", "dspy_mipro", "dspy_gepa", "opro"):
        if have(run_files(ds, m)):
            tg = load(ds, run_files(ds, m))
            cases[f"{m} (1 call)"] = [tg]
            cases[f"v0 + {m} (2 calls)"] = [v0, tg]
    for d in sorted(glob.glob(f"{OUT}/{n}/archive_M*/seed0")):
        if os.path.exists(f"{d}/final_archive.json"):
            arch = json.load(open(f"{d}/final_archive.json"))["archive"]
            fs = [{s: f"{d}/scores_{s}_p{i}.csv" for s in ("fit", "sel", "test")} for i in arch]
            if all(have(f) for f in fs):
                cases[f"archive {os.path.basename(os.path.dirname(d))} ({len(arch)} calls)"] = [load(ds, f) for f in fs]
    for c, parts in cases.items():
        raw, dec = evaluate(ds, parts)
        rows.append(dict(ds=n, case=c, raw=raw if len(parts) == 1 else np.nan, shad=dec))
df = pd.DataFrame(rows)
with pd.option_context("display.width", 200):
    print(df.pivot_table(index="case", columns="ds", values="shad", sort=False).round(1).assign(
        mean=lambda t: t.mean(1).round(1)))
    print("raw (single prompt):")
    print(df.dropna().pivot_table(index="case", columns="ds", values="raw", sort=False).round(1))

"""Scale-free version of conflict_vs_noise.py: per task the outcome is the rank agreement (Spearman) of the judge with
the humans, which no movement along the score scale changes. For each consecutive pair of evaluations (aligned on item
content) the per-task change in Spearman gets a paired bootstrap CI over items (B resamples); systematic conflict = one
task significantly up and another significantly down. Step-0 evaluations of different runs (same prompt) give the noise
floor.

  python scripts/rank_conflicts.py <results dir> [--col=pred|pred_rank]
"""

import glob
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd

B, ALPHA = 1000, 0.05
COL = next((a.split("=")[1] for a in sys.argv if a.startswith("--col=")), "pred")


def load(R, tasks):
    out = {}
    for f in glob.glob(os.path.join(R, "eval_step_*.parquet")):
        t = int(re.search(r"eval_step_(\d+)", f).group(1))
        d = pd.read_parquet(f)
        if f"{COL}_{tasks[0]}" not in d:
            continue
        key = d.inputs.map(lambda x: "||".join(str(v)[:300] for _, v in sorted(x.items())))
        d["_key"] = key + "#" + d.assign(k=key).groupby("k").cumcount().astype(str)
        out[t] = d.set_index("_key")
    return out


def ranks(x):
    return pd.Series(x).rank().to_numpy()


def boot(a, b, tasks, rng):
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    res = {}
    for k in tasks:
        pa, pb, g = a[f"{COL}_{k}"].to_numpy(float), b[f"{COL}_{k}"].to_numpy(float), a[f"gt_{k}"].to_numpy(float)
        m = ~(np.isnan(pa) | np.isnan(pb))
        pa, pb, g = pa[m], pb[m], g[m]
        n = len(g)
        idx = rng.integers(0, n, (B, n))
        def rho(p, i):
            return np.array([np.corrcoef(ranks(p[j]), ranks(g[j]))[0, 1] for j in i])
        d = rho(pb, idx) - rho(pa, idx)
        obs = np.corrcoef(ranks(pb), ranks(g))[0, 1] - np.corrcoef(ranks(pa), ranks(g))[0, 1]
        lo, hi = np.quantile(d, [ALPHA / 2, 1 - ALPHA / 2])
        res[k] = (obs, lo > 0, hi < 0)
    up = [k for k in tasks if res[k][1]]
    down = [k for k in tasks if res[k][2]]
    return dict(sys=bool(up and down), up=",".join(up), down=",".join(down), **{f"d_{k}": round(res[k][0], 3) for k in tasks})


def main(root):
    rng = np.random.default_rng(0)
    rows, step0 = [], {}
    for R in sorted(glob.glob(os.path.join(root, "*", "run"))):
        name = os.path.basename(os.path.dirname(R))
        tasks = [t["task_name"] for t in json.load(open(os.path.join(R, "run_summary.json")))["config"]["tasks"]]
        mode = re.search(r"-(?:glm53|luna|qwen38)-(\w+?)-val=", name).group(1)
        ev = load(R, tasks)
        if 0 in ev:
            step0[name] = ev[0]
        for t in sorted(ev):
            if t and t - 1 in ev:
                rows.append(dict(mode=mode, step=t, kind="edge", **boot(ev[t - 1], ev[t], tasks, rng)))
    for a, b in itertools.combinations(sorted(step0), 2):
        rows.append(dict(mode="-", step=0, kind="rerun0", **boot(step0[a], step0[b], tasks, rng)))
    E = pd.DataFrame(rows)
    print(f"per-task Spearman ({COL}_*), paired bootstrap {B}x, systematic = one task sig. up and another sig. down")
    print(E.to_string(index=False))
    for (kind, mode), g in E.groupby(["kind", "mode"]):
        print(f"{kind:7s} {mode:5s} n={len(g):3d} systematic {g.sys.mean():.0%} ({int(g.sys.sum())})")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")][0])

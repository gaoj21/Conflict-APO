"""Summarize local runs: test Spearman per mode at step 0 / final step, per-task, worst-task.

Usage: python scripts/summarize.py results/dsv41 [--val mae] [--steps 12]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TASKS = ["fluency", "relevance", "coherence", "consistency"]
NAME_RE = re.compile(r"TextGrad-(?P<ds>\w+)-(?P<tag>[\w.]+)-(?P<mode>[\w-]+?)-val=(?P<val>\w+)(?:-task=(?P<task>\w+))?-run_(?P<seed>\d+)$")


def rho(df, t):
    if f"pred_{t}" not in df:
        return np.nan
    p = pd.to_numeric(df[f"pred_{t}"], errors="coerce")
    g = pd.to_numeric(df[f"gt_{t}"], errors="coerce")
    v = p.notna() & g.notna()
    return spearmanr(p[v], g[v])[0] if v.sum() > 2 else np.nan


def final_instructions(run_dir):
    logs = sorted(glob.glob(os.path.join(run_dir, "run_logs", "step_*.parquet")))
    if not logs:
        return None
    st = json.loads(pd.read_parquet(logs[-1])["algorithm_state"].iloc[0])
    return st.get("previous_instructions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--val", default=None)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        m = NAME_RE.match(os.path.basename(d))
        if not m or (args.val and m["val"] != args.val):
            continue
        run = os.path.join(d, "run")
        for step in (0, args.steps):
            f = os.path.join(run, f"eval_step_{step}.parquet")
            if not os.path.exists(f):
                continue
            df = pd.read_parquet(f)
            tasks = [m["task"]] if m["task"] else TASKS
            r = {"mode": "ST" if m["task"] else m["mode"], "val": m["val"], "seed": int(m["seed"]), "step": step}
            for t in tasks:
                r[t] = rho(df, t)
            if step == args.steps:
                instr = final_instructions(run) or {}
                for t in tasks:
                    r[f"len_{t}"] = len(instr.get(t, "").split())
                r["accepted"] = sum(
                    str(pd.read_parquet(p)["prompt_accepted"].iloc[0]).lower() == "true"
                    for p in glob.glob(os.path.join(run, "run_logs", "step_*.parquet"))
                )
            rows.append(r)
    if not rows:
        print("no runs found")
        return
    df = pd.DataFrame(rows)
    # ST: merge the 4 single-task runs of a seed into one row
    st = df[df["mode"] == "ST"]
    if len(st):
        st = st.groupby(["mode", "val", "seed", "step"], as_index=False).first()
        df = pd.concat([df[df["mode"] != "ST"], st])
    df["avg"] = df[TASKS].mean(axis=1)
    df["worst"] = df[TASKS].min(axis=1)

    for val, dv in df.groupby("val"):
        print(f"\n=== validation gate: {val} ===")
        out = []
        for (mode, step), g in dv.groupby(["mode", "step"]):
            row = {"mode": mode, "step": step, "n": len(g)}
            for c in ["avg", "worst"] + TASKS:
                row[c] = f"{g[c].mean():.3f}±{g[c].std(ddof=0):.3f}"
            if step == args.steps:
                lens = [g[f"len_{t}"].mean() for t in TASKS if f"len_{t}" in g]
                row["len(f/r/c/co)"] = "/".join(f"{x:.0f}" for x in lens)
                if "accepted" in g:
                    row["acc"] = f"{g['accepted'].mean():.1f}"
            out.append(row)
        print(pd.DataFrame(out).sort_values(["step", "mode"]).to_string(index=False))


if __name__ == "__main__":
    main()

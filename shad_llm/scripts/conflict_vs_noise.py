"""How many of the step-level conflicts in multi-objective prompt optimization are real, and how many are noise?

For every TextGrad run (SummEval, all modes, val gate on/off, 3 seeds) the judge prompt is evaluated on the same test
items after every step. Each consecutive pair of evaluations is either
  * an EDGE: the step changed the prompt (accepted update), or
  * a RERUN: the step was rejected by the validation gate, so the same prompt was evaluated twice (pure noise);
    the initial prompt evaluated at step 0 of every run gives further rerun pairs.
Per task (dimension) the per-item outcome is off-by-one correctness (|pred - gt| <= 1, the paper's metric). For each pair:
  raw conflict        some item fixed and some item broken (any task);
  task-level conflict one task's off-by-one accuracy up and another's down;
  systematic conflict one task up and another down, both significant (exact McNemar p < 0.05).
The rates on reruns are the noise floor of each notion.

With --calibrate, each task's predicted scores are first mapped to the human scale by a monotone (isotonic) fit of
gt on pred, cross-fitted on two halves of the items (fit on one half, applied to the other), so a shift of the
whole score scale (more lenient / stricter) no longer changes off-by-one correctness by itself.

  python scripts/conflict_vs_noise.py <results dir, e.g. .../results/dsv41> [--calibrate]
"""

import glob
import itertools
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

ALPHA = 0.05


def mcnemar(a, b):
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    if n == 0:
        return 1.0, 0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n), int(np.sign(n10 - n01))


CALIBRATE = "--calibrate" in sys.argv
STORED_CAL = "--stored-cal" in sys.argv  # use the heads' outputs saved by TextGradAF (pred_cal_<task>)
TOL = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--tol=")), 1))  # 0 = exact match (BRIGHTER480)


def _ok(pred, gt):
    pred = np.asarray(pred, float)
    return (np.abs((np.round(pred) if TOL < 1 else pred) - np.asarray(gt, float)) <= TOL).astype(int)


def _pav(levels, means, weights):
    """Pool-adjacent-violators: non-decreasing fit of means over sorted levels."""
    blocks = [[m, w, [l]] for l, m, w in zip(levels, means, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m0, w0, l0 = blocks[i]
            m1, w1, l1 = blocks[i + 1]
            blocks[i] = [(m0 * w0 + m1 * w1) / (w0 + w1), w0 + w1, l0 + l1]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return {l: m for m, _, ls in blocks for l in ls}


def calibrated(pred, gt):
    """Cross-fitted isotonic map of pred onto gt (halves by item position, after content-sorting)."""
    out = np.empty(len(pred))
    half = np.arange(len(pred)) % 2 == 0
    for fit, app in ((half, ~half), (~half, half)):
        df = pd.DataFrame({"p": pred[fit], "g": gt[fit]}).groupby("p").g.agg(["mean", "size"]).sort_index()
        mp = _pav(list(df.index), list(df["mean"]), list(df["size"]))
        lv = np.array(sorted(mp))
        out[app] = [mp[lv[np.abs(lv - x).argmin()]] for x in pred[app]]  # unseen level -> nearest fitted level
    return out


def load_run(R, tasks):
    evals = {}
    for f in glob.glob(os.path.join(R, "eval_step_*.parquet")):
        t = int(re.search(r"eval_step_(\d+)", f).group(1))
        d = pd.read_parquet(f)
        # sample ids are per step and the item order differs between steps: align on the item content
        d["_key"] = d.inputs.map(lambda x: "||".join(str(v)[:300] for _, v in sorted(x.items()))) + \
            d[[f"gt_{k}" for k in tasks]].astype(str).agg("|".join, axis=1)
        d["_key"] = d["_key"] + "#" + d.groupby("_key").cumcount().astype(str)
        d = d.set_index("_key").sort_index()
        if STORED_CAL:
            evals[t] = {k: _ok(d[f"pred_cal_{k}"], d[f"gt_{k}"]) for k in tasks}
        elif CALIBRATE:
            evals[t] = {k: (np.abs(calibrated(d[f"pred_{k}"].to_numpy(float), d[f"gt_{k}"].to_numpy(float))
                                   - d[f"gt_{k}"].to_numpy(float)) <= 1).astype(int) for k in tasks}
        else:
            evals[t] = {k: _ok(d[f"pred_{k}"], d[f"gt_{k}"]) for k in tasks}
        evals[t]["_ids"] = d.index.to_numpy()
    acc = {}
    for f in glob.glob(os.path.join(R, "run_logs", "step_*.parquet")):
        s = pd.read_parquet(f).iloc[0]
        acc[int(s.step)] = bool(s.prompt_accepted) and s.old_prompt_template != s.new_prompt_template
    return evals, acc


def _common(a, b, tasks):
    """Restrict two evaluations to the rows both contain (aligned on the content key)."""
    _, ia, ib = np.intersect1d(a["_ids"], b["_ids"], return_indices=True)
    return ({k: a[k][ia] for k in tasks} | {"_ids": a["_ids"][ia]}), ({k: b[k][ib] for k in tasks} | {"_ids": b["_ids"][ib]})


def compare(c, p, tasks):
    res = {k: mcnemar(c[k], p[k]) for k in tasks}
    d = {k: c[k].mean() - p[k].mean() for k in tasks}
    raw = any(((c[k] == 1) & (p[k] == 0)).any() for k in tasks) and any(((c[k] == 0) & (p[k] == 1)).any() for k in tasks)
    level = max(d.values()) > 0 > min(d.values())
    up = [k for k in tasks if res[k][0] < ALPHA and res[k][1] > 0]
    down = [k for k in tasks if res[k][0] < ALPHA and res[k][1] < 0]
    return dict(raw=raw, level=level, sys=bool(up and down), up=",".join(up), down=",".join(down),
                **{f"d_{k}": round(100 * d[k], 1) for k in tasks})


def main(root):
    print("calibrated per task (cross-fitted isotonic)" if CALIBRATE else "heads' outputs (pred_cal_*)" if STORED_CAL
          else "raw predicted scores")
    rows, step0 = [], {}
    for R in sorted(glob.glob(os.path.join(root, "*", "run"))):
        name = os.path.basename(os.path.dirname(R))
        cfg = json.load(open(os.path.join(R, "run_summary.json")))["config"]
        tasks = [t["task_name"] for t in cfg["tasks"]]
        if len(tasks) < 2:
            continue  # single-task runs have no cross-task conflict
        mode = re.search(r"-(?:glm53|luna|qwen38)-(\w+?)-val=|-(SSS|SSC|SCC|CCC)-", name)
        if mode is None:
            continue  # other algorithms (MRB, ...) in the same results folder
        mode = mode.group(1) or mode.group(2)
        val = re.search(r"val=(\w+)", name).group(1)
        evals, acc = load_run(R, tasks)
        if 0 in evals:
            step0[name] = evals[0]
        for t in sorted(evals):
            if t == 0 or t - 1 not in evals or t not in acc:
                continue
            c, p = _common(evals[t], evals[t - 1], tasks)  # intermediate evals may use a subset of the test rows
            rows.append(dict(run=name, mode=mode, val=val, step=t, kind="edge" if acc[t] else "rerun", n=len(c["_ids"]),
                             **compare(c, p, tasks)))
    # the initial prompt is the same in every run: pairs of step-0 evaluations are reruns too
    names = sorted(step0)
    for a, b in itertools.combinations(names, 2):
        c, p = _common(step0[b], step0[a], tasks)
        rows.append(dict(run=f"{a} | {b}", mode="-", val="-", step=0, kind="rerun0", n=len(c["_ids"]), **compare(c, p, tasks)))
    E = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "conflict_vs_noise_" + os.path.basename(os.path.normpath(root)) + ("_cal" if CALIBRATE else "_storedcal" if STORED_CAL else "") + ".csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    E.to_csv(out, index=False)

    def line(df, label):
        return (f"{label:28s} n={len(df):4d}  raw {df.raw.mean():5.0%}  task-level {df.level.mean():5.0%}  "
                f"systematic {df.sys.mean():5.1%} ({int(df.sys.sum())})")

    print(f"tasks: {tasks}; per-item off-by-one; {E.run.str.contains(r' \| ').sum()} step-0 rerun pairs")
    print(line(E[E.kind == "rerun0"], "reruns (step 0, across runs)"))
    print(line(E[E.kind == "rerun"], "reruns (rejected steps)"))
    print(line(E[E.kind == "edge"], "edges (accepted steps)"))
    for m in sorted(E.loc[E.kind == "edge", "mode"].unique()):
        print(line(E[(E.kind == "edge") & (E["mode"] == m)], f"  edges {m}"))
    S = E[(E.kind == "edge") & E.sys]
    if len(S):
        print("\nsystematic conflicts: gained task(s) -> lost task(s)")
        print(S.groupby(["up", "down"]).size().sort_values(ascending=False).to_string())
    print(f"\nsaved {os.path.normpath(out)}")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")][0])

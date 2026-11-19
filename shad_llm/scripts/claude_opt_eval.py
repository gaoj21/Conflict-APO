"""Test-set comparison of the arms of scripts/claude_opt.py: per attribute Spearman of the final prompt, raw and with
the anchored readout (fitted on fit + sel of that prompt), paired bootstrap vs v0; plus the step trajectory on sel
(accepted edits: per-attribute change, number of attributes significantly up / down by paired bootstrap).

  python scripts/claude_opt_eval.py results/claude_hs2 ours text
"""
import json, os, sys
import numpy as np, pandas as pd

sys.argv, ARGS = sys.argv[:1] + [sys.argv[1], "x"], sys.argv[1:]
exec(open(os.path.join(os.path.dirname(__file__), "claude_opt.py")).read().split("# ---------------------------------------------------------------- main loop")[0])
R, ARMS = ARGS[0], ARGS[1:]


def load(arm, name):
    base = f"{R}/out" if name == "v0" else f"{R}/opt/{arm}/out"
    tr = pd.concat([frame(pd.read_csv(f"{base}/{name}_{s}.csv"), s) for s in ("fit", "sel")], ignore_index=True)
    te = frame(pd.read_csv(f"{base}/{name}_test.csv"), "test")
    return tr, te


def preds(tr, te):
    return te[[x + "_j" for x in A]].to_numpy(float), readout_scores(tr, te)


rng = np.random.default_rng(0)
tr0, te0 = load(None, "v0")
P0raw, P0ro = preds(tr0, te0)
rows = {"v0": (te0, P0raw, P0ro)}
for arm in ARMS:
    f = f"{R}/opt/{arm}/final.txt"
    if not os.path.exists(f):
        continue
    name = open(f).read().strip()
    if name == "v0":
        rows[arm] = rows["v0"]; continue
    if not os.path.exists(f"{R}/opt/{arm}/out/{name}_test.csv"):
        print(f"{arm}: final {name} not judged on test yet"); continue
    tr, te = load(arm, name)
    rows[arm + f" ({name})"] = (te, *preds(tr, te))
print(f"=== {os.path.basename(R)} test Spearman: {' '.join(x[:6] for x in A)} | mean   [Δmean vs v0-raw, 95% CI]")
ids0 = te0.id.to_numpy()
for k, (te, Praw, Pro) in rows.items():
    for lab, P in (("raw", Praw), ("readout", Pro)):
        H = te[A].to_numpy(float)
        r = per_task(H, P)
        ids = np.intersect1d(ids0, te.id)
        i0, i1 = pd.Index(ids0).get_indexer(ids), pd.Index(te.id).get_indexer(ids)
        Hc = te0[A].to_numpy(float)[i0]
        d = []
        for _ in range(500):
            b = rng.integers(0, len(ids), len(ids))
            d.append(np.nanmean(per_task(Hc[b], P[i1][b]) - per_task(Hc[b], P0raw[i0][b])))
        print(f"  {k[:22]:22s} {lab:7s} " + " ".join(f"{x:6.3f}" for x in r) + f" | {np.nanmean(r):.3f}   "
              f"[{np.nanmean(per_task(Hc, P[i1])) - np.nanmean(per_task(Hc, P0raw[i0])):+.3f} ({np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f})]")
for arm in ARMS:
    f = f"{R}/opt/{arm}/log.jsonl"
    if not os.path.exists(f):
        continue
    print(f"--- {arm} trajectory on sel")
    for l in open(f):
        s = json.loads(l)
        for c in s["cands"]:
            mark = "*" if s["accepted"] and c["name"] == s["current"] else " "
            print(f"  {mark}{c['name']} score {c['score']:+.3f}  d_raw {np.round(c['d_raw'], 3)}  u_raw {np.round(c['u_raw'], 1)}  extra {c['n_extra']}")

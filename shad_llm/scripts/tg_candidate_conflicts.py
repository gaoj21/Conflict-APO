"""Per-candidate conflicts of TextGrad CCC on human vs content labels (same protocol as the paper's diagnostic).

Every candidate edit (accepted or not) and the prompt it would replace are judged on TextGrad's validation items
(cache hits, no new calls). Per attribute: change of Spearman with the HUMAN scores, raw and after a fixed ShAD
decoder (fitted on v0's fit + sel scores); a significant conflict is one attribute up and another down, each by more
than two paired-bootstrap SE. len-move: mean |Spearman| between the per-item score change and log length.
  .venv/bin/python scripts/tg_candidate_conflicts.py <ds> [<ds> ...]
"""
import json, os, sys

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
import gem  # noqa: E402
from evaluate import sp  # noqa: E402
from shad import Decoder  # noqa: E402

assert not os.environ.get("GEM_TEACHER"), "evaluate against the human scores"


def judge(ds, instr, rows):
    df = ds.judge_msgs(lambda r: [dict(role="system", content=instr),
                                  dict(role="user", content=r["user"].strip() + "\n\n" + ds.FORMAT)], rows, 16, False)
    return ds.worst_case(df).set_index("id").loc[[r["id"] for r in rows]].reset_index()


def deltas(H, P0, P1, B=300):
    K = H.shape[1]
    d = np.array([sp(H[:, k], P1[:, k]) - sp(H[:, k], P0[:, k]) for k in range(K)])
    rng = np.random.default_rng(0); bs = []
    for _ in range(B):
        i = rng.integers(0, len(H), len(H))
        bs.append([sp(H[i, k], P1[i, k]) - sp(H[i, k], P0[i, k]) for k in range(K)])
    se = np.nanstd(np.array(bs), 0)
    return d, se, bool((d > 2 * se).any() and (d < -2 * se).any())


rows = []
for n in sys.argv[1:]:
    ds = gem.DS(n)
    v0 = pd.concat([pd.read_csv(f"{gem.OUT}/{n}/v0/seed0/scores_{s}.csv") for s in ("fit", "sel")])
    D = Decoder(ds, v0)
    val = ds.subset("sel", 100, seed=0)
    ids = [r["id"] for r in val]
    H = ds.human(ids); z = ds.gold.loc[ids, "z"].to_numpy(float)
    for m, lab in (("textgrad_ccc_cands", "human"), ("textgrad_ccc_content_cands", "content")):
        hist = json.load(open(f"{gem.OUT}/{n}/{m}/seed0/hist_all.json"))
        for prev, h in zip(hist[:-1], hist[1:]):
            S0, S1 = judge(ds, prev["prompt"], val), judge(ds, h["cand"], val)
            X0, X1 = S0[ds.A].to_numpy(float), S1[ds.A].to_numpy(float)
            dr, ser, cr = deltas(H, X0, X1)
            dd, sed, cd = deltas(H, D(X0, ids), D(X1, ids))
            mv = np.nanmean([abs(sp(X1[:, k] - X0[:, k], z)) for k in range(len(ds.A))])
            rows.append(dict(ds=n, labels=lab, step=h["step"], kept=h["kept"], conflict_raw=cr, conflict_dec=cd,
                             gain_raw=dr[dr > 0].sum() * 100, loss_raw=dr[dr < 0].sum() * 100,
                             gain_dec=dd[dd > 0].sum() * 100, loss_dec=dd[dd < 0].sum() * 100, len_move=mv))
df = pd.DataFrame(rows)
df.to_csv(f"{gem.OUT}/tg_candidate_conflicts.csv", index=False)
with pd.option_context("display.width", 200):
    g = df.groupby(["ds", "labels"])
    print(g.agg(cands=("step", "size"), kept=("kept", "sum"), conflict_raw=("conflict_raw", "sum"),
                conflict_dec=("conflict_dec", "sum"), len_move=("len_move", "mean"),
                gain_raw=("gain_raw", "mean"), loss_raw=("loss_raw", "mean")).round(3))
    print(df.groupby("labels")[["conflict_raw", "conflict_dec", "len_move"]].agg(["sum", "mean"]).round(3))

"""Closed-loop text rules: ShAD's remaining correction is added to simple, executable rules, round after round.

The prompt keeps v0 and adds, per attribute, (i) a length adjustment in rating points per word-count band and
(ii) at most one "move toward the <attribute> rating you already gave" pull, with attributes rated in an order where
a source comes before the attributes that lean on it. Each round the judge rates fit + sel + test with the current
rules; ShAD is refitted on its new ratings, and what ShAD still corrects is added to the rules: its length weight,
converted to points per band, is added to the band adjustments; its weight on the pulled attribute (relative to the
attribute's own score) is added to the pull fraction. No LLM writes the rules, so there is no search noise; the rules
converge as the judge's own ratings absorb the correction and ShAD approaches its anchor. The round reported is the
one with the best raw Spearman on sel. Optional fixed random demos (--demos k).
--step s: add only s times ShAD's remaining correction per round (damping; the judge tends to overshoot a full step).
--nopull: length rules only, no cross-attribute pulls.
  .venv/bin/python baselines/run_textloop.py <ds> [--rounds 3] [--demos 0] [--step 1] [--nopull]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--demos", type=int, default=0)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--step", type=float, default=1.0); ap.add_argument("--nopull", action="store_true")
ap.add_argument("--pick", choices=["random", "shad"], default="random")
ap.add_argument("--base", default="v0")  # start from an APO method's final prompt: textgrad_ccc | opro | mipro
ap.add_argument("--step1", type=float, default=None)  # step of the first update (default: --step)  # demos: random fit items, or ShAD's largest corrections
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = (f"textloop_{a.base}_" if a.base != "v0" else "textloop_") + f"d{a.demos}" + ("_shadpick" if a.pick == "shad" else "") + (f"_s{a.step:g}" if a.step != 1 else "") + (f"_f{a.step1:g}" if a.step1 is not None else "") + ("_nopull" if a.nopull else "")
gem.TAG.v = f"{a.ds}/{METHOD}/seed0"
OUTD = ds.out(METHOD, 0)
A, K = ds.A, len(ds.A)
WC = lambda i: len(ds.text[i].split())
if a.base == "v0":
    BASE = ds.v0.strip()
elif a.base == "mipro":
    prog = json.load(open(f"{gem.OUT}/{a.ds}/dspy_mipro/seed0/program.json"))["predict"]
    _shots = "\n\n".join(f"### Example {i + 1}\n{d['item'].strip()}\n\nHuman ratings: "
                          + json.dumps({x: round(float(d[x]), 1) for x in ds.A}) for i, d in enumerate(prog.get("demos", []))
                          if all(x in d for x in ds.A))
    BASE = prog["signature"]["instructions"].strip() + ("\n\n## Examples rated by human experts\n\n" + _shots if _shots else "")
else:
    BASE = open(f"{gem.OUT}/{a.ds}/{a.base}/seed0/final_prompt.txt").read().strip()
LAB = ds.rows["fit"] + ds.rows["sel"]
wc_lab = np.array([WC(r["id"]) for r in LAB])
cuts = np.quantile(wc_lab, [.2, .4, .6, .8]).round().astype(int)
band = lambda w: int(np.searchsorted(cuts, w, side="left"))
BLAB = [f"up to {cuts[0]} words"] + [f"{cuts[i] + 1}-{cuts[i + 1]} words" for i in range(3)] + [f"more than {cuts[3]} words"]

shots = ""
if a.demos:
    pick = np.random.default_rng(0).choice(len(ds.rows["fit"]), a.demos, replace=False)
    if a.pick == "shad":  # fit items whose v0 ratings ShAD corrects most (cross-fitted on v0's fit + sel scores)
        v0 = pd.concat([pd.read_csv(f"{gem.OUT}/{a.ds}/v0/seed0/scores_{s}.csv") for s in ("fit", "sel")], ignore_index=True)
        Xv = v0[A].to_numpy(float); Xv = np.where(np.isnan(Xv), np.nanmean(Xv, 0), Xv)
        zv = ds.gold.loc[v0.id, "z"].to_numpy(float); Fv = np.c_[(Xv - Xv.mean(0)) / Xv.std(0), (zv - zv.mean()) / zv.std()]
        Yv = ds.human(v0.id); myv, syv = Yv.mean(0), Yv.std(0); fold = np.arange(len(v0)) % 5; Pv = np.zeros_like(Yv)
        for f in range(5):
            Pv[fold == f] = anchored_fit(Fv[fold != f], (Yv[fold != f] - myv) / syv)(Fv[fold == f]) * syv + myv
        cor = np.abs(Pv - Xv).mean(1); fid = [r["id"] for r in ds.rows["fit"]]; pos = {i: n for n, i in enumerate(fid)}
        cor[~v0.id.isin(fid).to_numpy()] = -1
        pick = [pos[v0.id[i]] for i in np.argsort(-cor)[:a.demos]]
    shots = "\n\n## Examples rated by human experts\n" + "\n\n".join(
        f"### Example {n + 1} (Word count: {WC(ds.rows['fit'][i]['id'])})\n{ds.rows['fit'][i]['user'].strip()[:2500]}\n\n"
        f"Expert ratings: " + json.dumps({x: float(v) for x, v in zip(A, ds.human([ds.rows['fit'][i]['id']])[0])})
        for n, i in enumerate(pick))


def fit_shad(df):
    X = df[A].to_numpy(float); X = np.where(np.isnan(X), np.nanmean(X, 0), X)
    m, s = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
    z = ds.gold.loc[df.id, "z"].to_numpy(float); zs = (z - z.mean()) / z.std()
    Y = ds.human(df.id); my, sy = Y.mean(0), Y.std(0)
    W = anchored_fit(np.c_[(X - m) / s, zs], (Y - my) / sy)(np.eye(K + 1))
    b = np.array([band(WC(i)) for i in df.id]); bz = np.array([zs[b == j].mean() for j in range(5)])
    return W, s, sy, bz


def rules_text(order, pull, adj):
    out = []
    for pos, k in enumerate(order):
        parts = []
        if pull[k] is not None and pull[k][1] > 0.02:
            j, f = pull[k]
            parts.append(f"if it differs from the {A[j]} rating you already gave, move it {round(100 * min(f, .9))}% of the way toward that rating")
        if np.any(np.abs(adj[k]) >= .05):
            parts.append("then adjust for length: " + ", ".join(f"{l}: {'+' if v > 0 else ''}{v:.1f}" for l, v in zip(BLAB, adj[k])))
        out.append(f"{pos + 1}. {A[k]}: rate its own evidence" + ("; " + "; ".join(parts) if parts else "") + ".")
    return ("## How to rate (learned from expert ratings)\nRate the attributes in the order below and write each rating before "
            "the next; later attributes refer to ratings you already gave. The word count of the item is given at the top.\n"
            + "\n".join(out))


def judge(guide, split, t):
    f = f"{OUTD}/r{t}_{split}.csv"
    if not os.path.exists(f):
        sysm = (BASE + ("\n\n" + guide if guide else "") + shots + "\n\n" + ds.FORMAT +
                "\nHalf points and decimals are allowed. The word count of the item is given at the top.")
        ds.judge_msgs(lambda r: [dict(role="system", content=sysm),
                                 dict(role="user", content=f"(Word count: {WC(r['id'])})\n\n" + r["user"])], ds.rows[split],
                      a.workers, strict=False).to_csv(f, index=False)
    return ds.worst_case(pd.read_csv(f))


raw = lambda df: 100 * float(np.nanmean(ds.spearman(df)))
order, pull, adj, guide, log = None, [None] * K, np.zeros((K, 5)), "", []
for t in range(a.rounds + 1):
    S = {s: judge(guide, s, t) for s in ("fit", "sel", "test")}
    tr = pd.concat([S["fit"], S["sel"]], ignore_index=True)
    W, s_x, sy, bz = fit_shad(tr)
    X = tr[A].to_numpy(float); m, sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
    z = ds.gold["z"]; ztr, zte = z.loc[tr.id].to_numpy(float), z.loc[S["test"].id].to_numpy(float)
    F = lambda D, zz: np.c_[(D[A].to_numpy(float) - m) / sd, (zz - ztr.mean()) / ztr.std()]
    Yt = ds.human(tr.id)
    P = anchored_fit(F(tr, ztr), (Yt - Yt.mean(0)) / Yt.std(0))(F(S["test"], zte)); H = ds.human(S["test"].id)
    rec = dict(round=t, sel_raw=raw(S["sel"]), test_raw=raw(S["test"]),
               test_shad=100 * float(np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)])), guide=guide)
    log.append(rec); json.dump(log, open(f"{OUTD}/log.json", "w"), indent=1)
    print(f"[{a.ds}] round {t}: sel raw {rec['sel_raw']:.1f} | test raw (no decoder) {rec['test_raw']:.1f} | "
          f"test +ShAD {rec['test_shad']:.1f}", flush=True)
    if t == a.rounds:
        break
    if order is None:  # fixed after round 0: sources before the attributes that lean on them
        dep = np.abs(W[:K, :K]).copy(); np.fill_diagonal(dep, 0)
        order, left = [], list(range(K))
        while left:
            k = min(left, key=lambda k: sum(dep[j, k] for j in left if j != k))
            order.append(k); left.remove(k)
    for pos, k in enumerate(order):
        st = a.step1 if (a.step1 is not None and t == 0) else a.step
        adj[k] += st * W[K, k] * sy[k] * (bz - bz.mean())  # remaining length correction, in points per band
        earlier = [] if a.nopull else [j for j in order[:pos] if W[j, k] > 0]
        if earlier:
            j = pull[k][0] if pull[k] is not None else max(earlier, key=lambda j: W[j, k])
            f = W[j, k] / max(W[j, k] + abs(W[k, k]), 1e-6)  # share of the pulled rating in the corrected one
            pull[k] = (j, (pull[k][1] if pull[k] is not None else 0) + (a.step1 if (a.step1 is not None and t == 0) else a.step) * f)
    guide = rules_text(order, pull, adj)
    open(f"{OUTD}/guide_r{t + 1}.txt", "w").write(guide)
best = max(log, key=lambda r: r["sel_raw"])
print(f"[{a.ds}] {METHOD} final round {log[-1]['round']}: test raw {log[-1]['test_raw']:.1f}", flush=True)
print(f"[{a.ds}] {METHOD} done; best on sel: round {best['round']} | test raw {best['test_raw']:.1f} | +ShAD {best['test_shad']:.1f}", flush=True)

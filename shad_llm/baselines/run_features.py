"""Feature APO: prompt optimization of the judge's checklist, the extra sub-scores ShAD reads.

The judge prompt is v0 plus a checklist of up to J items (each a question with an integer answer: a 0-4 rating or a
count). In the same single call, the judge rates the attributes and answers every checklist item; ShAD decodes each
attribute from the attribute scores, all checklist answers and log length (anchored to the attribute's own score).
The optimizer edits the checklist, not the attribute rubric: it sees what the decoder still gets wrong and how each
current item relates to the human scores, and returns a revised checklist. A revision is kept if the decoder fitted
on the pool's fit part scores higher (mean Spearman over attributes) on the pool's sel part.
Inference cost: one judge call per item, as for any single prompt.
  .venv/bin/python baselines/run_features.py <ds> [--steps 12] [--J 12] [--seed 0]
"""
import argparse, json, os, re, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=12); ap.add_argument("--J", type=int, default=12)
ap.add_argument("--nf", type=int, default=200); ap.add_argument("--ns", type=int, default=300)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = f"features_J{a.J}"
gem.TAG.v = f"{a.ds}/{METHOD}/seed{a.seed}"
OUTD = ds.out(METHOD, a.seed)
A, K = ds.A, len(ds.A)
POOL = ds.subset("fit", a.nf, seed=a.seed) + ds.subset("sel", a.ns, seed=a.seed)
PIDS = [r["id"] for r in POOL]
IS_FIT = np.array([i < a.nf for i in range(len(POOL))])
H = ds.human(PIDS)


def system(items):
    if not items:
        return ds.v0.strip() + "\n\n" + ds.FORMAT  # = the v0 judge (cache hit)
    cl = "\n".join(f"- {it['name']}: {it['question']} ({it['answer']})" for it in items)
    return (ds.v0.strip() + "\n\n## Checklist\nAlso answer each checklist item about the item being evaluated:\n" + cl
            + "\n\n" + ds.FORMAT + "\nIn the same JSON line, after the attributes, also include one integer for each "
            "checklist item, with the item names as keys.")


def judge(items, rows, strict):
    keys = A + [it["name"] for it in items]
    sysm = system(items)
    df = ds.judge_msgs(lambda r: [dict(role="system", content=sysm), dict(role="user", content=r["user"])], rows,
                       a.workers, strict)
    for kx in keys:  # judge_msgs keeps only the attributes: re-parse the answers to the checklist
        if kx not in df:
            df[kx] = np.nan
    if items:
        sc = df["reason"].fillna("").map(gem.parse_scores)
        for it in items:
            df[it["name"]] = sc.map(lambda d: d.get(it["name"], np.nan))
    return df, keys


def pool_features(items):
    df, keys = judge(items, POOL, strict=False)
    df = df.set_index("id").loc[PIDS]
    X = df[keys].to_numpy(float)
    col = np.nanmean(X, 0) if len(X) else 0
    col = np.where(np.isfinite(col), col, 0)
    return np.where(np.isnan(X), col, X)


Z = ds.gold.loc[PIDS, "z"].to_numpy(float)


def fit_predict(X, tr, te):
    X = np.where(np.isnan(X), 0, X)
    m, s = X[tr].mean(0), np.where(X[tr].std(0) > 0, X[tr].std(0), 1)
    F = np.c_[(X - m) / s, (Z - Z[tr].mean()) / Z[tr].std()]
    Y = H[tr]; mu, sd = Y.mean(0), Y.std(0)
    return anchored_fit(F[tr], (Y - mu) / sd)(F[te]) * sd + mu


def holdout(X):
    P = fit_predict(X, IS_FIT, ~IS_FIT)
    return float(np.mean([sp(H[~IS_FIT, k], P[:, k]) for k in range(K)]))


def feedback(X, items):
    fit_idx = np.where(IS_FIT)[0]
    fold = np.arange(len(fit_idx)) % 5
    P = np.zeros((len(fit_idx), K))
    for f in range(5):  # cross-fitted on the fit part only
        tr = np.zeros(len(POOL), bool); te = np.zeros(len(POOL), bool)
        tr[fit_idx[fold != f]] = True; te[fit_idx[fold == f]] = True
        P[fold == f] = fit_predict(X, tr, te)
    r = np.array([sp(H[fit_idx, k], P[:, k]) for k in range(K)])
    out = ["Agreement of the final ratings with human experts on the training items (Spearman): "
           + ", ".join(f"{x} {v:.2f}" for x, v in zip(A, r)) + "."]
    if items:
        out.append("Current checklist, with each item's Spearman correlation with every human attribute score:")
        for j, it in enumerate(items):
            c = [sp(X[fit_idx, K + j], H[fit_idx, k]) for k in range(K)]
            out.append(f"- {it['name']}: {it['question']} ({it['answer']}) | "
                       + ", ".join(f"{x} {v:+.2f}" for x, v in zip(A, c)))
    else:
        out.append("The checklist is currently empty.")
    out.append("Items the final ratings get most wrong:")
    err = np.abs(H[fit_idx] - P).mean(1)
    for i in np.argsort(-err)[:4]:
        j = fit_idx[i]
        item = POOL[j]["user"].strip()
        item = item[:1500] + (" [...]" if len(item) > 1500 else "")
        hum = json.dumps({x: int(round(v)) for x, v in zip(A, H[j])})
        fin = json.dumps({x: round(float(v), 1) for x, v in zip(A, P[i])})
        out.append(f"---\n{item}\nHuman ratings: {hum}; final ratings: {fin}")
    return "\n\n".join(out)


TASK = """An LLM judge rates items on these attributes: {attrs}. In the same call it also answers a checklist of short questions about the item. The final rating of each attribute is NOT the judge's rating: a learned calibration combines the judge's attribute ratings, all checklist answers and the length of the item, separately for each attribute. It already handles leniency and how much each attribute rewards length. The checklist is how the judge can pass on information the calibration cannot get otherwise.

{fb}

Revise the checklist so that it captures what the final ratings still miss. Good items are concrete and observable (e.g. counts of specific error types, presence of specific features), each answered by an integer. You may keep, replace or add items; drop items that carry no information; use at most {J} items. Return only a JSON list, each element {{"name": "<snake_case>", "question": "<one sentence>", "answer": "<integer 0-4 rating, or a count>"}}, between <checklist> and </checklist>."""


def propose(fb):
    msgs = [dict(role="user", content=TASK.format(attrs=", ".join(A), fb=fb, J=a.J))]
    for _ in range(3):
        txt = gem.chat(msgs, model=gem.OPTIM, temperature=0.7, max_tokens=3000, cache=False)[0]
        m = re.search(r"<checklist>(.*?)</checklist>", txt, re.S)
        if not m:
            continue
        try:
            items = json.loads(m.group(1))
            items = [dict(name=re.sub(r"\W", "_", str(it["name"]).strip().lower())[:40], question=str(it["question"]),
                          answer=str(it.get("answer", "integer"))) for it in items][:a.J]
            names = [it["name"] for it in items]
            if items and len(set(names)) == len(names) and not set(names) & set(A):
                return items
        except Exception:  # noqa: BLE001
            continue
    return None


state_f = f"{OUTD}/state.json"
if os.path.exists(state_f):
    st = json.load(open(state_f))
else:
    st = dict(items=[], X=pool_features([]).tolist(), hist=[], step=0)
cur = holdout(np.array(st["X"]))
if not st["hist"]:
    st["hist"].append(dict(step=0, items=[], holdout=cur))
    print(f"[{a.ds}] step 0 no checklist holdout {cur:.4f}", flush=True)
for step in range(st["step"] + 1, a.steps + 1):
    X = np.array(st["X"])
    new = propose(feedback(X, st["items"]))
    rec = dict(step=step)
    if new is not None:
        Xn = pool_features(new)
        v = holdout(Xn)
        rec.update(cand=new, cand_holdout=v, kept=v > cur)
        if v > cur:
            st["items"], st["X"], cur = new, Xn.tolist(), v
    rec.update(holdout=cur, n_items=len(st["items"]))
    st["hist"].append(rec); st["step"] = step
    json.dump(st, open(state_f, "w"))
    print(f"[{a.ds}] step {step} cand {rec.get('cand_holdout', float('nan')):.4f} "
          f"{'keep' if rec.get('kept') else 'drop'} items {len(st['items'])} holdout {cur:.4f}", flush=True)

json.dump(st["items"], open(f"{OUTD}/final_checklist.json", "w"), indent=1)
for s in ("fit", "sel", "test"):
    df, keys = judge(st["items"], ds.rows[s], strict=True)
    df.to_csv(f"{OUTD}/scores_{s}.csv", index=False)
print(a.ds, METHOD, "done;", len(st["items"]), "items; calls", gem.spent(gem.TAG.v)[1], flush=True)

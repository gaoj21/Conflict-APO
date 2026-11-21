"""Conflict measurement for APO edits on BBH: many candidate edits of one parent instruction, plus reruns for noise.

Each candidate is written by the optimizer (same model, temperature 0.7) from the errors of the parent on its own
random minibatch of 40 training questions (the usual APO proposal step); there is no acceptance, every candidate is
evaluated on the eval set (30 per task). Reruns of the parent (fresh calls) measure how many task changes noise alone
produces. Per candidate and task, a change is up / down when it exceeds two paired-bootstrap SE.
  .venv/bin/python qa/apo_sweep.py [--n 30] [--reruns 3] [--workers 32]
Writes results/gemma_qa/bbh/sweep/{cand_k,rerun_k}.csv, cands.json, summary.json.
"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines"); sys.path.insert(0, f"{ROOT}/qa")
import gem  # noqa: E402
from bbh_common import pools, answer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30); ap.add_argument("--reruns", type=int, default=3)
ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/bbh/sweep"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = "qa/bbh/sweep"
PARENT = "Think step by step."
OPT = open(f"{ROOT}/qa/apo_conflicts.py").read().split('OPT = """')[1].split('"""')[0]  # same optimizer prompt


def evaluate(inst, rows, tag):
    f = f"{OUT}/{tag}.csv"
    if os.path.exists(f):
        return pd.read_csv(f)
    with ThreadPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(ex.map(lambda r: answer(inst, r), rows)).drop(columns="txt")
    df.to_csv(f, index=False)
    return df


def changes(child, parent, rng):
    m = child.merge(parent, on=["qid", "category"], suffixes=("", "_p")); out = {}
    for c, g in m.groupby("category"):
        d = (g.correct - g.correct_p).to_numpy()
        se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        out[c] = (100 * d.mean(), bool(se > 0 and abs(d.mean()) > 2 * se))
    return out


ev, tr = pools()
par = evaluate(PARENT, ev, "parent")
print(f"parent acc {100 * par.correct.mean():.1f}", flush=True)
rng = np.random.default_rng(0); summ = dict(reruns=[], cands=[])
for k in range(a.reruns):
    gem.SALT.v = f"sweep_rerun{k}"
    ch = changes(evaluate(PARENT, ev, f"rerun_{k}"), par, rng)
    gem.SALT.v = ""
    up = [c for c, (d, s) in ch.items() if s and d > 0]; dn = [c for c, (d, s) in ch.items() if s and d < 0]
    summ["reruns"].append(dict(up=up, down=dn, delta={c: d for c, (d, s) in ch.items()}))
    print(f"rerun {k}: up {up or '-'}, down {dn or '-'}", flush=True)

cf = f"{OUT}/cands.json"
cands = json.load(open(cf)) if os.path.exists(cf) else []
mrng = np.random.default_rng(1)
batches = [mrng.choice(len(tr), 40, replace=False) for _ in range(a.n)]
for k in range(len(cands), a.n):  # proposals first (resumable); each from its own minibatch of parent errors
    mb = [tr[i] for i in batches[k]]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda r: answer(PARENT, r), mb))
    wrong = [(r, x) for r, x in zip(mb, res) if not x["correct"]][:8]
    errs = "\n\n".join(f"[{r['category']}]\n{r['q'][:1200]}\n...model: {x['txt'][-400:]}\nCorrect answer: {r['gold']}" for r, x in wrong)
    gem.SALT.v = f"sweep_prop{k}"
    txt, *_ = gem.chat([dict(role="user", content=OPT.format(inst=PARENT, acc=100 * np.mean([x["correct"] for x in res]),
                                                             errs=errs or "(none)"))], temperature=0.7, max_tokens=1500)
    gem.SALT.v = ""
    m = re.findall(r"<instruction>(.*?)</instruction>", txt or "", re.S)
    cands.append(m[-1].strip() if m else PARENT)
    json.dump(cands, open(cf, "w"), indent=1)
print(f"{len(cands)} candidates proposed", flush=True)

for k, inst in enumerate(cands):
    df = evaluate(inst, ev, f"cand_{k}")
    ch = changes(df, par, rng)
    up = [c for c, (d, s) in ch.items() if s and d > 0]; dn = [c for c, (d, s) in ch.items() if s and d < 0]
    summ["cands"].append(dict(k=k, acc=100 * df.correct.mean(), tok=df.out_tok.mean(), up=up, down=dn,
                              delta={c: d for c, (d, s) in ch.items()}))
    print(f"cand {k}: acc {100 * df.correct.mean():.1f}, up {up or '-'}, down {dn or '-'}", flush=True)
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)

C, R = summ["cands"], summ["reruns"]
nsig = lambda x: len(x["up"]) + len(x["down"])
print(f"\nsignificant task changes per evaluation: reruns {np.mean([nsig(x) for x in R]):.2f}, candidates {np.mean([nsig(x) for x in C]):.2f}")
print(f"candidates with a conflict (>=1 up and >=1 down): {sum(bool(x['up']) and bool(x['down']) for x in C)}/{len(C)}; "
      f"reruns: {sum(bool(x['up']) and bool(x['down']) for x in R)}/{len(R)}")
cnt = pd.Series([c for x in C for c in x["up"] + x["down"]]).value_counts()
print("tasks most often changed:", cnt.head(8).to_dict())
print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

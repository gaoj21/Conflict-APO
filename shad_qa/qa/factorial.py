"""Claim 1 of the factorial design: do instruction clauses have effects that differ by question type?

  .venv/bin/python qa/factorial.py [--clauses 8] [--prompts 24] [--per 100] [--workers 32]
1. base instruction on a train pool (30 per source); the optimizer (same model) reads errors from all sources and
   proposes `clauses` short, general, diverse instructions (the usual APO proposal step)
2. `prompts` random prompts: every clause is included in exactly half of them (balanced random design); each prompt is
   the base instruction plus its clauses; every prompt answers the same test pool (per source)
3. per source and clause: effect = mean accuracy with the clause minus without (paired over questions), SE by a
   two-way bootstrap over prompts and questions; a clause conflicts when it is significantly positive for one source and
   significantly negative for another
Writes results/gemma_qa/mixed/factorial/{clauses.json, design.csv, runs.csv, effects.csv}.
"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines"); sys.path.insert(0, f"{ROOT}/qa")
import gem  # noqa: E402
from mixed import items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clauses", type=int, default=8); ap.add_argument("--prompts", type=int, default=24)
ap.add_argument("--per", type=int, default=100); ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/mixed/factorial"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = "qa/mixed/factorial"
BASE = "Think step by step."


def call(content, mt=2000, temperature=0.0):
    try:
        return gem.chat([dict(role="user", content=content)], max_tokens=mt, temperature=temperature)[0]
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001
        return None


def answer(inst, r):
    txt = call(r["prompt"](inst))
    c, pred = r["score"](txt)
    return dict(qid=r["qid"], source=r["source"], correct=int(c), pred=pred)


PROPOSE = """A model answers questions of several kinds (financial tables, multi-hop reading, math word problems, commonsense multiple choice) with the instruction "{base}". Below are questions it got wrong, with its answer and the correct one.

{errs}

Propose {k} different short instructions (one sentence each) that could be added to the instruction to fix such errors. Make them general (not about one question) and diverse (each about a different aspect of how to reason or how to answer). Reply with only a JSON list of strings."""

if __name__ == "__main__":
    cf = f"{OUT}/clauses.json"
    if not os.path.exists(cf):
        tr = items("train", 30)
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda r: (r, call(r["prompt"](BASE))), tr))
        errs = []
        for r, txt in res:
            c, pred = r["score"](txt)
            if not c:
                errs.append(f"[{r['source']}] {r['question'][:400]}\nModel's answer: {pred}\nModel's reasoning ends: ...{(txt or '')[-300:]}")
        rng = np.random.default_rng(0)
        pick = [errs[i] for i in rng.permutation(len(errs))[:24]]
        txt = call(PROPOSE.format(base=BASE, errs="\n\n".join(pick), k=a.clauses), 1500)
        json.dump(json.loads(re.findall(r"\[.*\]", txt, re.S)[-1])[:a.clauses], open(cf, "w"), indent=1)
    C = json.load(open(cf))
    print("clauses:\n" + "\n".join(f"  c{i}: {c}" for i, c in enumerate(C)), flush=True)

    rng = np.random.default_rng(0)
    Z = np.array([rng.permutation([1] * (a.prompts // 2) + [0] * (a.prompts - a.prompts // 2)) for _ in C]).T  # prompts x clauses
    pd.DataFrame(Z, columns=[f"c{i}" for i in range(len(C))]).to_csv(f"{OUT}/design.csv", index_label="prompt")
    te = items("test", a.per)
    rf = f"{OUT}/runs.csv"
    done = pd.read_csv(rf) if os.path.exists(rf) else pd.DataFrame()
    for p in range(a.prompts):
        if len(done) and (done.prompt == p).any():
            continue
        inst = BASE + "".join(" " + C[i] for i in range(len(C)) if Z[p, i])
        with ThreadPoolExecutor(a.workers) as ex:
            df = pd.DataFrame(ex.map(lambda r: answer(inst, r), te)); df.insert(0, "prompt", p)
        done = pd.concat([done, df], ignore_index=True); done.to_csv(rf, index=False)
        print(f"prompt {p}: " + ", ".join(f"{s} {100 * g.correct.mean():.0f}" for s, g in df.groupby("source"))
              + f"  (spent ${gem.spent()[0]:.2f})", flush=True)

    R = done; srcs = sorted(R.source.unique())
    M = {s: R[R.source == s].pivot(index="prompt", columns="qid", values="correct").sort_index().to_numpy(float) for s in srcs}

    def eff(Mx, Zx):
        return np.array([Mx[Zx[:, c] == 1].mean() - Mx[Zx[:, c] == 0].mean() for c in range(Zx.shape[1])])
    rows, B = [], 1000
    for s in srcs:
        e = eff(M[s], Z); bs = []
        for _ in range(B):
            ip = rng.integers(0, a.prompts, a.prompts); iq = rng.integers(0, M[s].shape[1], M[s].shape[1])
            Zb = Z[ip]
            if (Zb.sum(0) == 0).any() or (Zb.sum(0) == len(ip)).any():
                continue
            bs.append(eff(M[s][ip][:, iq], Zb))
        se = np.std(bs, axis=0)
        for c in range(len(C)):
            rows.append(dict(source=s, clause=f"c{c}", effect=100 * e[c], se=100 * se[c], sig=abs(e[c]) > 2 * se[c]))
    E = pd.DataFrame(rows); E.to_csv(f"{OUT}/effects.csv", index=False)
    print("\neffect of each clause on accuracy x100 (with minus without; * = beyond 2 SE)")
    T = E.assign(cell=E.apply(lambda r: f"{r.effect:+5.1f}{'*' if r.sig else ' '}", axis=1)).pivot(index="clause", columns="source", values="cell")
    print(T.to_string())
    nconf = sum(((g.effect > 0) & g.sig).any() and ((g.effect < 0) & g.sig).any() for _, g in E.groupby("clause"))
    print(f"\nclauses significantly helping one source and hurting another: {nconf}/{len(C)}")
    print("accuracy x100 by source, mean over prompts: " + ", ".join(f"{s} {100 * M[s].mean():.1f}" for s in srcs))
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

"""QA pilot (MMLU-Pro): named question features + every question answered with each of a few behaviors.

The data feed a small network (named features -> shared concepts -> one success head per behavior) whose weights are
later translated into prompt rules. This script only collects:
  features.csv  one row per question: the solver's own 1-5 ratings of named properties + measurable ones
  runs.csv      one row per (question, behavior): correct, input / output tokens
  .venv/bin/python qa/run_qa.py <train|test> [--per 100] [--workers 32]
"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines")
import gem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("split"); ap.add_argument("--per", type=int, default=100); ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/mmlupro/{a.split}"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = f"qa/mmlupro/{a.split}"

FEATS = ["calculation", "steps", "knowledge", "ambiguity", "confidence"]
ACTIONS = {
    "direct": ("Answer immediately without any reasoning. Reply with only: Answer: <letter>", 64),
    "brief": ("Reason in at most three short sentences, then end with: Answer: <letter>", 600),
    "cot": ("Think step by step, then end with: Answer: <letter>", 3000),
    "verify": ("Think step by step, then check your answer once more by verifying it a second way, "
               "then end with: Answer: <letter>", 3000),
}
FEAT_PROMPT = ("Do not solve the question. Rate it on each property from 1 (lowest) to 5 (highest) and reply with only "
               "one line of JSON:\n"
               '{"calculation": how much numerical calculation it needs, "steps": how many reasoning steps it needs, '
               '"knowledge": how much specialised factual knowledge it needs, "ambiguity": how ambiguous the question '
               'or the options are, "confidence": how likely you are to answer it correctly}')


def items():
    from datasets import load_dataset
    d = load_dataset("TIGER-Lab/MMLU-Pro", split="test").to_pandas()
    rng = np.random.default_rng(0); rows = []
    for c, g in d.groupby("category"):
        idx = rng.permutation(len(g))[:2 * a.per]
        rows.append(g.iloc[idx[:a.per] if a.split == "train" else idx[a.per:]])
    return pd.concat(rows, ignore_index=True)


def qtext(r):
    return r.question.strip() + "\n\n" + "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r.options))


def ask(msgs, mt):
    try:
        return gem.chat(msgs, max_tokens=mt)
    except gem.Abort:
        raise
    except Exception as e:  # noqa: BLE001  one failed call: recorded as missing, never stops the run
        return None, 0, 0, str(e)[:200]


def feature(r):
    txt, it, ot, _ = ask([dict(role="user", content=qtext(r) + "\n\n" + FEAT_PROMPT)], 200)
    m = re.findall(r"\{[^{}]*\}", txt or "")
    try:
        f = json.loads(m[-1]); f = {k: float(f[k]) for k in FEATS}
    except Exception:  # noqa: BLE001
        f = {k: np.nan for k in FEATS}
    return dict(qid=r.question_id, category=r.category, q_words=len(r.question.split()), n_opt=len(r.options), **f)


def run(r, act):
    ins, mt = ACTIONS[act]
    txt, it, ot, _ = ask([dict(role="user", content=qtext(r) + "\n\n" + ins)], mt)
    m = re.findall(r"[Aa]nswer\s*[:：]?\s*\**\(?([A-J])\b", txt or "")
    pred = m[-1] if m else None
    return dict(qid=r.question_id, category=r.category, action=act, pred=pred, correct=int(pred == r.answer),
                failed=int(txt is None), in_tok=it, out_tok=ot)


df = items()
print(f"[{a.split}] {len(df)} questions, {df.category.nunique()} categories", flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    F = pd.DataFrame(ex.map(feature, [r for r in df.itertuples()]))
    F.to_csv(f"{OUT}/features.csv", index=False)
    print(f"features done, {F[FEATS].isna().any(axis=1).sum()} unparsable, spent ${gem.spent()[0]:.2f}", flush=True)
    jobs = [(r, act) for act in ACTIONS for r in df.itertuples()]
    R = pd.DataFrame(ex.map(lambda x: run(*x), jobs))
R.to_csv(f"{OUT}/runs.csv", index=False)
print(R.groupby("action")[["correct", "failed", "out_tok"]].mean().round(3).to_string())
print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

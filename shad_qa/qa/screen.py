"""Cheap conflict screen for a candidate QA setting: a few questions per task, every behavior, no features.
  .venv/bin/python qa/screen.py bbh [--per 30] [--workers 32]
Writes results/gemma_qa/<name>/screen/runs.csv (same columns as run_qa.py); then: qa/conflicts.py <name>/screen
"""
import argparse, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines")
import gem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name", choices=["bbh", "bbeh"]); ap.add_argument("--per", type=int, default=30); ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--long", type=int, default=3000, help="max output tokens of cot / verify")
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/{a.name}/screen"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = f"qa/{a.name}/screen"
TAIL = "Your final line must be: Answer: <answer> (for options, give the letter in parentheses, e.g. (A))."
ACTIONS = {
    "direct": ("Answer immediately without any reasoning. Reply with only one line. " + TAIL, 200),
    "brief": ("Reason in at most three short sentences. " + TAIL, 600),
    "cot": ("Think step by step. " + TAIL, a.long),
    "verify": ("Think step by step, then check your answer once more by verifying it a second way. " + TAIL, a.long),
}


def items():
    from datasets import get_dataset_config_names, load_dataset
    rows, rng = [], np.random.default_rng(0)
    if a.name == "bbeh":  # BIG-Bench Extra Hard: 23 tasks in one split
        d = load_dataset("BBEH/bbeh", split="train").to_pandas()
        for t, g in d.groupby("task"):
            g = g.iloc[rng.permutation(len(g))[:a.per]]
            rows += [dict(qid=f"{t}/{i}", category=t, q=r.input, gold=r.target) for i, r in g.iterrows()]
        return rows
    for t in get_dataset_config_names("lukaemon/bbh"):
        d = load_dataset("lukaemon/bbh", t, split="test").to_pandas()
        d = d.iloc[rng.permutation(len(d))[:a.per]]
        rows += [dict(qid=f"{t}/{i}", category=t, q=r.input, gold=r.target) for i, r in d.iterrows()]
    return rows


norm = lambda s: " ".join(re.sub(r"^[\s\"'(\[]+|[\s\"'.)\]]+$", "", str(s)).replace(",", " ").lower().split())  # word lists: commas = spaces


def match(pred, gold):
    """Exact after normalization, or the answer's leading label: "valid (A)" -> valid, "Yes (Jerry ...)" -> yes,
    "(B) text" / "B" -> (B) when the gold is an option letter."""
    if pred is None:
        return False
    p, g = norm(pred), norm(gold)
    if p == g:
        return True
    if re.fullmatch(r"\([A-Z]\)", str(gold).strip()):
        m = re.match(r"\(?([a-z])\)?(\s|$)", p)
        return bool(m) and m.group(1) == g
    return len(g.split()) == 1 and bool(p) and p.split()[0].strip("()") == g


def run(r, act):
    ins, mt = ACTIONS[act]
    try:
        txt, it, ot, _ = gem.chat([dict(role="user", content=r["q"].strip() + "\n\n" + ins)], max_tokens=mt)
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001
        txt, it, ot = None, 0, 0
    m = re.findall(r"[Aa]nswer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
    pred = m[-1].strip() if m else None
    return dict(qid=r["qid"], category=r["category"], action=act, pred=pred, correct=int(match(pred, r["gold"])),
                failed=int(txt is None), in_tok=it, out_tok=ot)


rows = items()
print(f"[{a.name}] {len(rows)} questions, {len({r['category'] for r in rows})} tasks", flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    R = pd.DataFrame(ex.map(lambda x: run(*x), [(r, act) for act in ACTIONS for r in rows]))
R.to_csv(f"{OUT}/runs.csv", index=False)
print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

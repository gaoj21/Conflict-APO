"""Do APO edits conflict across tasks on QA?  OPRO/TextGrad-style hill climbing on BBH (27 tasks mixed).

Each step: the current instruction answers a minibatch of 40 training questions; the optimizer (same model) sees up to
8 of its errors and rewrites the instruction; the candidate is evaluated on the eval set (30 per task, the screen
questions) and accepted if mean accuracy does not drop. For every candidate vs its parent: per task, the paired change
is up / down when it exceeds two bootstrap SE. A rerun of the starting instruction (fresh calls) gives the noise level.
  .venv/bin/python qa/apo_conflicts.py [--steps 8] [--workers 32]
"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines"); sys.path.insert(0, f"{ROOT}/qa")
import gem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=8); ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/bbh/apo"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = "qa/bbh/apo"
TAIL = "Your final line must be: Answer: <answer> (for options, give the letter in parentheses, e.g. (A))."
norm = lambda s: " ".join(re.sub(r"^[\s\"'(\[]+|[\s\"'.)\]]+$", "", str(s)).replace(",", " ").lower().split())


def match(pred, gold):  # same rule as qa/screen.py
    if pred is None:
        return False
    p, g = norm(pred), norm(gold)
    if p == g:
        return True
    if re.fullmatch(r"\([A-Z]\)", str(gold).strip()):
        m = re.match(r"\(?([a-z])\)?(\s|$)", p)
        return bool(m) and m.group(1) == g
    return len(g.split()) == 1 and bool(p) and p.split()[0].strip("()") == g


def pools():
    from datasets import get_dataset_config_names, load_dataset
    ev, tr, rng = [], [], np.random.default_rng(0)
    for t in get_dataset_config_names("lukaemon/bbh"):
        d = load_dataset("lukaemon/bbh", t, split="test").to_pandas()
        idx = rng.permutation(len(d))  # same draw as qa/screen.py: first 30 = eval
        mk = lambda i: dict(qid=f"{t}/{i}", category=t, q=d.input[i], gold=d.target[i])
        ev += [mk(i) for i in idx[:30]]; tr += [mk(i) for i in idx[30:90]]
    return ev, tr


def answer(inst, r):
    try:
        txt, it, ot, _ = gem.chat([dict(role="user", content=r["q"].strip() + "\n\n" + inst.strip() + " " + TAIL)], max_tokens=3000)
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001
        txt, ot = None, 0
    m = re.findall(r"[Aa]nswer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
    pred = m[-1].strip() if m else None
    return dict(qid=r["qid"], category=r["category"], correct=int(match(pred, r["gold"])), out_tok=ot, pred=pred, txt=txt or "")


def evaluate(inst, rows, tag):
    f = f"{OUT}/{tag}.csv"
    if os.path.exists(f):
        return pd.read_csv(f)
    with ThreadPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(ex.map(lambda r: answer(inst, r), rows))
    df.drop(columns="txt").to_csv(f, index=False)
    return df


def compare(child, parent):
    rng = np.random.default_rng(0); up, dn = [], []
    m = child.merge(parent, on=["qid", "category"], suffixes=("", "_p"))
    for c, g in m.groupby("category"):
        d = (g.correct - g.correct_p).to_numpy()
        se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        if abs(d.mean()) > 2 * se and se > 0:
            (up if d.mean() > 0 else dn).append(c)
    return 100 * (child.correct.mean() - parent.correct.mean()), up, dn


OPT = """You are improving an instruction that is appended to every question of a mixed reasoning benchmark (27 different task types). The model answers with it, then gives its final answer.

Current instruction:
<instruction>{inst}</instruction>

Accuracy on a minibatch: {acc:.1f}%. Some questions it got wrong (task, question, the model's response ending, correct answer):
{errs}

Write an improved instruction that fixes these kinds of errors while staying general (it must work for all task types). Reply with only the new instruction inside <instruction></instruction>."""

ev, tr = pools()
rng = np.random.default_rng(1)
inst = "Think step by step."
cur = evaluate(inst, ev, "step0")
log = [dict(step=0, inst=inst, acc=100 * cur.correct.mean(), tok=cur.out_tok.mean())]
print(f"step 0 acc {log[0]['acc']:.1f}", flush=True)
gem.SALT.v = "rerun1"
rr = evaluate(inst, ev, "step0_rerun")
gem.SALT.v = ""
d, up, dn = compare(rr, cur)
print(f"rerun of step 0: change {d:+.1f}, tasks up {up or '-'}, down {dn or '-'}", flush=True)
log.append(dict(step="rerun", inst=inst, acc=100 * rr.correct.mean(), delta=d, up=up, down=dn))
for s in range(1, a.steps + 1):
    mb = [tr[i] for i in rng.choice(len(tr), 40, replace=False)]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda r: answer(inst, r), mb))
    wrong = [(r, x) for r, x in zip(mb, res) if not x["correct"]][:8]
    errs = "\n\n".join(f"[{r['category']}]\n{r['q'][:1200]}\n...model: {x['txt'][-400:]}\nCorrect answer: {r['gold']}" for r, x in wrong)
    txt, *_ = gem.chat([dict(role="user", content=OPT.format(inst=inst, acc=100 * np.mean([x["correct"] for x in res]), errs=errs or "(none)"))],
                       temperature=0.7, max_tokens=1500)
    m = re.findall(r"<instruction>(.*?)</instruction>", txt or "", re.S)
    cand = m[-1].strip() if m else inst
    ch = evaluate(cand, ev, f"step{s}")
    d, up, dn = compare(ch, cur)
    keep = ch.correct.mean() >= cur.correct.mean()
    log.append(dict(step=s, inst=cand, acc=100 * ch.correct.mean(), tok=ch.out_tok.mean(), delta=d, up=up, down=dn, keep=bool(keep)))
    print(f"step {s}: acc {100 * ch.correct.mean():.1f} ({d:+.1f}), tokens {ch.out_tok.mean():.0f}, up {up or '-'}, "
          f"down {dn or '-'}, {'keep' if keep else 'revert'}", flush=True)
    if keep:
        inst, cur = cand, ch
    json.dump(log, open(f"{OUT}/log.json", "w"), indent=1)
n = [x for x in log if x["step"] not in (0, "rerun")]
print(f"\nedits with a conflict (>=1 task up and >=1 down): {sum(bool(x['up']) and bool(x['down']) for x in n)}/{len(n)}; "
      f"with any significant task change: {sum(bool(x['up']) or bool(x['down']) for x in n)}/{len(n)}")
print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

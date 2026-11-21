"""Shared pieces of the BBH QA experiments: item pools, answer matching, one answer call."""
import re

import numpy as np

import gem

TAIL = "Your final line must be: Answer: <answer> (for options, give the letter in parentheses, e.g. (A))."
norm = lambda s: " ".join(re.sub(r"^[\s\"'(\[]+|[\s\"'.)\]]+$", "", str(s)).replace(",", " ").lower().split())


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


def pools():
    """eval = the 30 screen questions per task; train = the next 60 per task (same draw as qa/screen.py)."""
    from datasets import get_dataset_config_names, load_dataset
    ev, tr, rng = [], [], np.random.default_rng(0)
    for t in get_dataset_config_names("lukaemon/bbh"):
        d = load_dataset("lukaemon/bbh", t, split="test").to_pandas()
        idx = rng.permutation(len(d))
        mk = lambda i: dict(qid=f"{t}/{i}", category=t, q=d.input[i], gold=d.target[i])
        ev += [mk(i) for i in idx[:30]]; tr += [mk(i) for i in idx[30:90]]
    return ev, tr


def answer(inst, r, max_tokens=3000):
    try:
        txt, it, ot, _ = gem.chat([dict(role="user", content=r["q"].strip() + "\n\n" + inst.strip() + " " + TAIL)],
                                  max_tokens=max_tokens)
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001  one failed call counts as wrong, never stops the run
        txt, ot = None, 0
    m = re.findall(r"[Aa]nswer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
    pred = m[-1].strip() if m else None
    return dict(qid=r["qid"], category=r["category"], correct=int(match(pred, r["gold"])), out_tok=ot, pred=pred, txt=txt or "")

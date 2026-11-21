"""Mixed QA set for the factorial experiments: four sources with different demands, one shared instruction slot.

  tatqa    financial table + text, numeric reasoning (TAT-QA dev)
  musique  multi-hop reading over 20 paragraphs (MuSiQue validation, answerable)
  gsm8k    math word problems (GSM8K test)
  csqa     commonsense multiple choice (CommonsenseQA validation)
items(split, per) -> rows with source, prompt(inst) and score(txt) -> (correct, pred). Pools: "test" = first `per`
of a fixed permutation per source, "train" = the next `per`.
"""
import re, string
from collections import Counter

import numpy as np

import tatqa as T


def _snorm(s):
    s = "".join(ch for ch in str(s).lower() if ch not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def _last(txt, key="Answer"):
    m = re.findall(rf"{key}\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
    return m[-1].strip() if m else None


def _num(s):
    s = re.sub(r"[^0-9.\-]", "", str(s).replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return None


def _pick(n, per, split, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[:per] if split == "test" else idx[per:2 * per]


def items(split, per=100):
    from datasets import load_dataset
    rows = []
    # TAT-QA
    R = T.load("dev")
    for i in _pick(len(R), per, split, 0):
        r = R[i]
        rows.append(dict(qid=f"tatqa/{r['qid']}", source="tatqa", question=r["question"],
                         prompt=(lambda inst, r=r: T.prompt(r, inst)),
                         score=(lambda txt, r=r: T.score(txt, r)[:2])))
    # MuSiQue
    d = [x for x in load_dataset("dgslibisey/MuSiQue", split="validation") if x["answerable"]]
    for i in _pick(len(d), per, split, 1):
        x = d[int(i)]
        ctx = "\n\n".join(f"# {p['title']}\n{p['paragraph_text']}" for p in x["paragraphs"])
        golds = [x["answer"]] + list(x["answer_aliases"])
        rows.append(dict(qid=f"musique/{x['id']}", source="musique", question=x["question"],
                         prompt=(lambda inst, ctx=ctx, q=x["question"]: f"## Paragraphs\n{ctx}\n\n## Question\n{q}\n\n{inst.strip()}\n\n"
                                 "End with exactly one line: Answer: <a short answer>"),
                         score=(lambda txt, g=golds: (lambda p: (int(p is not None and any(_snorm(p) == _snorm(x) for x in g)), p))(_last(txt)))))
    # GSM8K
    g = load_dataset("openai/gsm8k", "main", split="test")
    for i in _pick(len(g), per, split, 2):
        x = g[int(i)]; gold = float(x["answer"].split("####")[-1].strip().replace(",", ""))
        rows.append(dict(qid=f"gsm8k/{int(i)}", source="gsm8k", question=x["question"],
                         prompt=(lambda inst, q=x["question"]: f"## Problem\n{q}\n\n{inst.strip()}\n\nEnd with exactly one line: Answer: <the number>"),
                         score=(lambda txt, gold=gold: (lambda p: (int(_num(p) is not None and abs(_num(p) - gold) < 1e-6), p))(_last(txt)))))
    # CommonsenseQA
    c = load_dataset("tau/commonsense_qa", split="validation")
    for i in _pick(len(c), per, split, 3):
        x = c[int(i)]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(x["choices"]["label"], x["choices"]["text"]))
        rows.append(dict(qid=f"csqa/{x['id']}", source="csqa", question=x["question"],
                         prompt=(lambda inst, q=x["question"], o=opts: f"## Question\n{q}\n\n{o}\n\n{inst.strip()}\n\nEnd with exactly one line: Answer: <letter>"),
                         score=(lambda txt, k=x["answerKey"]: (lambda p: (int(p is not None and re.sub(r"[^A-E]", "", p.upper())[:1] == k), p))(_last(txt)))))
    return rows

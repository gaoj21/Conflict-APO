"""TAT-QA (hybrid table + text QA over real financial reports): loading, prompt, scoring, and a quick screen.

Scoring follows the official metric in spirit: numbers match after normalization (commas, $, %, parentheses as
negative) within rounding (0.5% relative or 0.01 absolute) and the scale must match; spans match after lowercasing and
stripping punctuation and articles; multi-span answers match as sets.
  .venv/bin/python qa/tatqa.py screen [--n 300] [--workers 32]
Writes results/gemma_qa/tatqa/screen/runs.csv with the question's named properties (answer type, source, scale,
operation) next to correct / tokens, then prints accuracy by property.
"""
import argparse, json, os, re, string, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines")
import gem  # noqa: E402

SCALES = ["", "thousand", "million", "billion", "percent"]
FORMAT = ("End with exactly two lines:\nAnswer: <the answer; a number without units, or the text span(s) from the report, "
          "several spans separated by ' | '>\nScale: <none, thousand, million, billion or percent>")


def load(split):
    rows = []
    for doc in json.load(open(f"{ROOT}/data/tatqa/tatqa_dataset_{split}.json")):
        table = "\n".join(" | ".join(c.strip() for c in r) for r in doc["table"]["table"])
        paras = "\n".join(p["text"] for p in sorted(doc["paragraphs"], key=lambda p: p["order"]))
        for q in doc["questions"]:
            der = q.get("derivation") or ""
            op = ("division" if "/" in der else "multiplication" if "*" in der else "subtraction" if "-" in der
                  else "addition" if "+" in der else "none")
            rows.append(dict(qid=q["uid"], table=table, paras=paras, question=q["question"], gold=q["answer"],
                             answer_type=q["answer_type"], answer_from=q["answer_from"], scale=q["scale"], op=op))
    return rows


def prompt(r, inst):
    return (f"## Table\n{r['table']}\n\n## Text\n{r['paras']}\n\n## Question\n{r['question']}\n\n{inst.strip()}\n\n{FORMAT}")


def num(s):
    s = re.sub(r"(?i)\b(million|thousand|billion|percent)\b", "", str(s))
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    try:
        v = float(s.strip("()"))
    except ValueError:
        return None
    return -v if neg else v


def ntext(s):
    s = "".join(ch for ch in str(s).lower() if ch not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def same(p, g):
    a, b = num(p), num(g)
    if a is not None and b is not None:
        return abs(a - b) <= max(0.01, 0.005 * abs(b)) or abs(round(a, 2) - round(b, 2)) < 1e-9
    return ntext(p) == ntext(g)


def score(txt, r):
    a = re.findall(r"Answer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
    s = re.findall(r"Scale\s*[:：]\s*([A-Za-z]+)", (txt or "").replace("**", ""))
    if not a:
        return 0, None, 0.0
    sep = r"\s*\|\s*|;\s+|\s+and\s+" if r["answer_type"] == "multi-span" else r"\s*\|\s*"  # "A and B" only where several spans are asked
    pred = [x.strip() for x in re.split(sep, a[-1]) if x.strip()]
    gold = r["gold"] if isinstance(r["gold"], list) else [r["gold"]]
    scale = (s[-1].lower() if s else "none").replace("none", "")
    if r["answer_type"] in ("arithmetic", "count") and scale != r["scale"] and r["scale"] != "":
        return 0, a[-1], 0.0
    ok = len(pred) == len(gold) and all(any(same(p, g) for p in pred) for g in gold)
    return int(ok), a[-1], f1(pred, gold)


def f1(pred, gold):
    """Token F1 of the bags of words (numbers normalized), the official TAT-QA F1 in spirit."""
    tok = lambda xs: [f"{num(w):g}" if num(w) is not None else w for x in xs for w in ntext(x).split()]
    p, g = tok(pred), tok(gold)
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    return 0.0 if not common else 2 * common / (len(p) + len(g))


def answer(inst, r, max_tokens=3000):
    try:
        txt, it, ot, _ = gem.chat([dict(role="user", content=prompt(r, inst))], max_tokens=max_tokens)
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001
        txt, it, ot = None, 0, 0
    c, pred, f = score(txt, r)
    return dict(qid=r["qid"], answer_type=r["answer_type"], answer_from=r["answer_from"], scale=r["scale"], op=r["op"],
                correct=c, f1=f, pred=pred, gold=" | ".join(map(str, r["gold"] if isinstance(r["gold"], list) else [r["gold"]])),
                in_tok=it, out_tok=ot, failed=int(txt is None))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen"]); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    OUT = f"{gem.OUT}_qa/tatqa/screen"; os.makedirs(OUT, exist_ok=True)
    gem.TAG.v = "qa/tatqa/screen"
    rows = load("dev"); rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:a.n]]
    with ThreadPoolExecutor(a.workers) as ex:
        R = pd.DataFrame(ex.map(lambda r: answer("Think step by step.", r), rows))
    R.to_csv(f"{OUT}/runs.csv", index=False)
    print(f"EM x100 {100 * R.correct.mean():.1f}, F1 x100 {100 * R.f1.mean():.1f}  (n={len(R)}, unparsed {R.pred.isna().mean():.2f}, "
          f"out tokens {R.out_tok.mean():.0f})")
    for col in ("answer_type", "answer_from", "scale", "op"):
        g = R.groupby(col).correct.agg(["mean", "size"]); g["mean"] *= 100
        print(f"by {col}: " + ", ".join(f"{k or 'none'} {v['mean']:.0f} (n={int(v['size'])})" for k, v in g.iterrows()))
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

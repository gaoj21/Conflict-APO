"""Screen datasets whose answers follow writing conventions (rounding, percent vs fraction, units, span extent):
how many errors are about how the answer is written rather than what it is?

  .venv/bin/python qa/conv_screen.py <finqa|drop> [--n 300] [--workers 32]
strict  = the official exact-match style (format counts)
value   = the same answer up to format: numbers within 1% (percent <-> fraction allowed, sign must match); text spans
          with token F1 >= .5
The gap value - strict is the share of questions lost only to how the answer is written.
"""
import argparse, json, os, re, string, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines")
import gem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds", choices=["finqa", "drop"]); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/{a.ds}/screen"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = f"qa/{a.ds}/screen"
INST = "Think step by step."


def tnorm(s):
    s = "".join(ch for ch in str(s).lower() if ch not in string.punctuation or ch in ".%-")
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", str(s))
    if not m:
        return None, False
    v = float(m[-1].replace(",", "")) if len(m) == 1 else float(m[0].replace(",", ""))
    if re.search(r"\(\s*\$?\s*\d", str(s)) and not str(s).strip().startswith("-"):
        v = -abs(v)
    return v, "%" in str(s)


def f1(p, g):
    p, g = tnorm(p).split(), tnorm(g).split(); c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if not c else 2 * c / (len(p) + len(g))


def value_match(p, g):
    (pv, pp), (gv, gp) = num(p), num(g)
    if pv is None or gv is None:
        return f1(p, g) >= .5
    for f in (1, 100, .01):
        x = pv * f
        if abs(x - gv) <= max(.01 * abs(gv), .005) and (x == 0 or gv == 0 or np.sign(x) == np.sign(gv)):
            return True
    return False


def rows():
    if a.ds == "finqa":
        d = json.load(open(f"{ROOT}/data/finqa/test.json"))
        out = []
        for x in d:
            table = "\n".join(" | ".join(c for c in r) for r in x["table"])
            ctx = " ".join(x["pre_text"]) + "\n\n" + table + "\n\n" + " ".join(x["post_text"])
            if str(x["qa"]["answer"]).strip():
                out.append(dict(qid=x["id"], question=x["qa"]["question"], ctx=ctx, golds=[str(x["qa"]["answer"])]))
        return out
    from huggingface_hub import hf_hub_download
    d = pd.read_parquet(hf_hub_download("ucinlp/drop", "data/validation-00000-of-00001.parquet", repo_type="dataset"))
    return [dict(qid=r.query_id, question=r.question, ctx=r.passage, golds=list(dict.fromkeys(r.answers_spans["spans"])),  # validated alternatives
                 kind=str(r.answers_spans["types"][0])) for r in d.itertuples()]


def prompt(r):
    return (f"## Context\n{r['ctx']}\n\n## Question\n{r['question']}\n\n{INST}\n\nEnd with exactly one line: Answer: <the answer>")


def one(r):
    try:
        txt = gem.chat([dict(role="user", content=prompt(r))], max_tokens=3000)[0]
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001
        txt = None
    m = re.findall(r"Answer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
    p = m[-1].strip() if m else None
    g = r["golds"][0]
    strict = p is not None and any(tnorm(p).replace(" ", "") == tnorm(x).replace(" ", "") for x in r["golds"])
    val = p is not None and any(value_match(p, x) for x in r["golds"])
    return dict(qid=r["qid"], kind=r.get("kind", ""), question=r["question"], pred=p, gold=g, strict=int(strict),
                value=int(val or strict))


R = rows(); R = [R[i] for i in np.random.default_rng(0).permutation(len(R))[:a.n]]
with ThreadPoolExecutor(a.workers) as ex:
    D = pd.DataFrame(ex.map(one, R))
D.to_csv(f"{OUT}/runs.csv", index=False)
gap = D[(D.value == 1) & (D.strict == 0)]
print(f"[{a.ds}] n={len(D)}  strict {100 * D.strict.mean():.1f}  value {100 * D.value.mean():.1f}  "
      f"lost only to the writing: {100 * len(gap) / len(D):.1f} points ({100 * len(gap) / max(1, (D.strict == 0).sum()):.0f}% of strict errors)")
if a.ds == "drop":
    print(D.groupby("kind")[["strict", "value"]].mean().mul(100).round(1).to_string())
print("examples lost only to the writing (model -> reference):")
for x in gap.head(15).itertuples():
    print(f"   {str(x.pred)[:60]!r} -> {x.gold[:60]!r}")
print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

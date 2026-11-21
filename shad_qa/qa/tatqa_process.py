"""Process features of TAT-QA arithmetic answers, extracted the same way from the model's reasoning and from the gold
derivation, to see where the model's way of computing deviates systematically from the gold one.

  .venv/bin/python qa/tatqa_process.py [--workers 32]
1. answers every dev arithmetic question with "Think step by step." (cached for the 122 already screened);
2. one extractor prompt (same model, temperature 0) reads either the model's solution or the gold solution
   (derivation + answer) and returns the same JSON of process features;
3. prints, per feature, how often model and gold disagree and in which direction, overall and on wrong answers.
Writes results/gemma_qa/tatqa/process/features.csv.
"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines"); sys.path.insert(0, f"{ROOT}/qa")
import gem  # noqa: E402
from tatqa import load, prompt, score  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/tatqa/process"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = "qa/tatqa/process"
INST = "Think step by step."
FEATS = {
    "operation": "lookup | difference | sum | ratio | percent_change | average | product | other",
    "n_operands": "how many numbers from the report enter the computation (integer)",
    "order": "for a difference or percent change: later_minus_earlier | earlier_minus_later; otherwise na",
    "denominator": "for a ratio or percent change: earlier | later | total | other; otherwise na",
    "times_100": "true if the result is expressed as a percentage (multiplied by 100), else false",
    "sign": "negative | positive | zero (of the final answer)",
}
EXTRACT = ("Below is a solution to a question about a financial report. Describe how it computes the answer. "
           "Reply with only one line of JSON with these keys:\n" + json.dumps(FEATS, indent=0) +
           "\n\n## Question\n{q}\n\n## Solution\n{sol}")


def extract(q, sol):
    try:
        txt, *_ = gem.chat([dict(role="user", content=EXTRACT.replace("{q}", q).replace("{sol}", sol))], max_tokens=300)
        f = json.loads(re.findall(r"\{.*\}", txt or "", re.S)[-1])
        return {k: str(f.get(k, "na")).lower() for k in FEATS}
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001
        return {k: np.nan for k in FEATS}


def one(r):
    txt, *_ = gem.chat([dict(role="user", content=prompt(r, INST))], max_tokens=3000)  # same call as the screen
    c, pred, _ = score(txt, r)
    gold = r["gold"] if isinstance(r["gold"], list) else [r["gold"]]
    o = extract(r["question"], txt[-2500:] if txt else "")
    g = extract(r["question"], f"Computation: {r['derivation']}\nAnswer: {' | '.join(map(str, gold))}"
                               + (f" ({r['scale']})" if r["scale"] else ""))
    return dict(qid=r["qid"], question=r["question"], correct=c, pred=pred, gold=" | ".join(map(str, gold)), scale=r["scale"],
                answer_from=r["answer_from"], **{f"out_{k}": v for k, v in o.items()}, **{f"gt_{k}": v for k, v in g.items()})


rows = [r for r in load("dev") if r["answer_type"] == "arithmetic"]
der = {q["uid"]: q.get("derivation", "") for d in json.load(open(f"{ROOT}/data/tatqa/tatqa_dataset_dev.json")) for q in d["questions"]}
for r in rows:
    r["derivation"] = der[r["qid"]]
with ThreadPoolExecutor(a.workers) as ex:
    D = pd.DataFrame(ex.map(one, rows))
D.to_csv(f"{OUT}/features.csv", index=False)
print(f"{len(D)} arithmetic questions, accuracy x100 {100 * D.correct.mean():.1f}")
for k in FEATS:
    m = D[f"out_{k}"] != D[f"gt_{k}"]
    print(f"\n== {k}: model and gold differ on {100 * m.mean():.1f}% (on wrong answers {100 * m[D.correct == 0].mean():.1f}%, "
          f"on right answers {100 * m[D.correct == 1].mean():.1f}%)")
    pairs = D[m].groupby([f"gt_{k}", f"out_{k}"]).size().sort_values(ascending=False).head(6)
    print("   most frequent (gold -> model): " + ", ".join(f"{g} -> {o}: {n}" for (g, o), n in pairs.items()))
print(f"\ndone, spent ${gem.spent()[0]:.2f}", flush=True)

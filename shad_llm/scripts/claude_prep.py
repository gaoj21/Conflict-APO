"""Label-free rows + gold for the per-call Claude judge, for any scoring dataset.

  python scripts/claude_prep.py {hs2,se,br}
writes results/claude_<ds>/chunks/{fit,sel,test}_00.jsonl ({id, user}), gold.csv (id, split, attrs..., words, z),
attrs.txt, format.txt, prompts/v0.txt (the dataset's initial instructions, as in the TextGrad runs).
HS2 = HelpSteer2SafeG (fit/sel = alternate train rows, test = first 480); SE = SummEval (fit/sel = alternate rows of
the 160 train rows, test = 480); BR = BRIGHTER (fit = train rows 0-599, sel = 600-1199, test = first 480).
"""
import json, os, sys
import numpy as np, pandas as pd

DS = sys.argv[1]
OUT = f"results/claude_{DS}"
CFG = {
    "hs2": dict(path="data/HelpSteer2SafeG/HelpSteer2SafeG", attrs=["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
                lo=0, hi=4, len_col="response",
                render=lambda r: f"## User prompt:\n{r.prompt}\n\n## Assistant response:\n{r.response}\n\n## Your evaluation:"),
    "se": dict(path="data/SummEval/SummEval", attrs=["fluency", "relevance", "coherence", "consistency"], lo=1, hi=5,
               len_col="machine_summary",
               render=lambda r: f"## Source Text:\n{r.text}\n\n## Summary:\n{r.machine_summary}\n\n## Your evaluation:"),
    "br": dict(path="data/BRIGHTER/BRIGHTER", attrs=["anger", "fear", "joy", "sadness", "surprise"], lo=0, hi=3,
               len_col="text", render=lambda r: f"## Text:\n{r.text}\n\n## Your evaluation:"),
}[DS]
V0 = {
    "hs2": None,  # written earlier (results/claude_hs2/prompts/v0.txt)
    "se": """You are a careful, calibrated evaluator. Your goal is to produce an accurate evaluation by following the Instructions below.

## Task
Evaluate the Summary given the Source Text using the Instructions below.
1. Consider every strength and flaw you find when making your evaluation.
2. Based on the number and severity of the strengths and flaws, assign a value.

## Instructions:
- fluency: Evaluate the fluency and readability of the summary. Rate from 1 to 5.
- relevance: Evaluate whether the summary captures the most important information from the source text. Rate from 1 to 5.
- coherence: Evaluate the logical structure and organization of the summary. Rate from 1 to 5.
- consistency: Evaluate whether the summary is factually consistent with the source text. Rate from 1 to 5.""",
    "br": """Evaluate the emotion intensities in the Text.

## Instructions:
- anger: Evaluate the intensity of anger expressed in the text. Rate anger intensity from 0 (none) to 3 (most intense).
- fear: Evaluate the intensity of fear expressed in the text. Rate fear intensity from 0 (none) to 3 (most intense).
- joy: Evaluate the intensity of joy expressed in the text. Rate joy intensity from 0 (none) to 3 (most intense).
- sadness: Evaluate the intensity of sadness expressed in the text. Rate sadness intensity from 0 (none) to 3 (most intense).
- surprise: Evaluate the intensity of surprise expressed in the text. Rate surprise intensity from 0 (none) to 3 (most intense).""",
}[DS]
tr, te = pd.read_parquet(CFG["path"] + "-train.parquet"), pd.read_parquet(CFG["path"] + "-test.parquet").iloc[:480]
parts = ({"fit": tr.iloc[:600], "sel": tr.iloc[600:1200], "test": te} if DS == "br"
         else {"fit": tr.iloc[0::2], "sel": tr.iloc[1::2], "test": te})
for d in ("chunks", "prompts", "out"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
gold = []
for sp, d in parts.items():
    d = d.reset_index(drop=True)
    with open(f"{OUT}/chunks/{sp}_00.jsonl", "w") as f:
        for i, r in d.iterrows():
            f.write(json.dumps({"id": f"{sp}{i:04d}", "user": CFG["render"](r)}, ensure_ascii=False) + "\n")
            w = len(str(r[CFG["len_col"]]).split())
            gold.append(dict(id=f"{sp}{i:04d}", split=sp, **{a: float(r[a]) for a in CFG["attrs"]}, words=w, z=float(np.log1p(w))))
pd.DataFrame(gold).to_csv(f"{OUT}/gold.csv", index=False)
open(f"{OUT}/attrs.txt", "w").write("\n".join(CFG["attrs"]))
keys = ", ".join(f'"{a}": {CFG["lo"]}-{CFG["hi"]}' for a in CFG["attrs"])
open(f"{OUT}/format.txt", "w").write(
    "## Output format:\nThink briefly (at most a few sentences), then end your reply with exactly one line of JSON with an "
    f"integer from {CFG['lo']} to {CFG['hi']} for each attribute:\n{{{keys}}}")
if V0:
    open(f"{OUT}/prompts/v0.txt", "w").write(V0 + "\n")
print(DS, pd.DataFrame(gold).split.value_counts().to_dict())

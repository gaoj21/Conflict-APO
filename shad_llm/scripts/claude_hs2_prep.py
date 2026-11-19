"""Claude-in-the-loop on HelpSteer2SafeG: label-free rows for the per-call judge, gold kept separately.

Train (1159) is split by position into 'fit' (failures are read here) and 'sel' (edits are judged here); test (first
480 rows, as in the TextGrad runs) is touched only at the end.
  results/claude_hs2/chunks/<split>_00.jsonl  {id, user}   (rendered judge input: prompt + response, no labels)
  results/claude_hs2/gold.csv                 id, split, <5 attributes>, words, z (= log words)
"""
import json
import numpy as np, pandas as pd

OUT = "results/claude_hs2"
ATTR = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
tr = pd.read_parquet("data/HelpSteer2SafeG/HelpSteer2SafeG-train.parquet")
te = pd.read_parquet("data/HelpSteer2SafeG/HelpSteer2SafeG-test.parquet").iloc[:480]
parts = {"fit": tr.iloc[0::2], "sel": tr.iloc[1::2], "test": te}
gold = []
for sp, d in parts.items():
    d = d.reset_index(drop=True)
    with open(f"{OUT}/chunks/{sp}_00.jsonl", "w") as f:
        for i, r in d.iterrows():
            f.write(json.dumps({"id": f"{sp}{i:03d}", "user": f"## User prompt:\n{r.prompt}\n\n## Assistant response:\n"
                                f"{r.response}\n\n## Your evaluation:"}, ensure_ascii=False) + "\n")
            w = len(r.response.split())
            gold.append(dict(id=f"{sp}{i:03d}", split=sp, **{a: int(r[a]) for a in ATTR}, words=w, z=float(np.log1p(w))))
pd.DataFrame(gold).to_csv(f"{OUT}/gold.csv", index=False)
print(pd.DataFrame(gold).split.value_counts().to_dict())

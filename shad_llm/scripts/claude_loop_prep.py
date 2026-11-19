"""Claude-in-the-loop on PPELen: label-free chunks for judge subagents, gold labels kept separately.

Train (572) is split by pairs of rows into 'fit' (failures are read here) and 'sel' (edits are
judged here); test (480 = first 480 rows of PPELen-test, like the TextGrad runs) is touched only at the end.
  results/claude_loop/chunks/<split>_<k>.jsonl  {id, prompt, response_1, response_2}   (no kind, no label)
  results/claude_loop/gold.csv                  id, split, kind, preference, w1, w2, z
"""
import json, os
import numpy as np, pandas as pd

OUT, CH = "results/claude_loop", 48
tr = pd.read_parquet("data/PPELen/PPELen-train.parquet")
te = pd.read_parquet("data/PPELen/PPELen-test.parquet").iloc[:480]
blk = (np.arange(len(tr)) // 2) % 2  # kinds alternate row by row: split by pairs of rows
parts = {"fit": tr[blk == 0], "sel": tr[blk == 1], "test": te}
gold = []
for sp, d in parts.items():
    d = d.reset_index(drop=True)
    for k in range(0, len(d), CH):
        with open(f"{OUT}/chunks/{sp}_{k // CH:02d}.jsonl", "w") as f:
            for i, r in d.iloc[k:k + CH].iterrows():
                f.write(json.dumps({"id": f"{sp}{i:03d}", "prompt": r.prompt, "response_1": r.response_1,
                                    "response_2": r.response_2}, ensure_ascii=False) + "\n")
    for i, r in d.iterrows():
        w1, w2 = len(r.response_1.split()), len(r.response_2.split())
        gold.append(dict(id=f"{sp}{i:03d}", split=sp, kind=r.kind, preference=int(r.preference), w1=w1, w2=w2,
                         z=float(np.log((w1 + 1) / (w2 + 1)))))
pd.DataFrame(gold).to_csv(f"{OUT}/gold.csv", index=False)
print(pd.DataFrame(gold).groupby(["split", "kind"]).size())

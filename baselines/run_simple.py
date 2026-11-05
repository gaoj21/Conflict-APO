"""Non-optimized judges: v0 (the initial rubric prompt) and few-shot ICL (v0 + k labelled fit items with their human
scores, drawn with the seed).
  python baselines/run_simple.py <ds> v0|fewshot [--seed 0] [--k 5]
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("method"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--k", type=int, default=5); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
gem.TAG.v = f"{a.ds}/{a.method}/seed{a.seed}"
if a.method == "v0":
    instr = ds.v0
else:
    ex = ds.subset("fit", a.k, seed=1000 + a.seed)
    shots = "\n\n".join(f"### Example {i + 1}\n{r['user'].strip()}\n\nHuman ratings: " +
                        json.dumps({x: int(round(v)) for x, v in zip(ds.A, ds.human([r["id"]])[0])})
                        for i, r in enumerate(ex))
    instr = ds.v0 + "\n\n## Examples rated by human experts\n\n" + shots
gem.judge_all_splits(ds, a.method, a.seed, instr, a.workers)
print(a.ds, a.method, a.seed, "spent $%.3f" % gem.spent(gem.TAG.v)[0])

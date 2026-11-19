"""Raw and ShAD test Spearman x100 (human scores) of APO on human vs content labels.
  .venv/bin/python scripts/eval_content.py [ds ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
from gem import DS, OUT  # noqa: E402
from evaluate import scores  # noqa: E402

for n in sys.argv[1:] or ["hs2", "ell", "hanna"]:
    ds = DS(n)
    for m in ("v0", "dspy_mipro", "dspy_mipro_content", "textgrad_ccc", "textgrad_ccc_content"):
        d = f"{OUT}/{n}/{m}/seed0"
        if not os.path.exists(f"{d}/scores_test.csv"):
            continue
        r = scores(ds, d)
        print(f"{n:6s} {m:22s} raw {100*np.nanmean(r['raw']):5.1f}  +ShAD {100*np.nanmean(r['readout']):5.1f}  | raw per attr",
              dict(zip(ds.A, np.round(100 * r["raw"], 1))))

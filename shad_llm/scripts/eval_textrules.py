"""Collect run_textrules.py test results (raw Spearman x100, no decoder) into one table."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
import gem  # noqa: E402
from evaluate import scores, sp  # noqa: E402

ROWS = [("v0 + half + word count", ["textrules_d0_norules"]), ("rules", ["textrules_d0"]),
        ("ShAD demos", ["textrules_d4_norules"]), ("random demos (3 draws)", ["textrules_d4_norules_random", "textrules_d4_norules_random1", "textrules_d4_norules_random2"]),
        ("rules + ShAD demos", ["textrules_d4"]), ("rules + random demos (3 draws)", ["textrules_d4_random", "textrules_d4_random1", "textrules_d4_random2"])]
out = {}
for n in ["hs2", "hanna", "ell", "merlin"]:
    ds = gem.DS(n)
    for name, ms in ROWS:
        vals = []
        for m in ms:
            f = f"{gem.OUT}/{n}/{m}/seed0/scores_test.csv"
            if os.path.exists(f):
                df = pd.read_csv(f); H = ds.human(df.id)
                vals.append(100 * np.nanmean([sp(H[:, k], df[a].to_numpy(float)) for k, a in enumerate(ds.A)]))
        out.setdefault(name, {})[n] = (f"{np.mean(vals):.1f}" + (f"±{np.std(vals):.1f}" if len(vals) > 1 else "")) if vals else "-"
    for name, m in (("MIPROv2 raw", "dspy_mipro"), ("v0 + ShAD (decoder)", "v0")):
        r = scores(ds, f"{gem.OUT}/{n}/{m}/seed0")
        out.setdefault(name, {})[n] = f"{100 * np.nanmean(r['raw' if 'raw' in name else 'readout']):.1f}"
print(pd.DataFrame(out).T.to_string())

"""Other observable axes: does the readout gain (on v0, test) when cheap surface features of the item are added next to
length (newlines, list items, headers, bold, digits, '!', '?', capitals, sentences)? A feature that helps is an axis
humans reward differently per attribute, i.e. one along which a prompt edit could fake an improvement.
  python scripts/claude_axes.py results/claude_hs2r
"""
import os, re, sys
import numpy as np, pandas as pd
sys.argv, ARGS = sys.argv[:1] + [sys.argv[1], "x"], sys.argv[1:]
exec(open(os.path.join(os.path.dirname(__file__), "claude_opt.py")).read().split("# ---------------------------------------------------------------- main loop")[0])
SURF = {"newline": r"\n", "list": r"\n\s*(?:[-*•]|\d+[.)])\s", "header": r"\n#+ ", "bold": r"\*\*", "digit": r"\d",
        "excl": r"!", "quest": r"\?", "caps": r"\b[A-Z]{2,}\b", "sent": r"[.!?](?:\s|$)"}


def add(d, keys):
    d = d.copy()
    for k in keys:
        d[f"{k}_j"] = [np.log1p(len(re.findall(SURF[k], USER[i]))) for i in d.id]
    return d


tr = pd.concat([frame(pd.read_csv(f"{R}/out/v0_{s}.csv"), s) for s in ("fit", "sel")], ignore_index=True)
te = frame(pd.read_csv(f"{R}/out/v0_test.csv"), "test")
H = te[A].to_numpy(float)
rng = np.random.default_rng(0)
def run(keys):
    a_, b_ = add(tr, keys), add(te, keys)
    return per_task(H, readout_scores(a_, b_))
base, raw = run([]), per_task(H, te[[x + "_j" for x in A]].to_numpy(float))
allk = run(list(SURF))
print(f"=== {os.path.basename(R)} v0 test Spearman: {' '.join(x[:6] for x in A)} | mean")
print(f"  raw                     " + " ".join(f"{x:6.3f}" for x in raw) + f" | {np.nanmean(raw):.3f}")
print(f"  + length                " + " ".join(f"{x:6.3f}" for x in base) + f" | {np.nanmean(base):.3f}")
print(f"  + length + all surface  " + " ".join(f"{x:6.3f}" for x in allk) + f" | {np.nanmean(allk):.3f}")
for k in SURF:
    r = run([k]); print(f"  + length + {k:12s} " + " ".join(f"{x:6.3f}" for x in r) + f" | {np.nanmean(r):.3f}")

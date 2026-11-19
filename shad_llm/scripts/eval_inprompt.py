"""Re-parse and evaluate run_inprompt.py outputs (final ratings may come as "<attr>" or "final_<attr>").
  .venv/bin/python scripts/eval_inprompt.py [ds ...]"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
import gem  # noqa: E402
from evaluate import scores, sp  # noqa: E402

for n in sys.argv[1:] or ["hs2", "ell", "hanna"]:
    ds = gem.DS(n); A, K = ds.A, len(ds.A); d = f"{gem.OUT}/{n}/inprompt_v0/seed0"
    df = pd.read_csv(f"{d}/scores_test.csv"); rule = json.load(open(f"{d}/rule.json"))
    sc = df["reason"].fillna("").map(gem.parse_scores)
    fin = np.array([[x.get(a, x.get("final_" + a, np.nan)) for a in A] for x in sc], float)
    ini = np.array([[x.get("initial_" + a, np.nan) for a in A] for x in sc], float)
    cuts = rule["cuts"]; lv = np.array([1 + int(np.searchsorted(cuts, len(ds.text[i].split()), side="right")) for i in df.id], float)
    G = np.c_[ini, lv]
    ours = np.stack([rule["rules"][a]["icpt"] + G[:, rule["rules"][a]["idx"]] @ np.array(rule["rules"][a]["coef"]) for a in A], 1)
    H = ds.human(df.id)
    r = lambda P: 100 * np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)])
    v0 = scores(ds, f"{gem.OUT}/{n}/v0/seed0")
    print(f"{n:6s} unparsed {np.isnan(fin).any(1).mean():5.1%} | v0 raw {100*np.nanmean(v0['raw']):.1f}  v0+ShAD {100*np.nanmean(v0['readout']):.1f}"
          f" | judge initial {r(ini):.1f}  judge final (no decoder) {r(fin):.1f}  rule applied by us {r(ours):.1f}"
          f" | fidelity {100*np.nanmean([sp(fin[:, k], ours[:, k]) for k in range(K)]):.1f}")

"""Tokens and cost of every job of the run (from each job's usage.json, written live while it runs) and the total.
  python baselines/usage.py [ds ...]
"""
import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

names = sys.argv[1:] or list(common.DATASETS)
tot = dict(calls=0, uncounted=0, input_tokens=0, output_tokens=0, cost=0.0)
priced = True
for f in sorted(glob.glob(f"{common.OUT}/*/*/seed*/usage.json")):
    ds, method, seed = f.split("/")[-4:-1]
    if ds not in names:
        continue
    u = json.load(open(f))
    print(f"{ds:<7} {method:<16} {seed:<6} {u.get('calls', 0):>8} calls  {common.fmt_usage(u)}")
    for k in ("calls", "uncounted", "input_tokens", "output_tokens"):
        tot[k] += u.get(k) or 0
    if isinstance(u.get("cost"), (int, float)):
        tot["cost"] += u["cost"]
    else:
        priced = False
if not priced:
    tot["cost"] = None
print(f"TOTAL   {tot['calls']:>32} calls  {common.fmt_usage(tot)}")

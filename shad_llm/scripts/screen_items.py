"""Drop failed items before the experiment: every item whose v0 judge call failed (API error) or returned no parsable
rating for some attribute is added to $GEM_OUT/<ds>/excluded_ids.txt; all later steps skip it (gem.DS) and all
evaluations ignore it (DS.keep). Run right after the v0 judge.
  python scripts/screen_items.py <ds> [<ds> ...]"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
import gem  # noqa: E402

for n in sys.argv[1:]:
    ds = gem.DS(n)
    d = f"{gem.OUT}/{n}/v0/seed0"
    df = pd.concat([pd.read_csv(f"{d}/scores_{s}.csv") for s in ("fit", "sel", "test")], ignore_index=True)
    bad = df[ds.A].isna().any(axis=1)
    if "error" in df:
        bad |= df["error"].notna()
    ds.exclude(df.id[bad].tolist())
    n_all = sum(len(v) for v in gem.DS(n).rows.values()) + len(ds.excluded)
    print(f"{n}: {len(ds.excluded)} of {n_all} items excluded ({100 * len(ds.excluded) / max(n_all, 1):.1f}%)")

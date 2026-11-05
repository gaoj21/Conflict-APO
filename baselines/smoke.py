"""Smoke test of the whole path (llm -> judge -> score parsing) on a few items, before any run.
Makes real, billed calls (3 judge calls plus one optimizer-style call) and prints what came back.
  python baselines/smoke.py [ds] [--n 3]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds", nargs="?", default="hanna"); ap.add_argument("--n", type=int, default=3)
a = ap.parse_args()
ds = common.DS(a.ds)
print(f"results -> {common.OUT}")
print(f"{a.ds}: attributes {ds.A}, scale {ds.LO}-{ds.HI}, fit/sel/test = "
      f"{len(ds.rows['fit'])}/{len(ds.rows['sel'])}/{len(ds.rows['test'])}")
rows = ds.rows["test"][:a.n]
df = ds.judge(ds.v0, rows, workers=a.n, strict=False)
for _, r in df.iterrows():
    print(r["id"], {x: r[x] for x in ds.A})
print("human:", ds.gold.loc[[r["id"] for r in rows], ds.A].round(2).to_dict("records"))
txt = common.chat([dict(role="user", content="Rewrite this instruction in one sentence: " + ds.v0[:400])], temperature=0.7)
print("optimizer-style call:", repr(txt[:200]))
parsed = df[ds.A].notna().all(axis=1).mean()
print(f"parsed {parsed:.0%} of judge outputs")
print("OK" if parsed == 1 else "CHECK: some judge outputs did not parse (see the 'reason' column / provider params)")

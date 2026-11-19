"""Test Spearman x100 (ShAD fitted on fit + sel) for boosted APO vs its noise control, one dataset.
  .venv/bin/python scripts/boost_eval.py <ds> <method> [<method> ...]   (method dirs as <method>/seed<s>)"""
import os, sys
sys.argv, args = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_archive as E  # noqa: E402  (prints its own table first)
from gem import DS, OUT  # noqa: E402

ds = DS(args[0])
def part(m):
    d = f"{OUT}/{args[0]}/{m}"
    return E.load(ds, {s: f"{d}/scores_{s}.csv" for s in ("fit", "sel", "test")})
combos = [a.split("+") for a in args[1:]]
print("\n=== boosting,", args[0])
for c in combos:
    raw, dec = E.evaluate(ds, [part(m) for m in c])
    print(f"{' + '.join(c):70s} calls {len(c)}  raw(first) {raw:.1f}  ShAD {dec:.1f}")

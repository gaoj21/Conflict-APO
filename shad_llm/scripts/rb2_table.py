"""Per-kind accuracy and token change of every saved variant vs `base` (paired bootstrap 95% CI, * = excludes 0).

Usage: python scripts/rb2_table.py <out_dir> [order.json]
"""

import glob
import json
import os
import sys

import numpy as np

KINDS = ["focus", "factuality", "safety", "math", "precise_if"]


def summary(d, ids):
    v = {}
    for k in KINDS:
        rows = [d[i] for i in ids if d[i]["kind"] == k]
        v[k] = float(np.mean([r["pred"] == r["gt"] for r in rows])) if rows else float("nan")
    v["in_tok"] = float(np.mean([d[i]["in"] or 0 for i in ids]))
    v["out_tok"] = float(np.mean([d[i]["out"] or 0 for i in ids]))
    v["err"] = float(np.mean([d[i]["err"] for i in ids]))
    return v


def main(out_dir, order=None):
    runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(out_dir, "*.json"))}
    base = runs["base"]
    ids = sorted(base)
    labels = [x for x in (order or sorted(runs)) if x in runs and x != "base"]
    b0 = summary(base, ids)
    print("base:", {k: round(v, 3) for k, v in b0.items()})
    rng = np.random.default_rng(0)
    boots = [rng.choice(len(ids), len(ids)) for _ in range(500)]
    cols = KINDS + ["out_tok", "err"]
    print(f"{'':24s}" + "".join(f"{c:>12s}" for c in cols) + "   (accuracy points; out_tok %; err points)")
    for lab in labels:
        d = runs[lab]
        v = summary(d, ids)
        cells = []
        for c in cols:
            diff = lambda a, b: (a[c] / b[c] - 1) if c == "out_tok" else (a[c] - b[c])  # noqa: E731
            bs = []
            for b in boots:
                sub = [ids[i] for i in b]
                bs.append(diff(summary(d, sub), summary(base, sub)))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            cells.append(f"{100 * diff(v, b0):+11.1f}{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{lab:24s}" + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1], json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None)

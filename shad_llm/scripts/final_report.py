"""Final report of the designed experiment (one table per dataset).

Rows (all test, mean Spearman x100 over attributes; raw = the judge's own ratings, no decoder):
  v0                       unoptimized rubric                                 raw  | + ShAD decoder
  TextGrad                 plain TextGrad CCC                                 raw  | + ShAD decoder
  TextGrad + format        same prompt, word count shown, half points          raw          (control for all rules)
  + length rules           one-shot, ShAD length weights as per-band points   raw
  + text rules             one-shot, ShAD cross-attribute weights in words    raw
  + combined               one-shot, text + length rules                      raw
  TextGrad x ShAD rounds   rules updated after every TextGrad step            raw  | + ShAD decoder
Conflict columns (rule rows vs "TextGrad + format"): attributes whose Spearman changes by more than 2 paired-bootstrap
SE (test items) up / down; conflict = at least one up and one down.
  python scripts/final_report.py [ds ...]      (uses gem.OUT, so the same GEM_PROVIDER / GEM_OUT as the runs)
"""
import glob, json, os, sys

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
import gem  # noqa: E402
from evaluate import scores, sp  # noqa: E402


def mat(ds, f):
    df = ds.worst_case(pd.read_csv(f)).set_index("id")
    return df.loc[[r["id"] for r in ds.rows["test"]], ds.A].to_numpy(float)


def first(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[0] if fs else None


def conflict(ds, H, P0, P1, B=300):
    K = len(ds.A)
    d = np.array([sp(H[:, k], P1[:, k]) - sp(H[:, k], P0[:, k]) for k in range(K)])
    rng = np.random.default_rng(0); bs = []
    for _ in range(B):
        i = rng.integers(0, len(H), len(H)); bs.append([sp(H[i, k], P1[i, k]) - sp(H[i, k], P0[i, k]) for k in range(K)])
    se = np.nanstd(np.array(bs), 0)
    up, dn = int((d > 2 * se).sum()), int((d < -2 * se).sum())
    return up, dn, int(up > 0 and dn > 0), " ".join(f"{a[:6]}{100 * v:+.1f}" for a, v in zip(ds.A, d))


for n in sys.argv[1:] or ["hs2", "ell", "merlin"]:
    ds = gem.DS(n); H = ds.human([r["id"] for r in ds.rows["test"]]); K = len(ds.A)
    m = lambda P: 100 * np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)])
    O = gem.OUT; rows = []

    def add(name, raw_file=None, method_dir=None, base=None):
        r = dict(setting=name)
        if method_dir and os.path.exists(f"{method_dir}/scores_test.csv"):
            s = scores(ds, method_dir); r["raw"] = 100 * np.nanmean(s["raw"]); r["+ShAD"] = 100 * np.nanmean(s["readout"])
            raw_file = f"{method_dir}/scores_test.csv"
        elif raw_file and os.path.exists(raw_file):
            r["raw"] = m(mat(ds, raw_file))
        else:
            return
        if base is not None and raw_file:
            r["up"], r["down"], r["conflict"], r["per attribute (vs format)"] = conflict(ds, H, base, mat(ds, raw_file))
        rows.append(r)

    add("v0", method_dir=f"{O}/{n}/v0/seed0")
    add("TextGrad", method_dir=f"{O}/{n}/textgrad_ccc/seed0")
    fmt = f"{O}/{n}/postrules_textgrad_ccc/seed0/scores_test_base.csv"
    base = mat(ds, fmt) if os.path.exists(fmt) else None
    add("TextGrad + format", raw_file=fmt)
    add("+ length rules", raw_file=f"{O}/{n}/postrules_textgrad_ccc/seed0/scores_test_rules.csv", base=base)
    add("+ text rules", raw_file=first(f"{O}/{n}/llmrules_textgrad_ccc/seed0/c*_test.csv"), base=base)
    add("+ combined", raw_file=first(f"{O}/{n}/llmrules_comb_textgrad_ccc/seed0/c*_test.csv"), base=base)
    add("TextGrad x ShAD rounds", method_dir=f"{O}/{n}/textgrad_ccc_shadrules/seed0", base=base)
    if rows:
        print(f"\n=== {n} (test Spearman x100)")
        with pd.option_context("display.width", 250, "display.max_colwidth", 90):
            print(pd.DataFrame(rows).round(1).fillna("").to_string(index=False))

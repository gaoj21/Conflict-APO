"""ShAD executed by the judge itself: no decoder at inference.

ShAD is fitted on the base prompt's fit + sel scores (attribute scores + a length level), its weights are converted to
a short linear rule per attribute on the original scale, and the rule is written into the prompt. In one call the judge
rates the attributes as before ("initial"), reads the word count given with the item, looks up the length level, and
computes each final rating with the rule. We evaluate the judge's final ratings directly (test Spearman, no decoder),
and check how faithfully it applied the rule (Spearman with the rule applied by us to its own initial ratings).
  .venv/bin/python baselines/run_inprompt.py <ds> [--terms 3]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--base", default="v0"); ap.add_argument("--terms", type=int, default=3)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = f"inprompt_{a.base}"
gem.TAG.v = f"{a.ds}/{METHOD}/seed0"
OUTD = ds.out(METHOD, 0)
A, K = ds.A, len(ds.A)
words = lambda i: len(ds.text[i].split())

# length level 1-5 from word-count quintiles of the labelled items
tr = pd.concat([pd.read_csv(f"{gem.OUT}/{a.ds}/{a.base}/seed0/scores_{s}.csv") for s in ("fit", "sel")], ignore_index=True)
W_tr = np.array([words(i) for i in tr.id])
cuts = np.quantile(W_tr, [.2, .4, .6, .8]).round().astype(int)
level = lambda w: 1 + int(np.searchsorted(cuts, w, side="right"))

# ShAD on [scores, level] in standardized units, converted to the original scale
X = tr[A].to_numpy(float); X = np.where(np.isnan(X), np.nanmean(X, 0), X)
L = np.array([level(w) for w in W_tr], float)
F = np.c_[X, L]; m, s = F.mean(0), F.std(0)
Y = ds.human(tr.id); my, sy = Y.mean(0), Y.std(0)
Wst = anchored_fit((F - m) / s, (Y - my) / sy)(np.eye(K + 1))  # (K+1, K)
coef = Wst * sy[None, :] / s[:, None]  # original units
names = [f"initial {x}" for x in A] + ["length level"]
rules, kept = [], {}
for k in range(K):
    imp = np.abs(Wst[:, k])
    idx = sorted(np.argsort(-imp)[:a.terms])
    c = coef[idx, k]
    icpt = my[k] - c @ m[idx]
    kept[A[k]] = dict(idx=[int(i) for i in idx], coef=c.tolist(), icpt=float(icpt))
    terms = " + ".join(f"{v:.2f} × ({names[i]})" for i, v in zip(idx, c)).replace("+ -", "- ")
    rules.append(f"- final {A[k]} = {icpt:.2f} + {terms}")
json.dump(dict(cuts=cuts.tolist(), rules=kept), open(f"{OUTD}/rule.json", "w"), indent=1)

band = "; ".join([f"level 1: up to {cuts[0]} words"] + [f"level {i + 2}: {cuts[i] + 1}-{cuts[i + 1]} words"
                                                         for i in range(3)] + [f"level 5: more than {cuts[3]} words"])
CAL = f"""## Final ratings
First rate every attribute as described above; these are your initial ratings. Then compute the final rating of each attribute with the rules below, which were learned from human expert ratings (they correct how much each attribute should depend on the other attributes and on the length of the item). The word count of the item is given at the top of the item. Length level: {band}.
{chr(10).join(rules)}
Compute each final rating exactly (decimals are fine)."""
FMT = ("## Output format:\nThink briefly, then end your reply with exactly one line of JSON containing, for every attribute, "
       "its initial rating (integer, key \"initial_<attribute>\") and its final rating (number, key \"<attribute>\").")
SYSTEM = ds.v0.strip() + "\n\n" + CAL + "\n\n" + FMT
open(f"{OUTD}/prompt.txt", "w").write(SYSTEM)
print(SYSTEM[-1800:], flush=True)

rows = ds.rows["test"]
df = ds.judge_msgs(lambda r: [dict(role="system", content=SYSTEM),
                              dict(role="user", content=f"(Word count: {words(r['id'])})\n\n" + r["user"])], rows,
                   a.workers, strict=False)
sc = df["reason"].fillna("").map(gem.parse_scores)
for x in A:
    df["initial_" + x] = sc.map(lambda d: d.get("initial_" + x, np.nan))
df.to_csv(f"{OUTD}/scores_test.csv", index=False)

H = ds.human(df.id)
fin = df[A].to_numpy(float); ini = df[["initial_" + x for x in A]].to_numpy(float)
lv = np.array([level(words(i)) for i in df.id], float)
G = np.c_[ini, lv]
ours = np.stack([kept[x]["icpt"] + G[:, kept[x]["idx"]] @ np.array(kept[x]["coef"]) for x in A], 1)
r = lambda P: 100 * np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)])
print(f"\n[{a.ds}] unparsed final {np.isnan(fin).any(1).mean():.1%}")
print(f"[{a.ds}] test Spearman x100: judge final {r(fin):.1f} | judge initial {r(ini):.1f} | rule applied by us "
      f"{r(ours):.1f} | fidelity (final vs ours) {100 * np.nanmean([sp(fin[:, k], ours[:, k]) for k in range(K)]):.1f}",
      flush=True)

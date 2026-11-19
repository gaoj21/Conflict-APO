"""ShAD written as text rules the judge can follow, no decoder and no numbers to compute.

From ShAD fitted on v0's fit + sel scores (attribute scores + log length):
  order:  attributes are rated in an order where an attribute comes after the attributes it borrows from most, and each
          rule refers to ratings the judge has already written ("move it toward the grammar rating you gave");
  rules:  per attribute, which earlier ratings to lean on and how much (in words), and the length effect in rating
          points between word-count bands (the word count is given with every item); half points are allowed;
  demos (--demos k): the k fit items whose ratings ShAD corrects most, with their human ratings, as worked examples.
Single shot (no search, so no selection noise); evaluated on test, raw ratings only.
Ablations: --norules (demos only), --pick random (random fit items as demos instead of ShAD's largest corrections).
  .venv/bin/python baselines/run_textrules.py <ds> [--demos 0] [--norules] [--pick shad|random]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--demos", type=int, default=0); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--norules", action="store_true"); ap.add_argument("--pick", choices=["shad", "random"], default="shad")
ap.add_argument("--dseed", type=int, default=0)  # which random draw of demos (--pick random)
ap.add_argument("--simple", action="store_true")  # executable rules: length as per-band +/- points, at most 1 cross-attribute reference
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = f"textrules{'S' if a.simple else ''}_d{a.demos}" + ("_norules" if a.norules else "") + (f"_random{a.dseed or ''}" if a.pick == "random" else "")
gem.TAG.v = f"{a.ds}/{METHOD}/seed0"
OUTD = ds.out(METHOD, 0)
A, K = ds.A, len(ds.A)
WC = lambda i: len(ds.text[i].split())

tr = pd.concat([pd.read_csv(f"{gem.OUT}/{a.ds}/v0/seed0/scores_{s}.csv") for s in ("fit", "sel")], ignore_index=True)
X = tr[A].to_numpy(float); X = np.where(np.isnan(X), np.nanmean(X, 0), X)
m, s = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
z = ds.gold.loc[tr.id, "z"].to_numpy(float); zs = (z - z.mean()) / z.std()
F = np.c_[(X - m) / s, zs]
Y = ds.human(tr.id); my, sy = Y.mean(0), Y.std(0)
W = anchored_fit(F, (Y - my) / sy)(np.eye(K + 1))  # (K+1, K) standardized

# rating order: an attribute is rated after the attributes it leans on (greedy: fewest unmet dependencies first)
dep = np.abs(W[:K, :K]).copy(); np.fill_diagonal(dep, 0)
order, left = [], list(range(K))
while left:
    k = min(left, key=lambda k: sum(dep[j, k] for j in left if j != k))
    order.append(k); left.remove(k)

wc = np.array([WC(i) for i in tr.id]); q1, q4 = np.quantile(wc, [.2, .8]).round().astype(int)
dz = zs[wc > q4].mean() - zs[wc < q1].mean()
word = lambda r: ("about as much as" if r >= .8 else "more than half as much as" if r >= .5 else
                  "about a third as much as" if r >= .25 else "a little, much less than")
rules = []
bands = np.quantile(wc, [.2, .4, .6, .8]).round().astype(int)
bz = [zs[(wc > lo) & (wc <= hi)].mean() for lo, hi in zip([-1] + list(bands), list(bands) + [10 ** 9])]  # mean z per band
for pos, k in enumerate(order if a.simple else []):
    own = W[k, k]
    cand = [j for j in order[:pos] if W[j, k] > 0]
    parts = []
    if cand:
        j = max(cand, key=lambda j: W[j, k]); r = W[j, k] / max(abs(own), 1e-6)
        if r >= .15:
            parts.append(f"if it differs from the {A[j]} rating you already gave, move it {'halfway' if r >= .5 else 'a quarter of the way'} toward that rating")
    adj = [W[K, k] * sy[k] * (b - np.mean(bz)) for b in bz]
    adj = [round(2 * v) / 2 for v in adj]  # half points
    if any(v != 0 for v in adj):
        lab = [f"up to {bands[0]} words"] + [f"{bands[i] + 1}-{bands[i + 1]} words" for i in range(3)] + [f"more than {bands[3]} words"]
        parts.append("then adjust for length: " + ", ".join(f"{l}: {'+' if v > 0 else ''}{v:g}" for l, v in zip(lab, adj)))
    rules.append(f"{pos + 1}. {A[k]}: rate its own evidence" + ("; " + "; ".join(parts) if parts else "") + ".")
for pos, k in enumerate([] if a.simple else order):
    own, parts = W[k, k], []
    for j in order[:pos]:
        if abs(W[j, k]) / max(abs(own), 1e-6) >= .15:
            r = abs(W[j, k]) / max(abs(own), 1e-6)
            d = "toward" if W[j, k] > 0 else "away from"
            parts.append(f"move it {d} the {A[j]} rating you already gave, {word(r)} its own evidence")
    pts = W[K, k] * sy[k] * dz
    if abs(pts) >= .2:
        parts.append(f"essays/items over {q4} words get about {abs(pts):.1f} point {'more' if pts > 0 else 'less'} than items under "
                     f"{q1} words with the same evidence (scale linearly in between)")
    rules.append(f"{pos + 1}. {A[k]}: judge its own evidence first" + ("; then " + "; ".join(parts) if parts else "") + ".")
guide = ("## How to rate (learned from expert ratings)\nRate the attributes in the order below and write each rating before "
         "the next; later attributes refer to ratings you already gave. The word count of the item is given at the top.\n"
         + "\n".join(rules))

shots = ""
if a.demos:
    fold = np.arange(len(tr)) % 5; P = np.zeros_like(Y)
    for f in range(5):
        P[fold == f] = anchored_fit(F[fold != f], (Y[fold != f] - my) / sy)(F[fold == f]) * sy + my
    fit_ids = set(r["id"] for r in ds.rows["fit"])
    corr = np.abs(P - X).mean(1); corr[~tr.id.isin(fit_ids).to_numpy()] = -1
    pick = np.argsort(-corr)[:a.demos]
    if a.pick == "random":
        pick = np.random.default_rng(a.dseed).choice(np.where(tr.id.isin(fit_ids).to_numpy())[0], a.demos, replace=False)
    shots = "\n\n## Examples rated by human experts\n" + "\n\n".join(
        f"### Example {n + 1} (Word count: {WC(tr.id[i])})\n{ds.text[tr.id[i]].strip()[:2500]}\n\nExpert ratings: "
        + json.dumps({A[k]: float(Y[i, k]) for k in order}) for n, i in enumerate(pick))

SYSTEM = (ds.v0.strip() + ("" if a.norules else "\n\n" + guide) + shots + "\n\n" + ds.FORMAT +
          "\nHalf points (e.g. 2.5) are allowed when an item falls between two levels."
          + ("" if a.norules else " Write the JSON keys in the rating order above.") + " The word count of the item is given at the top.")
open(f"{OUTD}/prompt.txt", "w").write(SYSTEM)
print(guide, flush=True)

df = ds.judge_msgs(lambda r: [dict(role="system", content=SYSTEM),
                              dict(role="user", content=f"(Word count: {WC(r['id'])})\n\n" + r["user"])], ds.rows["test"],
                   a.workers, strict=False)
df.to_csv(f"{OUTD}/scores_test.csv", index=False)
H = ds.human(df.id)
r = 100 * np.nanmean([sp(H[:, k], df[A[k]].to_numpy(float)) for k in range(K)])
print(f"\n[{a.ds}] {METHOD}: unparsed {df[A].isna().any(axis=1).mean():.1%} | test raw (no decoder) {r:.1f}", flush=True)

"""APO, then ShAD described in plain words by an LLM and added to the final prompt (no numbers, no decoder at inference).

ShAD is fitted once on the final prompt's existing fit + sel scores. What it does beyond reading each attribute from
its own score is described per attribute in words (which other ratings it leans on and how strongly, whether longer
items should get higher or lower ratings and how strongly, how much decoding helps the attribute). The optimizer LLM
turns this into plain-language rating guidance (no numbers, no formulas, no word-count thresholds). C candidates are
judged on sel; the best raw Spearman is kept and judged on test. Same format as run_postrules.py (word count, half
points), so its "base" row is the control.
--combine: the text rules cover only the cross-attribute corrections (length is left out of the description and the
LLM is told not to mention it); the numeric length rules of run_postrules.py are added next to them.
  .venv/bin/python baselines/run_llmrules.py <ds> <v0|textgrad_ccc|opro|mipro> [--cands 3] [--combine]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("base"); ap.add_argument("--cands", type=int, default=3)
ap.add_argument("--workers", type=int, default=16); ap.add_argument("--combine", action="store_true")
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = f"llmrules_{'comb_' if a.combine else ''}{a.base}"
gem.TAG.v = f"{a.ds}/{METHOD}/seed0"
OUTD = ds.out(METHOD, 0)
A, K = ds.A, len(ds.A)
WC = lambda i: len(ds.text[i].split())
src = f"{gem.OUT}/{a.ds}/{'dspy_mipro' if a.base == 'mipro' else a.base}/seed0"
if a.base == "v0":
    prompt = ds.v0.strip()
elif a.base == "mipro":
    prog = json.load(open(f"{src}/program.json"))["predict"]
    shots = "\n\n".join(f"### Example {i + 1}\n{d['item'].strip()}\n\nHuman ratings: "
                        + json.dumps({x: round(float(d[x]), 1) for x in A}) for i, d in enumerate(prog.get("demos", []))
                        if all(x in d for x in A))
    prompt = prog["signature"]["instructions"].strip() + ("\n\n## Examples rated by human experts\n\n" + shots if shots else "")
else:
    prompt = open(f"{src}/final_prompt.txt").read().strip()

tr = ds.keep(pd.concat([pd.read_csv(f"{src}/scores_{s}.csv") for s in ("fit", "sel")], ignore_index=True))
X = tr[A].to_numpy(float); X = np.where(np.isnan(X), np.nanmean(X, 0), X)
z = ds.gold.loc[tr.id, "z"].to_numpy(float); zs = (z - z.mean()) / z.std()
F = np.c_[(X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1), zs]
Y = ds.human(tr.id); Ys = (Y - Y.mean(0)) / Y.std(0)
W = anchored_fit(F, Ys)(np.eye(K + 1))
fold = np.arange(len(tr)) % 5; P = np.zeros_like(Ys)
for f in range(5):
    P[fold == f] = anchored_fit(F[fold != f], Ys[fold != f])(F[fold == f])
raw = [sp(Y[:, k], X[:, k]) for k in range(K)]; dec = [sp(Y[:, k], P[:, k]) for k in range(K)]
strength = lambda r: ("about as strongly as" if r >= .8 else "more than half as strongly as" if r >= .5 else
                      "about a third as strongly as" if r >= .25 else "slightly, much less than")
lines = []
for k in range(K):
    own, parts = W[k, k], []
    for j in np.argsort(-np.abs(W[:, k])):
        if j == k:
            continue
        r = abs(W[j, k]) / max(abs(own), 1e-6)
        if r < .1:
            continue
        if j == K and a.combine:
            continue
        if j == K:
            parts.append(f"longer items should get {'higher' if W[j, k] > 0 else 'lower'} {A[k]} ratings, {strength(r)} "
                         f"the {A[k]} evidence itself counts")
        else:
            parts.append(f"items the judge rates {'higher' if W[j, k] > 0 else 'lower'} on {A[j]} should get higher {A[k]} "
                         f"ratings, {strength(r)} the {A[k]} evidence itself counts")
    lines.append(f"- {A[k]} (agreement with experts {'improves a lot' if dec[k] - raw[k] > .05 else 'improves a little' if dec[k] - raw[k] > .01 else 'barely changes'} "
                 f"after correction): " + ("; ".join(parts) if parts else "no correction needed") + ".")
corr = "\n".join(lines)
open(f"{OUTD}/correction.txt", "w").write(corr)

TASK = f"""An LLM judge rates items with the rubric below. A statistical calibration learned from human expert ratings shows how the judge's ratings should be corrected to match the experts, attribute by attribute:

### Rubric
{prompt}

### Corrections the calibration makes
{corr}

Write rating guidance, in plain words, that makes the judge's own ratings already include these corrections: for each attribute that needs correction, say what to look at and what should raise or lower the rating, including how much length should matter. Use no numbers, no formulas and no word-count thresholds.{" Do not mention length: it is handled separately." if a.combine else ""} Keep it concise (at most 8 bullet points). Return only the guidance between <guidance> and </guidance>."""
LEN = open(f"{gem.OUT}/{a.ds}/postrules_{a.base}/seed0/rules.txt").read().strip() if a.combine else ""
TAIL = "\nHalf points and decimals are allowed. The word count of the item is given at the top."


def judge(g, split, tag):
    f = f"{OUTD}/{tag}_{split}.csv"
    if not os.path.exists(f):
        sysm = (prompt + ("\n\n## Rating guidance (learned from expert ratings)\n" + g if g else "")
                + ("\n\n" + LEN if a.combine and LEN else "") + "\n\n" + ds.FORMAT + TAIL)
        ds.judge_msgs(lambda r: [dict(role="system", content=sysm),
                                 dict(role="user", content=f"(Word count: {WC(r['id'])})\n\n" + r["user"])], ds.rows[split],
                      a.workers, strict=False).to_csv(f, index=False)
    return ds.worst_case(pd.read_csv(f))


rawsp = lambda df: 100 * float(np.nanmean(ds.spearman(df)))
best = None
for c in range(a.cands):
    gf = f"{OUTD}/c{c}_guidance.txt"
    if not os.path.exists(gf):
        txt = gem.chat([dict(role="user", content=TASK)], model=gem.OPTIM, temperature=0.8, max_tokens=2500, cache=False)[0]
        open(gf, "w").write(txt.split("<guidance>", 1)[1].split("</guidance>", 1)[0].strip()
                            if "<guidance>" in txt and "</guidance>" in txt else "")
    g = open(gf).read()
    if not g:
        continue
    v = rawsp(judge(g, "sel", f"c{c}"))
    print(f"  [{a.ds}] {a.base} cand {c}: sel raw {v:.1f}", flush=True)
    if best is None or v > best[0]:
        best = (v, g, c)
open(f"{OUTD}/guidance.txt", "w").write(best[1])
te = judge(best[1], "test", f"c{best[2]}")
base = json.load(open(f"{gem.OUT}/{a.ds}/postrules_{a.base}/seed0/result.json"))
res = dict(orig=base["orig"], base=base["base"], numeric_rules=base["rules"], text_rules=rawsp(te))
if a.combine:
    res["text_only"] = json.load(open(f"{gem.OUT}/{a.ds}/llmrules_{a.base}/seed0/result.json"))["text_rules"]
json.dump(res, open(f"{OUTD}/result.json", "w"))
print(f"[{a.ds}] {a.base}: original {res['orig']:.1f} | + format {res['base']:.1f} | + numeric length rules "
      f"{res['numeric_rules']:.1f} | + LLM text rules {res.get('text_only', res['text_rules']):.1f}"
      + (f" | + combined {res['text_rules']:.1f}" if a.combine else ""), flush=True)

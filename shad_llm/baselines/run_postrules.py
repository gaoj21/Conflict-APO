"""APO, then ShAD written into the final prompt as text rules (one step, no loop, no decoder at inference).

Take the final prompt of an APO method; fit ShAD once on that prompt's existing fit + sel scores (attribute scores +
log length; no new calls); turn its length weights into per-attribute adjustments in rating points per word-count band;
append them to the prompt. The judge sees the word count of every item; half points are allowed. Test is judged
twice: the final prompt with the same format changes but without rules ("base"), and with the rules ("+rules"), so
the gain of the rules is separated from that of the format.
  .venv/bin/python baselines/run_postrules.py <ds> <v0|textgrad_ccc|opro|mipro>
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("base"); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = f"postrules_{a.base}"
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

# ShAD on the final prompt's own fit + sel scores
tr = ds.keep(pd.concat([pd.read_csv(f"{src}/scores_{s}.csv") for s in ("fit", "sel")], ignore_index=True))
X = tr[A].to_numpy(float); X = np.where(np.isnan(X), np.nanmean(X, 0), X)
z = ds.gold.loc[tr.id, "z"].to_numpy(float); zs = (z - z.mean()) / z.std()
F = np.c_[(X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1), zs]
Y = ds.human(tr.id); sy = Y.std(0)
W = anchored_fit(F, (Y - Y.mean(0)) / sy)(np.eye(K + 1))
wc = np.array([WC(i) for i in tr.id]); cuts = np.quantile(wc, [.2, .4, .6, .8]).round().astype(int)
b = np.searchsorted(cuts, wc, side="left"); bz = np.array([zs[b == j].mean() for j in range(5)])
lab = [f"up to {cuts[0]} words"] + [f"{cuts[i] + 1}-{cuts[i + 1]} words" for i in range(3)] + [f"more than {cuts[3]} words"]
rules = []
for k in range(K):
    adj = W[K, k] * sy[k] * (bz - bz.mean())
    if np.any(np.abs(adj) >= .05):
        rules.append(f"- {A[k]}: after rating it from the evidence, adjust for length: "
                     + ", ".join(f"{l}: {'+' if v > 0 else ''}{v:.1f}" for l, v in zip(lab, adj)) + ".")
guide = ("## Length adjustment (learned from expert ratings)\nThe word count of the item is given at the top. "
         "Experts weigh length differently for each attribute; apply these adjustments after rating each attribute.\n"
         + "\n".join(rules)) if rules else ""
open(f"{OUTD}/rules.txt", "w").write(guide)
print(guide, flush=True)

TAIL = "\nHalf points and decimals are allowed. The word count of the item is given at the top."
H = ds.human([r["id"] for r in ds.rows["test"]])
res = {}
for name, g in (("base", ""), ("rules", guide)):
    f = f"{OUTD}/scores_test_{name}.csv"
    if not os.path.exists(f):
        sysm = prompt + ("\n\n" + g if g else "") + "\n\n" + ds.FORMAT + TAIL
        ds.judge_msgs(lambda r: [dict(role="system", content=sysm),
                                 dict(role="user", content=f"(Word count: {WC(r['id'])})\n\n" + r["user"])], ds.rows["test"],
                      a.workers, strict=False).to_csv(f, index=False)
    df = ds.worst_case(pd.read_csv(f)).set_index("id").loc[[r["id"] for r in ds.rows["test"]]]
    res[name] = 100 * np.nanmean([sp(H[:, k], df[A[k]].to_numpy(float)) for k in range(K)])
orig = pd.read_csv(f"{src}/scores_test.csv").set_index("id").loc[[r["id"] for r in ds.rows["test"]]]
res["orig"] = 100 * np.nanmean([sp(H[:, k], orig[A[k]].to_numpy(float)) for k in range(K)])
json.dump(res, open(f"{OUTD}/result.json", "w"))
print(f"[{a.ds}] {a.base}: original {res['orig']:.1f} | + format (half, word count) {res['base']:.1f} | "
      f"+ ShAD length rules {res['rules']:.1f}", flush=True)

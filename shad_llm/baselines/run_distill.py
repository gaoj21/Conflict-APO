"""Decoder-guided APO: ShAD as the teacher of the judge prompt.

Each round fits ShAD on the current prompt's fit + sel scores and describes to the optimizer what the decoder
corrects: (1) its weights in words (how much each attribute is read from its own score, the other attributes' scores
and length), (2) the fit items it moves most up and down per attribute (cross-fitted), with raw, decoded and human
ratings. The optimizer rewrites the rubric so that the judge's own ratings already make these corrections. The new
prompt judges fit / sel / test; we report raw and ShAD test Spearman of every round.
  .venv/bin/python baselines/run_distill.py <ds> [--rounds 3]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = "distill"
gem.TAG.v = f"{a.ds}/{METHOD}/seed0"
OUTD = ds.out(METHOD, 0)
A, K = ds.A, len(ds.A)


def load(d):
    S = {s: pd.read_csv(f"{d}/scores_{s}.csv") for s in ("fit", "sel", "test")}
    return pd.concat([S["fit"], S["sel"]], ignore_index=True), S["test"]


def design(tr, te):
    Xtr, Xte = tr[A].to_numpy(float), te[A].to_numpy(float)
    mu = np.nanmean(Xtr, 0); Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    m, s = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1)
    z = ds.gold["z"]; ztr, zte = z.loc[tr.id].to_numpy(float), z.loc[te.id].to_numpy(float)
    F = lambda X, zz: np.c_[(X - m) / s, (zz - ztr.mean()) / ztr.std()]
    return F(Xtr, ztr), F(Xte, zte), Xtr, Xte


def evaluate(d):
    tr, te = load(d)
    Ftr, Fte, Xtr, Xte = design(tr, te)
    Y = ds.human(tr.id); f = anchored_fit(Ftr, (Y - Y.mean(0)) / Y.std(0))
    H = ds.human(te.id); P = f(Fte)
    return dict(raw=100 * np.nanmean([sp(H[:, k], Xte[:, k]) for k in range(K)]),
                shad=100 * np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)]))


def teach(d, prompt):
    tr, _ = load(d)
    fit_ids = set(r["id"] for r in ds.rows["fit"])
    Ftr, _, Xtr, _ = design(tr, tr)
    Y = ds.human(tr.id); mu, sd = Y.mean(0), Y.std(0)
    W = anchored_fit(Ftr, (Y - mu) / sd)(np.eye(Ftr.shape[1]))  # (features, K) standardized weights
    names = A + ["length"]
    lines = ["How the calibration reads each attribute (standardized weights; larger magnitude = more influence):"]
    for k in range(K):
        order = np.argsort(-np.abs(W[:, k]))[:4]
        lines.append(f"- {A[k]}: " + ", ".join(f"{'judge ' + names[i] + ' score' if i < K else 'response length'} "
                                                 f"{W[i, k]:+.2f}" for i in order))
    # cross-fitted decoded ratings on the labelled items, then the largest corrections on fit items
    fold = np.arange(len(tr)) % 5
    P = np.zeros_like(Y)
    for f in range(5):
        a_, b_ = fold != f, fold == f
        P[b_] = anchored_fit(Ftr[a_], (Y[a_] - mu) / sd)(Ftr[b_]) * sd + mu
    D = P - Xtr
    on_fit = tr.id.isin(fit_ids).to_numpy()
    lines.append("\nItems the calibration corrects most (raw = the judge's rating, calibrated = after correction, "
                 "human = experts):")
    for k in np.argsort(-np.abs(D[on_fit]).mean(0))[:3]:
        idx = np.where(on_fit)[0]
        for tag, sel in (("raised", idx[np.argsort(-D[idx, k])[:2]]), ("lowered", idx[np.argsort(D[idx, k])[:2]])):
            for j in sel:
                txt = ds.text[tr.id.iloc[j]].strip()
                txt = txt[:1200] + (" [...]" if len(txt) > 1200 else "")
                lines.append(f"--- {A[k]} {tag}: raw {Xtr[j, k]:.0f}, calibrated {P[j, k]:.1f}, human {Y[j, k]:.0f}"
                             f" ({len(txt.split())} words)\n{txt}")
    fb = "\n".join(lines)
    msg = f"""Below is the rubric prompt of an LLM judge. A calibration learned from human expert ratings corrects the judge's ratings, and the corrections show how the judge misreads the attributes: which evidence it ignores, which attributes it confuses with each other, and how it misjudges length.

### Current rubric prompt
{prompt}

### What the calibration corrects
{fb}

Rewrite the rubric so that the judge's own ratings already make these corrections: for each attribute, state what evidence should raise or lower the rating, separate attributes the judge confuses, and say how length should matter where it should. Keep the attribute names and the rating scale; do not write an output-format section (it is appended separately). Return only the new rubric between <prompt> and </prompt>."""
    for _ in range(3):
        txt = gem.chat([dict(role="user", content=msg)], model=gem.OPTIM, temperature=0.7, max_tokens=4000, cache=False)[0]
        if "<prompt>" in txt and "</prompt>" in txt:
            return txt.split("<prompt>", 1)[1].split("</prompt>", 1)[0].strip(), fb
    raise gem.Abort("optimizer gave no prompt")


prev_d, prompt = f"{gem.OUT}/{a.ds}/v0/seed0", ds.v0
log = [dict(round=0, **evaluate(prev_d))]
print(f"[{a.ds}] round 0 (v0): raw {log[0]['raw']:.1f} ShAD {log[0]['shad']:.1f}", flush=True)
for t in range(1, a.rounds + 1):
    d = f"{OUTD}/round{t}"; os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/prompt.txt"):
        new, fb = teach(prev_d, prompt)
        open(f"{d}/prompt.txt", "w").write(new); open(f"{d}/teacher.txt", "w").write(fb)
    prompt = open(f"{d}/prompt.txt").read()
    for s in ("fit", "sel", "test"):
        if not os.path.exists(f"{d}/scores_{s}.csv"):
            ds.judge(prompt, ds.rows[s], a.workers).to_csv(f"{d}/scores_{s}.csv", index=False)
    log.append(dict(round=t, **evaluate(d)))
    print(f"[{a.ds}] round {t}: raw {log[-1]['raw']:.1f} ShAD {log[-1]['shad']:.1f}", flush=True)
    prev_d = d
json.dump(log, open(f"{OUTD}/log.json", "w"), indent=1)
print(a.ds, METHOD, "done", flush=True)

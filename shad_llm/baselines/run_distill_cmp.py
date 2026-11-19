"""Key experiment: can ShAD be distilled into a frozen APO prompt, and does ShAD guidance beat plain error feedback?

Frozen start prompt (default: TextGrad CCC's final prompt). Data roles: fit builds the teacher and the rules; sel only
selects among candidates; test is judged once per arm for the selected candidate.
  teacher: ShAD fitted on the start prompt's FIT scores, cross-fitted (5 folds) so every fit item gets an out-of-fold
           teacher rating t_i (human scale).
Arms (same format for all: word count shown, half points; same budget: C candidates, <= 8 bullets each):
  control  start prompt, no guidance
  coef     ShAD's cross-attribute and length weights described in words -> LLM writes guidance (current method)
  teacher  contrastive pairs where the judge's ranking and the teacher's disagree most -> LLM writes evidence rules
  human    the same, with human ratings in place of the teacher (error-feedback baseline, AutoCalibrate-like)
Selection on sel by Q(c) = mean_b drho_b(c) - gamma * SD_b drho_b(c) (paired bootstrap over sel items, drho = change of
mean Spearman vs the control prompt; the control itself scores 0 and is always a candidate; gamma fixed in advance).
  .venv/bin/python baselines/run_distill_cmp.py <ds> [--base textgrad_ccc] [--cands 3] [--pairs 2] [--gamma 1]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--base", default="textgrad_ccc"); ap.add_argument("--cands", type=int, default=3)
ap.add_argument("--pairs", type=int, default=2); ap.add_argument("--gamma", type=float, default=1.0)
ap.add_argument("--B", type=int, default=200); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = f"distillcmp_{a.base}"
gem.TAG.v = f"{a.ds}/{METHOD}/seed0"
OUTD = ds.out(METHOD, 0)
A, K = ds.A, len(ds.A)
WC = lambda i: len(ds.text[i].split())
src = f"{gem.OUT}/{a.ds}/{a.base}/seed0"
prompt = open(f"{src}/final_prompt.txt").read().strip()
TAIL = "\nHalf points and decimals are allowed. The word count of the item is given at the top."

# ---------------------------------------------------------------- teacher (fit only, cross-fitted)
fit = ds.worst_case(pd.read_csv(f"{src}/scores_fit.csv")).set_index("id").loc[[r["id"] for r in ds.rows["fit"]]].reset_index()
X = fit[A].to_numpy(float); Y = ds.human(fit.id)
z = np.log([WC(i) for i in fit.id]); zs = (z - z.mean()) / z.std()
F = np.c_[(X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1), zs]
my, sy = Y.mean(0), Y.std(0)
fold = np.random.default_rng(0).permutation(len(fit)) % 5
T = np.zeros_like(Y)
for f in range(5):
    T[fold == f] = anchored_fit(F[fold != f], (Y[fold != f] - my) / sy)(F[fold == f]) * sy + my
W = anchored_fit(F, (Y - my) / sy)(np.eye(K + 1))  # for the coefficient description


def describe_coef():
    word = lambda r: ("about as strongly as" if r >= .8 else "more than half as strongly as" if r >= .5 else
                      "about a third as strongly as" if r >= .25 else "slightly, much less than")
    out = []
    for k in range(K):
        parts = []
        for j in np.argsort(-np.abs(W[:, k])):
            r = abs(W[j, k]) / max(abs(W[k, k]), 1e-6)
            if j == k or r < .1:
                continue
            if j == K:
                parts.append(f"longer items should get {'higher' if W[j, k] > 0 else 'lower'} {A[k]} ratings, {word(r)} the {A[k]} evidence itself counts")
            else:
                parts.append(f"items the judge rates {'higher' if W[j, k] > 0 else 'lower'} on {A[j]} should get higher {A[k]} ratings, {word(r)} the {A[k]} evidence itself counts")
        out.append(f"- {A[k]}: " + ("; ".join(parts) if parts else "no correction needed") + ".")
    return "\n".join(out)


def pairs(target):
    """Per attribute, fit-item pairs whose judge ratings are close but whose target order is the reverse of (or much
    sharper than) the judge's: rank by target gap minus judge gap in the judge's direction."""
    out = []
    for k in range(K):
        s, t = X[:, k], target[:, k]
        ii, jj = np.triu_indices(len(s), 1)
        ds_, dt = s[ii] - s[jj], t[ii] - t[jj]
        score = np.abs(dt) - np.sign(dt) * ds_  # large target gap, judge gap small or in the wrong direction
        for n in np.argsort(-score)[:a.pairs]:
            i, j = (ii[n], jj[n]) if dt[n] > 0 else (jj[n], ii[n])  # i should be rated higher than j
            out.append((k, i, j))
    return out


def show_pairs(pp, who):
    blocks = []
    for k, i, j in pp:
        ti, tj = ds.text[fit.id[i]].strip()[:1200], ds.text[fit.id[j]].strip()[:1200]
        blocks.append(f"### Attribute: {A[k]}\nText A:\n{ti}\n\nText B:\n{tj}\n\nThe judge rated {A[k]}: A {X[i, k]:g}, B {X[j, k]:g}. "
                      f"{who} A should be rated clearly higher than B on {A[k]}.")
    return "\n\n".join(blocks)


INSTR = ("Write rating guidance for the judge: at most 8 bullet points, in plain words, no numbers or formulas. Each bullet "
         "should state observable evidence that should (or should not) count for a specific attribute, so that the judge's "
         "own ratings get these cases right. Return only the guidance between <guidance> and </guidance>.")
TASKS = {
    "coef": f"An LLM judge rates items with the rubric below. A statistical calibration learned from human expert ratings "
            f"shows how its ratings should be corrected, attribute by attribute:\n\n### Rubric\n{prompt}\n\n### Corrections\n"
            f"{describe_coef()}\n\n{INSTR}",
    "teacher": f"An LLM judge rates items with the rubric below. For each pair of texts, a calibrated model of expert ratings "
               f"says the judge ranked them wrongly. Find the observable differences between the texts that explain the "
               f"correct order.\n\n### Rubric\n{prompt}\n\n{show_pairs(pairs(T), 'The calibrated expert model says')}\n\n{INSTR}",
    "human": f"An LLM judge rates items with the rubric below. For each pair of texts, human experts disagree with the judge's "
             f"order. Find the observable differences between the texts that explain the experts' order.\n\n### Rubric\n"
             f"{prompt}\n\n{show_pairs(pairs(Y), 'Human experts say')}\n\n{INSTR}",
}


def judge(g, split, tag):
    f = f"{OUTD}/{tag}_{split}.csv"
    if not os.path.exists(f):
        sysm = prompt + ("\n\n## Rating guidance (learned from expert ratings)\n" + g if g else "") + "\n\n" + ds.FORMAT + TAIL
        ds.judge_msgs(lambda r: [dict(role="system", content=sysm),
                                 dict(role="user", content=f"(Word count: {WC(r['id'])})\n\n" + r["user"])], ds.rows[split],
                      a.workers, strict=False).to_csv(f, index=False)
    df = ds.worst_case(pd.read_csv(f)).set_index("id")
    return df.loc[[r["id"] for r in ds.rows[split]], A].to_numpy(float)


def spear(H, P):
    return np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)])


Hs = ds.human([r["id"] for r in ds.rows["sel"]]); Ht = ds.human([r["id"] for r in ds.rows["test"]])
P0 = judge("", "sel", "control")
rng = np.random.default_rng(0); BOOT = [rng.integers(0, len(Hs), len(Hs)) for _ in range(a.B)]


def Q(P):
    d = np.array([spear(Hs[b], P[b]) - spear(Hs[b], P0[b]) for b in BOOT])
    return float(d.mean() - a.gamma * d.std()), float(spear(Hs, P) - spear(Hs, P0))


res = dict(control=dict(test=100 * spear(Ht, judge("", "test", "control"))))
for arm, task in TASKS.items():
    best = ("", 0.0, "control")  # the control prompt is always a candidate (Q = 0)
    for c in range(a.cands):
        gf = f"{OUTD}/{arm}_c{c}_guidance.txt"
        if not os.path.exists(gf):
            txt = gem.chat([dict(role="user", content=task)], model=gem.OPTIM, temperature=0.8, max_tokens=2500, cache=False)[0]
            open(gf, "w").write(txt.split("<guidance>", 1)[1].split("</guidance>", 1)[0].strip()
                                if "<guidance>" in txt and "</guidance>" in txt else "")
        g = open(gf).read()
        if not g:
            continue
        q, d = Q(judge(g, "sel", f"{arm}_c{c}"))
        print(f"  [{a.ds}] {arm} cand {c}: Q {100 * q:+.2f}  sel drho {100 * d:+.2f}", flush=True)
        if q > best[1]:
            best = (g, q, f"{arm}_c{c}")
    test = 100 * spear(Ht, judge(best[0], "test", best[2])) if best[0] else res["control"]["test"]
    res[arm] = dict(selected=best[2], Q=100 * best[1], test=test)
    if best[0]:
        open(f"{OUTD}/{arm}_selected.txt", "w").write(best[0])
# external ShAD reference (fit only, same teacher fit), applied to the start prompt's test scores
te = ds.worst_case(pd.read_csv(f"{src}/scores_test.csv")).set_index("id").loc[[r["id"] for r in ds.rows["test"]]]
zt = np.log([WC(i) for i in te.index]); Xt = te[A].to_numpy(float)
Ft = np.c_[(Xt - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1), (zt - z.mean()) / z.std()]
res["shad_decoder_ref"] = dict(test=100 * spear(Ht, anchored_fit(F, (Y - my) / sy)(Ft)))
res["start_prompt_raw"] = dict(test=100 * spear(Ht, Xt))
json.dump(res, open(f"{OUTD}/result.json", "w"), indent=1)
print(f"[{a.ds}] {a.base}: " + " | ".join(f"{k} {v['test']:.1f}" + (f" ({v['selected']})" if 'selected' in v else "")
                                       for k, v in res.items()), flush=True)

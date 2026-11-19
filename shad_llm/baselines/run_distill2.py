"""Decoder-guided APO, v2: distil ShAD into the judge prompt as appended calibration notes.

v0 is kept unchanged; the optimizer writes a short "Calibration notes" section (concrete rules) from what ShAD
corrects (its weights in words and the items it moves most). Each round proposes several candidate notes; each is
judged on sel and the one with the highest raw Spearman on sel replaces the current notes only if it beats them.
The accepted prompt then judges fit and test, ShAD is refitted, and the next round teaches from it.
The teacher text is the same as in run_distill.py (v1, which rewrote the whole rubric without selection).
  .venv/bin/python baselines/run_distill2.py <ds> [--rounds 3] [--cands 4]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--cands", type=int, default=4); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = "distill2"
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


def teacher_text(d):
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
    return "\n".join(lines)


MSG = """Below is the rubric prompt of an LLM judge, followed by its current calibration notes. A calibration learned from human expert ratings corrects the judge's ratings; the corrections show where the judge misreads the attributes.

### Rubric prompt (fixed)
{rubric}

### Current calibration notes
{notes}

### What the calibration corrects
{fb}

Write new calibration notes: at most 8 short, concrete rules that make the judge's own ratings agree with the corrected ratings, e.g. what evidence should raise or lower a rating, how to keep two attributes apart, how harsh or lenient to be on each attribute, and how length should matter where it should. Each rule must be something the judge can check in the item. Return only the notes as a bullet list between <notes> and </notes>."""


def full(notes):
    return ds.v0 + ("\n\n## Calibration notes\n" + notes if notes else "")


def sel_raw(prompt, tag):
    f = f"{OUTD}/{tag}_sel.csv"
    if not os.path.exists(f):
        ds.judge(prompt, ds.rows["sel"], a.workers, strict=False).to_csv(f, index=False)
    df = ds.worst_case(pd.read_csv(f))
    return float(np.nanmean(ds.spearman(df))), f


notes, prev_d = "", f"{gem.OUT}/{a.ds}/v0/seed0"
cur, _ = sel_raw(full(""), "r0")
log = [dict(round=0, sel_raw=100 * cur, **evaluate(prev_d))]
print(f"[{a.ds}] round 0 (v0): sel raw {100 * cur:.1f} | test raw {log[0]['raw']:.1f} ShAD {log[0]['shad']:.1f}", flush=True)
for t in range(1, a.rounds + 1):
    fb = teacher_text(prev_d)
    open(f"{OUTD}/teacher_r{t}.txt", "w").write(fb)
    best = None
    for c in range(a.cands):
        cf = f"{OUTD}/r{t}c{c}_notes.txt"
        if not os.path.exists(cf):
            txt = gem.chat([dict(role="user", content=MSG.format(rubric=ds.v0, notes=notes or "(none)", fb=fb))],
                           model=gem.OPTIM, temperature=0.9, max_tokens=2000, cache=False)[0]
            m = txt.split("<notes>", 1)[1].split("</notes>", 1)[0].strip() if "<notes>" in txt and "</notes>" in txt else ""
            open(cf, "w").write(m)
        cn = open(cf).read()
        if not cn:
            continue
        v, f = sel_raw(full(cn), f"r{t}c{c}")
        print(f"  [{a.ds}] round {t} cand {c}: sel raw {100 * v:.1f}", flush=True)
        if best is None or v > best[0]:
            best = (v, cn, f)
    if best is not None and best[0] > cur:
        cur, notes = best[0], best[1]
        d = f"{OUTD}/round{t}"; os.makedirs(d, exist_ok=True)
        open(f"{d}/notes.txt", "w").write(notes)
        pd.read_csv(best[2]).to_csv(f"{d}/scores_sel.csv", index=False)
        for s in ("fit", "test"):
            if not os.path.exists(f"{d}/scores_{s}.csv"):
                ds.judge(full(notes), ds.rows[s], a.workers).to_csv(f"{d}/scores_{s}.csv", index=False)
        prev_d = d
    log.append(dict(round=t, sel_raw=100 * cur, kept=prev_d.endswith(f"round{t}"), **evaluate(prev_d)))
    print(f"[{a.ds}] round {t}: {'keep' if log[-1]['kept'] else 'no better candidate'} | sel raw {100 * cur:.1f} | "
          f"test raw {log[-1]['raw']:.1f} ShAD {log[-1]['shad']:.1f}", flush=True)
json.dump(log, open(f"{OUTD}/log.json", "w"), indent=1)
print(a.ds, METHOD, "done", flush=True)

"""ShAD as the gradient of APO: iteratively move what the decoder corrects into the prompt (no decoder at inference).

Round t: fit ShAD on the current prompt's fit + sel scores; read what it still corrects, i.e. how far its weights are
from the anchor (each attribute read from its own score only): which other attributes' scores and how much length it
adds, relative to the attribute's own score, plus the per-attribute gain of decoding (cross-fitted). This remaining
correction is described in words and given to the optimizer, which revises the prompt's "rating guidance" section so
that the judge's own ratings make it. C candidates are judged on sel; the best raw Spearman (no decoder) is kept if it
beats the current prompt. The kept prompt then judges fit and test, ShAD is refitted, and the next round starts.
As the guidance absorbs the corrections, the decoder approaches its anchor and the raw judge approaches the decoded one.
--v2: (i) selection on one half of sel and confirmation on the other: the candidate that is best on half A replaces
the current prompt only if it also beats it on half B (no threshold); (ii) the word count is given with every item and
the length correction is stated in rating points between word-count bands, so the judge can apply it.
--init mipro: start from the MIPROv2 judge (its instruction and demos written out as a text prompt), so APO first finds
the content and the ShAD rules then add what the decoder still corrects. --fitsel: the two selection halves are drawn
from fit + sel (the optimizer only sees the aggregated correction, never items), halving the selection noise.
  .venv/bin/python baselines/run_shadgrad.py <ds> [--rounds 4] [--cands 3] [--v2] [--init mipro] [--fitsel]
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--cands", type=int, default=3)
ap.add_argument("--workers", type=int, default=16); ap.add_argument("--v2", action="store_true")
ap.add_argument("--init", choices=["v0", "mipro"], default="v0"); ap.add_argument("--fitsel", action="store_true")
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = ("shadgrad2" if a.v2 else "shadgrad") + ("" if a.init == "v0" else "_" + a.init) + ("_fitsel" if a.fitsel else "")


def base_prompt():
    if a.init == "v0":
        return ds.v0.strip()
    prog = json.load(open(f"{gem.OUT}/{a.ds}/dspy_{a.init}/seed0/program.json"))["predict"]
    shots = "\n\n".join(f"### Example {i + 1}\n{d['item'].strip()}\n\nHuman ratings: "
                         + json.dumps({x: round(float(d[x]), 1) for x in ds.A if x in d})
                         for i, d in enumerate(prog.get("demos", [])) if all(x in d for x in ds.A))
    return prog["signature"]["instructions"].strip() + ("\n\n## Examples rated by human experts\n\n" + shots if shots else "")


BASE = base_prompt()
gem.TAG.v = f"{a.ds}/{METHOD}/seed0"
OUTD = ds.out(METHOD, 0)
A, K = ds.A, len(ds.A)
HALF = "\nHalf points (e.g. 2.5) are allowed when an item falls between two levels."


WC = lambda i: len(ds.text[i].split())
SEL_B = set(r["id"] for i, r in enumerate(ds.rows["sel"]) if i % 2)  # confirmation half (v2)
ALL_B = set(r["id"] for i, r in enumerate(ds.rows["fit"] + ds.rows["sel"]) if i % 2)  # with --fitsel


def user(r):
    return (f"(Word count: {WC(r['id'])})\n\n" if a.v2 else "") + r["user"]


def system(guidance):
    if not guidance and not a.v2:
        return BASE + "\n\n" + ds.FORMAT  # = the v0 judge (cache hit)
    return BASE + "\n\n## Rating guidance (learned from expert ratings)\n" + guidance + "\n\n" + ds.FORMAT + HALF


def judge(guidance, split, tag):
    f = f"{OUTD}/{tag}_{split}.csv"
    if not os.path.exists(f):
        sysm = system(guidance)
        ds.judge_msgs(lambda r: [dict(role="system", content=sysm), dict(role="user", content=user(r))], ds.rows[split],
                      a.workers, strict=False).to_csv(f, index=False)
    return ds.worst_case(pd.read_csv(f))


def raw_sp(df, half=None):
    if half is not None:
        df = df[df.id.isin(ALL_B if a.fitsel else SEL_B) == (half == "B")]
    return 100 * float(np.nanmean(ds.spearman(df)))


def features(tr, te):
    Xtr, Xte = tr[A].to_numpy(float), te[A].to_numpy(float)
    m, s = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1)
    z = ds.gold["z"]; ztr, zte = z.loc[tr.id].to_numpy(float), z.loc[te.id].to_numpy(float)
    F = lambda X, zz: np.c_[(X - m) / s, (zz - ztr.mean()) / ztr.std()]
    return F(Xtr, ztr), F(Xte, zte)


def shad_test(tr, te):
    Ftr, Fte = features(tr, te)
    Y = ds.human(tr.id); P = anchored_fit(Ftr, (Y - Y.mean(0)) / Y.std(0))(Fte)
    H = ds.human(te.id)
    return 100 * float(np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)]))


def correction(tr):
    """Words for what ShAD still does beyond reading each attribute from its own score."""
    Ftr, _ = features(tr, tr)
    Y = ds.human(tr.id); Ys = (Y - Y.mean(0)) / Y.std(0)
    W = anchored_fit(Ftr, Ys)(np.eye(K + 1))
    fold = np.arange(len(tr)) % 5; P = np.zeros_like(Ys)
    for f in range(5):
        P[fold == f] = anchored_fit(Ftr[fold != f], Ys[fold != f])(Ftr[fold == f])
    raw = [sp(Y[:, k], tr[A[k]]) for k in range(K)]; dec = [sp(Y[:, k], P[:, k]) for k in range(K)]
    names = A + ["length"]
    wc = np.array([WC(i) for i in tr.id]); q1, q4 = np.quantile(wc, [.2, .8]).round().astype(int)
    top, bot = wc > q4, wc < q1
    word = lambda r: ("about as much as" if r >= .8 else "more than half as much as" if r >= .5 else
                      "about a third as much as" if r >= .25 else "a little, compared with")
    out = []
    for k in range(K):
        own = W[k, k]
        parts = []
        for j in np.argsort(-np.abs(W[:, k])):
            if j == k or abs(W[j, k]) < .02:
                continue
            r = abs(W[j, k]) / max(abs(own), 1e-6)
            if names[j] == "length" and a.v2:
                pts = W[j, k] * Y[:, k].std() * (Ftr[top, -1].mean() - Ftr[bot, -1].mean())
                parts.append(f"items over {q4} words should get about {abs(pts):.1f} rating points {'more' if pts > 0 else 'less'} {A[k]} "
                             f"than items under {q1} words, beyond what their {A[k]} evidence shows")
            elif names[j] == "length":
                d = "longer" if W[j, k] > 0 else "shorter"
                parts.append(f"{d} items should get a higher {A[k]} rating, weighted {word(r)} the judge's own {A[k]} evidence")
            else:
                d = "higher" if W[j, k] > 0 else "lower"
                parts.append(f"the judge's {names[j]} rating should move the {A[k]} rating ({d} {names[j]} -> {'higher' if d == 'higher' else 'lower'} {A[k]}), weighted {word(r)} its own {A[k]} evidence")
        gap = dec[k] - raw[k]
        out.append(f"- {A[k]} (agreement with experts {100 * raw[k]:.0f} as rated, {100 * dec[k]:.0f} after correction): "
                   + ("; ".join(parts) if parts else "no correction beyond its own evidence") + ".")
    return "\n".join(out), float(np.mean(dec) - np.mean(raw))


TASK = """An LLM judge rates items with the rubric below. {wcnote}A calibration learned from human expert ratings shows how the judge's ratings should still be corrected; the corrections below are what remains AFTER the current rating guidance.

### Rubric (fixed)
{rubric}

### Current rating guidance
{guidance}

### Remaining corrections (per attribute)
{corr}

Revise the rating guidance so that the judge's own ratings already make these corrections. For each attribute that needs correction, write concrete rating behaviour: what evidence to look at, what should raise or lower the rating, how much the other aspects should count, and how length should matter. No formulas or numeric weights; use words such as "almost as much", "about half as much". Keep the parts of the current guidance that still apply. At most 10 bullet points. Return only the guidance between <guidance> and </guidance>."""


def propose(guidance, corr, c):
    f = f"{OUTD}/r{t}c{c}_guidance.txt"
    if not os.path.exists(f):
        note = "The judge sees the word count of every item. " if a.v2 else ""
        txt = gem.chat([dict(role="user", content=TASK.format(wcnote=note, rubric=BASE, guidance=guidance or "(none)", corr=corr))],
                       model=gem.OPTIM, temperature=0.8, max_tokens=2500, cache=False)[0]
        g = txt.split("<guidance>", 1)[1].split("</guidance>", 1)[0].strip() if "<guidance>" in txt and "</guidance>" in txt else ""
        open(f, "w").write(g)
    return open(f).read()


guidance, tag = "", "r0"
cur = {s: judge("", s, "r0") for s in ("fit", "sel", "test")}
best_sel = raw_sp(cur["sel"])
cur_B = raw_sp(pd.concat([cur["fit"], cur["sel"]], ignore_index=True) if a.fitsel else cur["sel"], "B")
log = []


def report(t, kept):
    tr = pd.concat([cur["fit"], cur["sel"]], ignore_index=True)
    rec = dict(round=t, kept=kept, sel_raw=best_sel, test_raw=raw_sp(cur["test"]), test_shad=shad_test(tr, cur["test"]))
    log.append(rec); json.dump(log, open(f"{OUTD}/log.json", "w"), indent=1)
    print(f"[{a.ds}] round {t}: {'keep' if kept else 'no better candidate'} | sel raw {best_sel:.1f} | "
          f"test raw (no decoder) {rec['test_raw']:.1f} | test +ShAD {rec['test_shad']:.1f}", flush=True)


report(0, True)
for t in range(1, a.rounds + 1):
    tr = pd.concat([cur["fit"], cur["sel"]], ignore_index=True)
    corr, gap = correction(tr)
    open(f"{OUTD}/r{t}_correction.txt", "w").write(corr)
    best = None
    for c in range(a.cands):
        g = propose(guidance, corr, c)
        if not g:
            continue
        df = (pd.concat([judge(g, "fit", f"r{t}c{c}"), judge(g, "sel", f"r{t}c{c}")], ignore_index=True) if a.fitsel
              else judge(g, "sel", f"r{t}c{c}"))
        v = raw_sp(df, "A") if a.v2 else raw_sp(df)
        print(f"  [{a.ds}] round {t} cand {c}: sel{' A' if a.v2 else ''} raw {v:.1f}"
              + (f" (B {raw_sp(df, 'B'):.1f}, current B {cur_B:.1f})" if a.v2 else ""), flush=True)
        if best is None or v > best[0]:
            best = (v, g, f"r{t}c{c}", raw_sp(df, "B"), raw_sp(df))
    if a.v2:
        kept = best is not None and best[3] > cur_B  # best on half A, confirmed on half B
    else:
        kept = best is not None and best[0] > best_sel
    if kept:
        guidance, tag = best[1], best[2]
        best_sel, cur_B = best[4], best[3]
        open(f"{OUTD}/guidance_r{t}.txt", "w").write(guidance)
        cur = {s: judge(guidance, s, tag) for s in ("fit", "sel", "test")}
    report(t, kept)
print(a.ds, METHOD, "done", flush=True)

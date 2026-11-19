"""Archive APO: prompt optimization that produces features for ShAD.

The judge is an archive A of up to M prompts (A[0] = v0); ShAD decodes each attribute from the scores of every prompt
in A plus log length. The optimizer does not look for one prompt that is better on every attribute: it is shown what
the current decoder still gets wrong and writes a new prompt that rates the same attributes from another angle. A
candidate enters A if it raises the decoder's cross-validated Spearman on the pool; when A is full, it replaces the
member (never v0) whose swap raises it most. An edit that helps one attribute and hurts another is still useful, since
the decoder reads each attribute from whichever prompt carries its information.

Pool: nf items of fit + ns items of sel, judged by every prompt that is tried. Score of an archive: mean over attributes
of the 5-fold (within-fold) CV Spearman of ShAD fitted on the other folds. Feedback items come from the fit part only.
Final: every prompt in A judges fit, sel and test (v0's scores are reused from the v0 run); evaluate with
scripts/eval_archive.py.
--init mipro: A[0] is the DSPy MIPROv2 judge (its scores are reused, no calls), so the archive starts from the best
single-prompt APO result and only adds what the decoder still misses.
--select holdout: a candidate is scored by the Spearman on the pool's sel part of the decoder fitted on its fit part
(the feedback only shows fit items), instead of the CV Spearman over the whole pool.
  .venv/bin/python baselines/run_archive.py <ds> [--steps 12] [--M 3] [--seed 0] [--init v0|mipro] [--select cv|holdout]
"""
import argparse, json, os, random, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
from evaluate import anchored_fit, sp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=12); ap.add_argument("--M", type=int, default=3)
ap.add_argument("--nf", type=int, default=200); ap.add_argument("--ns", type=int, default=200)
ap.add_argument("--nshow", type=int, default=3); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--init", choices=["v0", "mipro"], default="v0"); ap.add_argument("--select", choices=["cv", "holdout"], default="cv")
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = f"archive_M{a.M}" + ("" if a.init == "v0" else f"_{a.init}") + ("" if a.select == "cv" else f"_{a.select}")
gem.TAG.v = f"{a.ds}/{METHOD}/seed{a.seed}"
OUTD = ds.out(METHOD, a.seed)
A, K = ds.A, len(ds.A)
POOL = ds.subset("fit", a.nf, seed=a.seed) + ds.subset("sel", a.ns, seed=a.seed)
PIDS = [r["id"] for r in POOL]
IS_FIT = np.array([i < a.nf for i in range(len(POOL))])
Z = ds.gold.loc[PIDS, "z"].to_numpy(float)
H = ds.human(PIDS)
FOLD = np.random.default_rng(a.seed).permutation(len(POOL)) % 5


def pool_scores(prompt):
    df = ds.worst_case(ds.judge(prompt, POOL, a.workers, strict=False)).set_index("id").loc[PIDS]
    return df[A].to_numpy(float)


def features(S_list, idx_tr, idx_te):
    X = np.hstack(S_list)
    m, s = X[idx_tr].mean(0), np.where(X[idx_tr].std(0) > 0, X[idx_tr].std(0), 1)
    z = (Z - Z[idx_tr].mean()) / Z[idx_tr].std()
    F = np.c_[(X - m) / s, z]  # v0's own scores come first, so ShAD's anchor is v0's own score
    return F[idx_tr], F[idx_te]


def holdout(S_list):
    """Spearman on the pool's sel part of the decoder fitted on its fit part (selection score with --select holdout)."""
    tr, te = IS_FIT, ~IS_FIT
    Ftr, Fte = features(S_list, tr, te)
    Y = H[tr]
    P = anchored_fit(Ftr, (Y - Y.mean(0)) / Y.std(0))(Fte)
    return np.array([sp(H[te, k], P[:, k]) for k in range(K)])


def select_score(S_list):
    return float(np.mean(cv(S_list)[1] if a.select == "cv" else holdout(S_list)))


def cv(S_list):
    """Out-of-fold decoded predictions (standardized human scale) and the mean within-fold Spearman."""
    P = np.zeros_like(H)
    rs = []
    for f in range(5):
        tr, te = FOLD != f, FOLD == f
        Ftr, Fte = features(S_list, tr, te)
        Y = H[tr]; mu, sd = Y.mean(0), Y.std(0)
        P[te] = anchored_fit(Ftr, (Y - mu) / sd)(Fte) * sd + mu
        rs.append([sp(H[te, k], P[te, k]) for k in range(K)])
    r = np.nanmean(np.array(rs), 0)
    return P, r


def feedback(S_list, prompts):
    P, r = cv(S_list)  # out-of-fold over the whole pool; only fit items are shown
    worst = np.argsort(r)[:a.nshow]
    out = [f"The judge is currently made of {len(prompts)} prompt(s). A learned per-attribute calibration combines all "
           f"of their ratings and the length of the item into the final ratings; it already handles leniency and how "
           f"much each attribute rewards length. Agreement of the final ratings with human experts (Spearman): "
           + ", ".join(f"{x} {v:.2f}" for x, v in zip(A, r)) + "."]
    for i, p in enumerate(prompts):
        out.append(f"### Prompt {i + 1} in the judge\n{p}" + (INIT_NOTE if i == 0 and a.init != "v0" else ""))
    out.append("### Items the final ratings get most wrong (on the weakest attributes)")
    fit_idx = np.where(IS_FIT)[0]
    for k in worst:
        err = (H[fit_idx, k] - P[fit_idx, k])
        for j in fit_idx[np.argsort(-np.abs(err))[:2]]:
            item = POOL[j]["user"].strip()
            item = item[:1500] + (" [...]" if len(item) > 1500 else "")
            out.append(f"--- attribute '{A[k]}': human {H[j, k]:.0f}, final rating {P[j, k]:.1f}\n{item}\n"
                       f"Human ratings: {json.dumps({x: int(round(v)) for x, v in zip(A, H[j])})}; final ratings: "
                       f"{json.dumps({x: round(float(v), 1) for x, v in zip(A, P[j])})}")
    return "\n\n".join(out), r


TASK = """You improve an LLM judge that rates items on these attributes: {attrs}. The judge's ratings are not used directly: they are combined with the ratings of the other prompts in the judge by a learned calibration, separately for each attribute.

{fb}

Write ONE new judge prompt to ADD to the judge. It must rate the same attributes on the same scale, but it should capture what the current prompts miss on items like the ones above, e.g. by defining the weak attributes differently, asking the judge to look for evidence the current prompts ignore, or separating aspects the current prompts conflate. It does not need to be good on every attribute by itself, because the calibration reads each attribute from whichever prompt carries its information. Keep the attribute names and the rating scale; do not write an output-format section (it is appended separately).

Return only the new prompt between <prompt> and </prompt>."""


def propose(fb):
    msgs = [dict(role="user", content=TASK.format(attrs=", ".join(A), fb=fb))]
    for _ in range(3):
        txt = gem.chat(msgs, model=gem.OPTIM, temperature=0.7, max_tokens=4000, cache=False)[0]
        if "<prompt>" in txt and "</prompt>" in txt:
            return txt.split("<prompt>", 1)[1].split("</prompt>", 1)[0].strip()
    return None


INIT_D = f"{gem.OUT}/{a.ds}/{'v0' if a.init == 'v0' else 'dspy_' + a.init}/seed0"
INIT_NOTE = "\n(This prompt is followed by a few examples rated by human experts and asks for a short reasoning first.)"


def init_prompt():
    if a.init == "v0":
        return ds.v0, pool_scores(ds.v0)
    prog = json.load(open(f"{INIT_D}/program.json"))["predict"]
    df = pd.concat([pd.read_csv(f"{INIT_D}/scores_{s}.csv") for s in ("fit", "sel")]).set_index("id")
    return prog["signature"]["instructions"], ds.worst_case(df.reset_index()).set_index("id").loc[PIDS, A].to_numpy(float)


state_f = f"{OUTD}/state.json"
if os.path.exists(state_f):  # resume
    st = json.load(open(state_f))
else:
    p0, s0 = init_prompt()
    st = dict(prompts=[p0], scores={"0": s0.tolist()}, archive=[0], hist=[], step=0)
S = lambda i: np.array(st["scores"][str(i)])
cur = select_score([S(i) for i in st["archive"]])
if not st["hist"]:
    st["hist"].append(dict(step=0, archive=[0], cv=cur))
    print(f"[{a.ds}] step 0 archive [0] CV {cur:.4f}", flush=True)
for step in range(st["step"] + 1, a.steps + 1):
    arch = st["archive"]
    fb, r = feedback([S(i) for i in arch], [st["prompts"][i] for i in arch])
    new = propose(fb)
    rec = dict(step=step)
    if new is not None:
        st["prompts"].append(new); n = len(st["prompts"]) - 1
        st["scores"][str(n)] = pool_scores(new).tolist()
        options = [arch + [n]] if len(arch) < a.M else [[x for x in arch if x != d] + [n] for d in arch[1:]]
        vals = [select_score([S(i) for i in o]) for o in options]
        b = int(np.argmax(vals))
        rec.update(cand=n, cand_cv=vals[b], kept=vals[b] > cur)
        if vals[b] > cur:
            st["archive"], cur = options[b], vals[b]
    rec.update(archive=st["archive"], cv=cur)
    st["hist"].append(rec); st["step"] = step
    json.dump(st, open(state_f, "w"))
    print(f"[{a.ds}] step {step} cand CV {rec.get('cand_cv', float('nan')):.4f} "
          f"{'keep' if rec.get('kept') else 'drop'} archive {st['archive']} CV {cur:.4f}", flush=True)

json.dump(dict(prompts={str(i): st["prompts"][i] for i in st["archive"]}, archive=st["archive"]),
          open(f"{OUTD}/final_archive.json", "w"), indent=1)
v0d = INIT_D
for i in st["archive"]:
    for s in ("fit", "sel", "test"):
        f = f"{OUTD}/scores_{s}_p{i}.csv"
        if os.path.exists(f):
            continue
        if i == 0 and os.path.exists(f"{v0d}/scores_{s}.csv"):
            pd.read_csv(f"{v0d}/scores_{s}.csv").to_csv(f, index=False)
        else:
            ds.judge(st["prompts"][i], ds.rows[s], a.workers).to_csv(f, index=False)
print(a.ds, METHOD, "done; archive", st["archive"], "calls", gem.spent(gem.TAG.v)[1], flush=True)

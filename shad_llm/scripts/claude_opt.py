"""Automated multi-objective prompt optimization of a per-call Claude judge (Haiku), optimizer = Sonnet (claude -p).

Arms (same initial prompt v0, same budget: STEPS steps x CANDS candidates, each judged on fit and sel):
  text   TextGrad/CCC-like: the optimizer sees per-attribute raw Spearman and the largest raw errors of each
         attribute on fit; the candidate (or the current prompt) with the best mean raw Spearman on sel is kept.
  decoded  decoder-aware TextGrad: same optimizer and feedback format as text, but every prompt is read through its
         per-attribute readout (the decoder): the optimizer is shown the decoded Spearman and the items the decoded
         scores rank most wrongly (cross-fitted on fit), and an edit is kept by its mean decoded Spearman change on
         sel (the current prompt scores 0). A move along the length axis is absorbed by the decoder and earns nothing.
  ours   readout-decoupled: every prompt gets a calibration readout (anchored ridge from all its numeric scores +
         length to each human attribute, refitted per prompt on fit, CV-chosen shrinkage toward the raw score;
         the raw score itself is one of the options). The optimizer sees what the readout cannot fix: the top
         SVD direction of the cross-fitted residual (human - readout) on fit, with the rows at both ends, and may
         ask for extra integer sub-scores. Candidates are scored on sel by readout-adjusted per-attribute
         Spearman change vs the current prompt, divided by its paired-bootstrap SD, aggregated by soft-min
         (-log mean exp(-u)); the current prompt scores 0, so keeping it needs no threshold.
  ours-hist  ours + the optimizer sees every edit tried so far (its one-line idea) with its per-attribute
         readout-adjusted change on sel, so it can look for changes that keep the gains without the losses.
  ours-nosoftmin / ours-nores / ours-noaux: ablations (mean instead of soft-min / raw-error feedback instead of
         residual SVD / no extra sub-scores).

  python scripts/claude_opt.py results/claude_se ours [--steps 4] [--cands 3] [--workers 12]
Everything is cached under <root>/opt/<arm>/ (resumable); log.jsonl has one line per step.
"""
import argparse, glob, json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("arm")
ap.add_argument("--steps", type=int, default=4); ap.add_argument("--cands", type=int, default=3)
ap.add_argument("--workers", type=int, default=12); ap.add_argument("--judge", default="haiku")
ap.add_argument("--optimizer", default="sonnet"); ap.add_argument("--dry", action="store_true")
ap.add_argument("--test", action="store_true")  # judge the final prompt on test (after the run)
ap.add_argument("--names", default="")
ap.add_argument("--split", default="test")  # with --test: the split to judge the versions on (test or route)
ap.add_argument("--nsub", type=int, default=0)  # use only this many fit and sel rows (fixed random subset)  # with --test: comma-separated prompt versions to judge on test instead
a = ap.parse_args()
R, ARM = a.root, a.arm
# judge "or:<model>" calls OpenRouter (key in ~/.openrouter_key); its runs live under opt/<arm>__<model>
OR_MODEL = a.judge[3:] if a.judge.startswith("or:") else None
O = f"{R}/opt/{ARM}" + (f"__{OR_MODEL.split('/')[-1]}" if OR_MODEL else "")
os.makedirs(f"{O}/prompts", exist_ok=True); os.makedirs(f"{O}/out", exist_ok=True)
A = open(f"{R}/attrs.txt").read().split()
FORMAT = open(f"{R}/format.txt").read().strip()
GOLD = pd.read_csv(f"{R}/gold.csv")
LO, HI = map(int, re.search(r"integer from (\d+) to (\d+)", FORMAT).groups())
ROWS = {s: [json.loads(l) for f in sorted(glob.glob(f"{R}/chunks/{s}_*.jsonl")) for l in open(f)] for s in ("fit", "sel", "test", "route")}
if a.nsub:
    for s_ in ("fit", "sel"):
        keep = sorted(np.random.default_rng(0).permutation(len(ROWS[s_]))[:a.nsub])
        ROWS[s_] = [ROWS[s_][i] for i in keep]
IDS = {r["id"] for s_ in ROWS for r in ROWS[s_]}
# arm "sep-<attr>": a separate prompt optimized for one attribute only (the others are still output but ignored)
SEP = ARM[4:] if ARM.startswith("sep-") else None
USER = {r["id"]: r["user"] for s in ROWS for r in ROWS[s]}
META = ("id", "reason", "in_tok", "out_tok", "cost")
LAMS = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1e5]
TAU = 1.0
HISTORY = "hist"  # arms whose name contains this see the edits tried so far and their per-attribute effect
HIST = []
sp = lambda x, y: pd.Series(np.asarray(x, float)).corr(pd.Series(np.asarray(y, float)), method="spearman")
std = lambda X: (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)


class Abort(Exception):
    pass


def log(msg):
    print(msg, flush=True)
    open(f"{O}/run.log", "a").write(msg + "\n")


# ---------------------------------------------------------------- judging (one claude -p call per row, cached CSV)
def judge_rows(name, prompt, split):
    path = f"{O}/out/{name}_{split}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    v0 = f"{R}/out/{name}_{split}.csv"  # the initial prompt was already judged by scripts/claude_judge.py
    if name == "v0" and os.path.exists(v0) and not OR_MODEL:
        return pd.read_csv(v0)
    system = prompt.strip() + "\n\n" + FORMAT

    def one_or(r):
        err = ""
        for t in range(4):
            try:
                body = json.dumps(dict(model=OR_MODEL, messages=[dict(role="system", content=system), dict(role="user", content=r["user"])],
                                       temperature=0, max_tokens=2000, reasoning=dict(enabled=False), usage=dict(include=True))).encode()
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
                    "Authorization": "Bearer " + open(os.path.expanduser("~/.openrouter_key")).read().strip(),
                    "Content-Type": "application/json"})
                o = json.loads(urllib.request.urlopen(req, timeout=180).read())
                if "error" in o:
                    raise RuntimeError(str(o["error"])[:200])
                txt, u = o["choices"][0]["message"].get("content") or "", o.get("usage", {})
                sc = {}
                for js in reversed(re.findall(r"\{[^{}]*\}", txt)):
                    try:
                        sc = {k: float(v) for k, v in json.loads(js).items() if isinstance(v, (int, float))}
                        break
                    except Exception:  # noqa: BLE001
                        continue
                return dict(id=r["id"], **sc, reason=" ".join(txt.split())[-300:], in_tok=u.get("prompt_tokens", 0),
                            out_tok=u.get("completion_tokens", 0), cost=u.get("cost", 0.0) or 0.0)
            except Exception as e:  # noqa: BLE001
                err = str(e)[:300]
                time.sleep(2 * (t + 1))
        return dict(id=r["id"], reason="ERROR " + err, in_tok=0, out_tok=0, cost=0.0)

    def one(r):
        if OR_MODEL:
            return one_or(r)
        err = ""
        for _ in range(3):
            try:
                p = subprocess.run(["claude", "-p", "--model", a.judge, "--output-format", "json", "--no-session-persistence",
                                    "--tools", "", "--setting-sources", "", "--system-prompt", system],
                                   input=r["user"], capture_output=True, text=True, timeout=300, cwd="/tmp")
                o = json.loads(p.stdout)
                txt, u = o.get("result", ""), o.get("usage", {})
                if o.get("is_error"):
                    raise RuntimeError(txt[:200])
                sc = {}
                for js in reversed(re.findall(r"\{[^{}]*\}", txt)):
                    try:
                        sc = {k: float(v) for k, v in json.loads(js).items() if isinstance(v, (int, float))}
                        break
                    except Exception:  # noqa: BLE001
                        continue
                return dict(id=r["id"], **sc, reason=" ".join(txt.split())[-300:],
                            in_tok=u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
                            out_tok=u.get("output_tokens", 0), cost=o.get("total_cost_usd", 0.0))
            except Exception as e:  # noqa: BLE001
                err = (str(e) + " " + (p.stdout[-200:] if "p" in locals() and p.stdout else ""))[:300]
        return dict(id=r["id"], reason="ERROR " + err, in_tok=0, out_tok=0, cost=0.0)

    with ThreadPoolExecutor(a.workers) as ex:
        out = pd.DataFrame(list(ex.map(one, ROWS[split])))
    bad = out.reason.str.startswith("ERROR")
    if bad.mean() > 0.2:  # quota / moderation / outage: stop everything, never retry around it
        out.to_csv(path + ".failed", index=False)
        raise Abort(f"{name}_{split}: {bad.sum()}/{len(out)} calls failed, e.g. {out.reason[bad].iloc[0]}")
    out.to_csv(path, index=False)
    log(f"  judged {name}_{split}: {out[A[0]].notna().sum() if A[0] in out else 0}/{len(out)} parsed, ${out.cost.sum():.2f}")
    return out


def frame(df, split):
    d = GOLD[(GOLD.split == split) & GOLD.id.isin(IDS)].merge(df, on="id", suffixes=("", "_j"))
    d = d.rename(columns={c: c + "_j" for c in df.columns if c not in META and c not in A and c + "_j" not in d})
    d = d.dropna(subset=[x + "_j" for x in A])
    return d


def feats(d, ref):
    """Judge's numeric outputs (required attributes first, then extra sub-scores) + standardized log length."""
    cols = [x + "_j" for x in A] + sorted(c for c in ref.columns if c.endswith("_j") and c[:-2] not in A)
    cols = [c for c in cols if ref[c].notna().any()]
    X = d.reindex(columns=cols).astype(float)
    X = X.fillna(ref[cols].astype(float).mean())
    Xr = ref[cols].astype(float).fillna(ref[cols].astype(float).mean())
    m, s = Xr.mean(0).to_numpy(), Xr.std(0).replace(0, 1).fillna(1).to_numpy()
    return np.c_[(X.to_numpy() - m) / s, (d.z.to_numpy() - ref.z.mean()) / ref.z.std()]


# ---------------------------------------------------------------- readout
def anchored_fit(X, Y, folds=5):
    """Per attribute k: ridge shrunk toward the own-score slope, strength by K-fold CV Spearman; option 'raw'
    (the own score alone, ties kept) competes in the same CV. Returns a predictor X -> scores."""
    fold = np.arange(len(X)) % folds
    Ws = []
    for k in range(Y.shape[1]):
        y = Y[:, k]

        def fit(Xa, ya, lam):
            w0 = np.zeros(Xa.shape[1]); w0[k] = (Xa[:, k] @ ya) / max(Xa[:, k] @ Xa[:, k], 1e-9)
            if lam is None:
                return w0
            return np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya + lam * w0)

        opts = LAMS + [None]
        cv = []
        for lam in opts:
            # Spearman within each held-out fold, averaged: pooling folds would break the raw score's ties
            # through the folds' slightly different slopes
            cv.append(np.nanmean([sp(y[fold == f], X[fold == f] @ fit(X[fold != f], y[fold != f], lam))
                                  for f in range(folds)]))
        Ws.append(fit(X, y, opts[int(np.nanargmax(cv))]))
    W = np.array(Ws).T
    return lambda Xn: Xn @ W


def readout_scores(fit_df, eval_df):
    """Readout fitted on fit_df, applied to eval_df: (n_eval, K) predictions."""
    f = anchored_fit(feats(fit_df, fit_df), std(fit_df[A].to_numpy(float)))
    return f(feats(eval_df, fit_df))


def per_task(H, P):
    return np.array([sp(H[:, k], P[:, k]) for k in range(len(A))])


# ---------------------------------------------------------------- evaluation of one prompt
def evaluate(name, prompt):
    fit, sel = frame(judge_rows(name, prompt, "fit"), "fit"), frame(judge_rows(name, prompt, "sel"), "sel")
    raw = sel[[x + "_j" for x in A]].to_numpy(float)
    ro = readout_scores(fit, sel)
    return dict(name=name, prompt=prompt, fit=fit, sel=sel, raw=raw, ro=ro,
                sel_raw=per_task(sel[A].to_numpy(float), raw), sel_ro=per_task(sel[A].to_numpy(float), ro))


def paired_u(cur, cand, readout=True, B=300):
    """Per attribute: (Spearman change on the sel rows both judged) / its paired-bootstrap SD."""
    ids = np.intersect1d(cur["sel"].id, cand["sel"].id)
    ic = cur["sel"].reset_index(drop=True).set_index("id").index.get_indexer(ids)
    ia = cand["sel"].reset_index(drop=True).set_index("id").index.get_indexer(ids)
    H = cur["sel"][A].to_numpy(float)[ic]
    Pc, Pa = (cur["ro"], cand["ro"]) if readout else (cur["raw"], cand["raw"])
    Pc, Pa = Pc[ic], Pa[ia]
    d = per_task(H, Pa) - per_task(H, Pc)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(ids), len(ids))
        bs.append(per_task(H[i], Pa[i]) - per_task(H[i], Pc[i]))
    sd = np.nanstd(np.array(bs), 0)
    return d, d / np.where(sd > 0, sd, 1)


def softmin(u):
    return float(-TAU * np.log(np.mean(np.exp(-np.asarray(u) / TAU))))


# ---------------------------------------------------------------- feedback for the optimizer
def show(i_row, d, cols_j):
    txt = USER[d.id.iloc[i_row]]
    txt = txt if len(txt) < 1800 else txt[:1500] + " [...] " + txt[-300:]
    hs = ", ".join(f"{x} {d[x].iloc[i_row]:g}" for x in A)
    js = ", ".join(f"{c[:-2]} {d[c].iloc[i_row]:g}" for c in cols_j)
    return f"<item>\n{txt}\n</item>\nHuman scores: {hs}\nJudge scores: {js}\nJudge's reasoning (end): {d.reason.iloc[i_row]}\n"


def raw_error_feedback(cur):
    d = cur["fit"]
    cols = [x + "_j" for x in A]
    out = ["Spearman correlation with the human scores on held-out items: " +
           ", ".join(f"{x} {v:.2f}" for x, v in zip(A, per_task(d[A].to_numpy(float), d[cols].to_numpy(float))))]
    for x in ([SEP] if SEP else A):
        err = (d[x + "_j"] - d[x]).abs().to_numpy()
        for i in np.argsort(-err, kind="stable")[:(6 if SEP else 2)]:
            out.append(f"--- Large error on {x}:\n" + show(int(i), d, cols))
    return "\n".join(out)


def decoded_error_feedback(cur):
    """text-arm feedback after the decoder: decoded Spearman and the items the cross-fitted decoded scores rank
    most wrongly per attribute (percentile gap), weakest attribute first."""
    d = cur["fit"]
    cols = [x + "_j" for x in A]
    H = d[A].to_numpy(float)
    P = np.zeros((len(d), len(A)))
    fold = np.arange(len(d)) % 5
    for f in range(5):
        P[fold == f] = readout_scores(d[fold != f], d[fold == f])
    r = np.nanmean([per_task(H[fold == f], P[fold == f]) for f in range(5)], 0)
    pct = lambda x: pd.Series(x).rank(pct=True).to_numpy()
    out = ["Spearman correlation with the human scores on held-out items, after the calibration layer: " +
           ", ".join(f"{x} {v:.2f}" for x, v in zip(A, r))]
    for k in np.argsort(r):
        err = pct(H[:, k]) - pct(P[:, k])
        for i in np.argsort(-np.abs(err), kind="stable")[:2]:
            out.append(f"--- Large error on {A[k]} after calibration: humans put this item at percentile "
                       f"{100 * pct(H[:, k])[i]:.0f}, the calibrated judge at {100 * pct(P[:, k])[i]:.0f}.\n" + show(int(i), d, cols))
    return "\n".join(out), dict(decoded_fit=[float(x) for x in r])


DECODED_NOTE = """After judging, a calibration layer (refitted for every version of the instructions) maps the judge's scores plus the item's length to each attribute; it already corrects leniency and how much each attribute should reward or penalise length, so edits that only make the judge reward length more or less earn nothing. The correlations and errors below are measured after that layer: they show what the judge's scores do not capture."""


def residual_feedback(cur):
    d = cur["fit"]
    cols = [c for c in d.columns if c.endswith("_j")]
    H = std(d[A].to_numpy(float))
    P = np.zeros_like(H)
    fold = np.arange(len(d)) % 5
    for f in range(5):  # cross-fitted readout on fit
        tr, te = d[fold != f], d[fold == f]
        P[fold == f] = readout_scores(tr, te)
    Rm = H - P
    U, s, Vt = np.linalg.svd(Rm - Rm.mean(0), full_matrices=False)
    v, u = Vt[0] * np.sign(Vt[0].sum() or 1), U[:, 0] * np.sign(Vt[0].sum() or 1)
    share = s[0] ** 2 / np.sum(s ** 2)
    lead = [A[k] for k in np.argsort(-np.abs(v)) if abs(v[k]) > 0.3 * np.abs(v).max()]
    sign = {A[k]: ("higher" if v[k] > 0 else "lower") for k in range(len(A))}
    Hd = d[A].to_numpy(float)
    ro_sp = np.nanmean([per_task(Hd[fold == f], P[fold == f]) for f in range(5)], 0)  # within folds (ties)
    out = [
        "Spearman correlation with the human scores on held-out items, after the calibration layer: " +
        ", ".join(f"{x} {r:.2f}" for x, r in zip(A, ro_sp)) + "  (raw scores: " +
        ", ".join(f"{x} {r:.2f}" for x, r in zip(A, per_task(d[A].to_numpy(float), d[[x + '_j' for x in A]].to_numpy(float)))) + ")",
        f"What the calibrated scores still miss: the largest shared direction of the remaining errors ({100 * share:.0f}% of "
        f"them) has loadings " + ", ".join(f"{x} {w:+.2f}" for x, w in zip(A, v)) +
        f" (mostly {', '.join(lead)}). On the items below at the POSITIVE end humans gave relatively higher "
        f"{'/'.join(x for x in lead if v[A.index(x)] > 0) or '-'} and lower {'/'.join(x for x in lead if v[A.index(x)] < 0) or '-'} "
        "than the calibrated judge predicts; at the NEGATIVE end the reverse. Find the property of the items that "
        "the instructions make the judge overlook, and that explains both ends."]
    for tag, idx in (("POSITIVE end", np.argsort(-u)[:3]), ("NEGATIVE end", np.argsort(u)[:3])):
        for i in idx:
            out.append(f"--- {tag}:\n" + show(int(i), d, cols))
    return "\n".join(out), dict(share=float(share), loadings=[float(x) for x in v])


OPT_SYS = """You improve the instructions (system prompt) of an LLM judge. The judge reads one item and rates it on several attributes at once ({attrs}; integers {lo}-{hi}); its ratings are compared with human ratings by Spearman correlation per attribute, and every attribute matters equally: a change that helps one attribute and hurts another is not an improvement.
The judge's output-format section is appended automatically after the instructions; do not write one. Keep the attribute names and the {lo}-{hi} scale. Keep the instructions reasonably concise (well under 1500 words)."""

OURS_NOTE = """After judging, a calibration layer (refitted for every version of the instructions on held-out items) maps all the judge's numeric outputs plus the item's length to each human attribute. It already corrects: how lenient or strict each attribute is, how much each attribute should reward or penalise length, and how much one attribute's score leaks into another. Edits that only move scores along those directions earn nothing. What it cannot correct is information the judge's scores do not contain: the instructions should make the judge notice what it overlooks."""

AUX_NOTE = """You may also ask the judge to output a few extra integer sub-scores as additional JSON keys (e.g. "factual_errors": count), if they would capture what the judge overlooks; the calibration layer uses every numeric key. The required attribute keys must stay."""

OPT_TASK = """<current_instructions>
{prompt}
</current_instructions>

<feedback>
{feedback}
</feedback>

Write {n} different revised versions of the instructions, each a complete replacement, each trying a different idea. Reply with a one-line rationale for each, then the versions as <prompt1>...</prompt1>, <prompt2>...</prompt2>, ..."""


def optimize(cur, step):
    path = f"{O}/prompts/s{step}_opt.txt"
    if ARM == "text" or ARM == "ours-nores" or SEP:
        fb, info = raw_error_feedback(cur), {}
    elif ARM == "decoded":
        fb, info = decoded_error_feedback(cur)
    else:
        fb, info = residual_feedback(cur)
    notes = (f"\n\nThis judge is used ONLY for the attribute {SEP}: only its {SEP} score is compared with humans; "
             f"the other attributes are ignored, so optimize the instructions for {SEP} alone." if SEP else
             "") if (ARM == "text" or SEP) else "\n\n" + DECODED_NOTE if ARM == "decoded" else "\n\n" + OURS_NOTE + ("" if ARM == "ours-noaux" else "\n" + AUX_NOTE)
    system = OPT_SYS.format(attrs=", ".join(A), lo=LO, hi=HI) + notes
    if HISTORY in ARM and HIST:
        fb += ("\n\nEdits tried so far and their measured effect on held-out items (Spearman change per attribute "
               "after the calibration layer; 'kept' = became the current instructions). Edits that raised some "
               "attributes but lowered others were rejected: find changes that keep the gains without the losses.\n" +
               "\n".join(f"- step {h['step']} ({'kept' if h['kept'] else 'rejected'}): {h['idea']} -> " +
                         ", ".join(f"{x} {d:+.3f}" for x, d in zip(A, h['d'])) for h in HIST[-12:]))
    user = OPT_TASK.format(prompt=cur["prompt"], feedback=fb, n=a.cands)
    if os.path.exists(path):
        txt = open(path).read()
    else:
        txt = ""
        for _ in range(3):
            p = subprocess.run(["claude", "-p", "--model", a.optimizer, "--output-format", "json", "--no-session-persistence",
                                "--tools", "", "--setting-sources", "", "--system-prompt", system],
                               input=user, capture_output=True, text=True, timeout=900, cwd="/tmp")
            try:
                o = json.loads(p.stdout)
            except Exception:  # noqa: BLE001
                continue
            if o.get("is_error"):
                raise Abort("optimizer: " + str(o.get("result", ""))[:300])
            txt = o.get("result", "")
            log(f"  optimizer ${o.get('total_cost_usd', 0):.3f}")
            if re.search(r"<prompt1>", txt):
                break
        open(path, "w").write(txt)
        open(f"{O}/prompts/s{step}_opt_input.txt", "w").write(system + "\n\n=====\n\n" + user)
    cands = [m.strip() for m in re.findall(r"<prompt\d>(.*?)</prompt\d>", txt, re.S)][:a.cands]
    if not cands:
        raise Abort(f"optimizer reply without prompts at step {step}")
    head = [l.strip(" -*#") for l in txt.split("<prompt1>")[0].splitlines() if len(l.strip()) > 15]
    ideas = (head[-len(cands):] if len(head) >= len(cands) else [f"version {k + 1}" for k in range(len(cands))])
    return cands, info, [i[:300] for i in ideas]


# ---------------------------------------------------------------- main loop
def main():
    v0 = open(f"{R}/prompts/v0.txt").read()
    cur = evaluate("v0", v0)
    open(f"{O}/prompts/current_s0.txt", "w").write(v0)
    if a.dry:
        print((raw_error_feedback(cur) if ARM in ("text", "ours-nores") or SEP else
               decoded_error_feedback(cur)[0] if ARM == "decoded" else residual_feedback(cur)[0])[:3000]); return
    log(f"[{ARM}] step 0 sel raw {np.round(cur['sel_raw'], 3)} ro {np.round(cur['sel_ro'], 3)}")
    for step in range(1, a.steps + 1):
        cands, info, ideas = optimize(cur, step)
        rows = []
        for k, c in enumerate(cands):
            name = f"s{step}c{k}"
            open(f"{O}/prompts/{name}.txt", "w").write(c)
            ev = evaluate(name, c)
            d_raw, u_raw = paired_u(cur, ev, readout=False)
            d_ro, u_ro = paired_u(cur, ev, readout=True)
            if SEP:
                score = float(d_raw[A.index(SEP)])
            elif ARM == "text":
                score = float(np.nanmean(ev["sel_raw"]) - np.nanmean(cur["sel_raw"]))
            elif ARM == "decoded":
                score = float(np.nanmean(d_ro))
            elif ARM == "ours-nosoftmin":
                score = float(np.nanmean(u_ro))
            else:
                score = softmin(u_ro)
            rows.append((score, ev, dict(name=name, score=score, d_raw=d_raw.round(4).tolist(), u_raw=u_raw.round(2).tolist(),
                                         d_ro=d_ro.round(4).tolist(), u_ro=u_ro.round(2).tolist(),
                                         n_extra=len([c_ for c_ in ev["fit"].columns if c_.endswith("_j")]) - len(A))))
            log(f"  {name}: score {score:+.3f} d_raw {np.round(d_raw, 3)} d_ro {np.round(d_ro, 3)} extra {rows[-1][2]['n_extra']}")
        best = max(rows, key=lambda r: r[0])
        accepted = best[0] > 0  # the current prompt is candidate 0 with score 0
        for k, r in enumerate(rows):
            HIST.append(dict(step=step, idea=ideas[k], d=r[2]["d_ro"], kept=accepted and r is best))
        if accepted:
            cur = best[1]
        open(f"{O}/prompts/current_s{step}.txt", "w").write(cur["prompt"])
        open(f"{O}/log.jsonl", "a").write(json.dumps(dict(step=step, accepted=accepted, current=cur["name"], feedback=info,
                                                          cands=[r[2] for r in rows])) + "\n")
        log(f"[{ARM}] step {step}: {'accept ' + cur['name'] if accepted else 'keep'}; sel raw {np.round(cur['sel_raw'], 3)} "
            f"ro {np.round(cur['sel_ro'], 3)}")
    open(f"{O}/final.txt", "w").write(cur["name"])
    log(f"[{ARM}] final {cur['name']}")


def test():
    for name in (a.names.split(",") if a.names else [open(f"{O}/final.txt").read().strip()]):
        prompt = open(f"{R}/prompts/v0.txt" if name == "v0" else f"{O}/prompts/{name}.txt").read()
        judge_rows(name, prompt, a.split)
        log(f"[{ARM}] judged {name} on {a.split}")


if __name__ == "__main__":
    try:
        test() if a.test else main()
    except Abort as e:
        log(f"ABORT {e}")
        sys.exit(2)

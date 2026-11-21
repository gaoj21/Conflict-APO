"""ShAD with evaluation dimensions learned per dataset (offline: no val / test).

  QA_DS=<drop|musique|hotpot> .venv/bin/python qa/dimlearn.py [--k 24] [--workers 32]
Input: the 1000 pool answers of TextGrad's final prompt (qa/drop_textgrad.py tg) with their references.
1. candidates: the model reads contrasting pairs (its answer vs the reference) from the first 300 pool questions and
   proposes K dimensions of how an answer is written (1-5 with anchors), each judgeable from the question and one
   answer alone, none about correctness
2. ratings: the same model rates the model's answer A and the reference GT, each alone with the question, on all K
3. ShAD decoder: per dimension k, ridge GT_k ~ all K dimensions of A + log length; DEV_k = A_k - decoded GT_k
4. which dimensions matter (learned, not chosen): L1 logistic P(wrong) ~ [DEV+, DEV-] of all K, strength by 5-fold CV
   on all 1000; the dimensions with nonzero weight are this dataset's dimensions
5. shared axes: non-negative factorization of the selected [DEV+, DEV-] columns (rank by CV reconstruction); each axis
   = the deviations that move together; its cost = logistic weight of P(wrong) on the axis
6. translation by template (no LLM): per axis with positive cost, one global rule naming the axis' deviations with
   the anchors the dimensions were defined with; ranked by cost x share -> dimlearn/ranked.json
   (the same format as qa/dimshad.py, so qa/drop_dimshad_combine.py --src dimlearn adds them one by one on val)
"""
import hashlib, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE) + "/baselines")
import gem  # noqa: E402

DS = os.environ.get("QA_DS", "drop")
argv = sys.argv[1:]; opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
K, W = int(opt("--k", "24")), int(opt("--workers", "32"))
BASE = f"{gem.OUT}_qa/{DS}/shadqa" + ("_official" if DS == "drop" else "") + ("_devval" if os.environ.get("QA_VAL") == "dev" else ""); OUT = f"{BASE}/dimlearn"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = f"qa/{DS}/dimlearn"

PROPOSE = """Below are questions with a model's answer and the reference answer (mostly pairs where they differ).

{pairs}

Propose {k} dimensions of HOW AN ANSWER IS WRITTEN on which answers like these can differ (its form, extent, level of
detail, wording, kind of answer, what it adds or leaves out, ...). Rules for every dimension:
- it can be rated from the question and ONE answer alone, without seeing any reference;
- it is not about whether the answer is correct, good or helpful;
- the answerer could move an answer up or down on it without changing what the answer refers to.
Reply with only a JSON list: [{{"name": "<short snake_case name>", "low": "<what a 1 looks like>", "high": "<what a 5 looks like>"}}]"""
RATE = """Question: {q}
Answer: {a}

Rate how this answer is written, from 1 to 5, on each dimension (do not judge whether it is correct):
{dims}
Reply with only one line of JSON mapping each dimension name to its rating."""


def pairs(A, n_wrong=40, n_right=10):
    W_, R_ = A[A.correct == 0].head(n_wrong), A[A.correct == 1].head(n_right)
    return "\n".join(f"- Q: {r.question[:150]}\n  model: {r.pred[:100]!r}  reference: {r.gold[:100]!r}" for r in pd.concat([W_, R_]).itertuples())


def rate(q, a, dims, prompt_dims):
    txt = gem.chat([dict(role="user", content=RATE.format(q=q, a=a if str(a).strip() else "(no answer)", dims=prompt_dims))], max_tokens=600)[0]
    try:
        d = json.loads(re.findall(r"\{.*\}", txt or "", re.S)[-1])
        return {k: float(d[k]) for k in dims}
    except Exception:  # noqa: BLE001
        return {k: np.nan for k in dims}


if __name__ == "__main__":
    sysp = json.load(open(f"{BASE}/textgrad/tg.json"))["sys"]
    A = pd.read_csv(f"{BASE}/textgrad/answers_shadfit_{hashlib.sha1((sysp + chr(0)).encode()).hexdigest()[:8]}.csv")
    A["pred"] = A.pred.fillna("").astype(str); A["gold"] = A.gold.astype(str)

    df = f"{OUT}/dims.json"  # 1. candidate dimensions, from the first 300 pool answers only
    if not os.path.exists(df):
        D_ = []
        for t in range(3):  # a reply that is not a JSON list is asked again (new sample)
            gem.SALT.v = f"propose{t}" if t else ""
            txt = gem.chat([dict(role="user", content=PROPOSE.format(pairs=pairs(A.head(300)), k=K))], max_tokens=3000, temperature=0.0 if t == 0 else 0.7)[0]
            try:
                D_ = json.loads(re.findall(r"\[.*\]", txt or "", re.S)[-1]); break
            except Exception:  # noqa: BLE001
                continue
        gem.SALT.v = ""
        if not D_:
            print(f"[{DS}] the model did not propose dimensions as a JSON list: stopping"); sys.exit(0)
        D_ = [d for d in D_ if re.fullmatch(r"[a-z][a-z0-9_]*", str(d.get("name", ""))) and d.get("low") and d.get("high")]
        json.dump(list({d["name"]: d for d in D_}.values()), open(df, "w"), indent=1)
    DIMS = json.load(open(df)); names = [d["name"] for d in DIMS]
    if not names:
        print(f"[{DS}] no usable dimension in {df}: stopping (delete it to propose again)"); sys.exit(0)
    prompt_dims = "\n".join(f"- {d['name']}: 1 = {d['low']}; 5 = {d['high']}" for d in DIMS)
    print(f"[{DS}] {len(DIMS)} candidate dimensions proposed from 300 pool answers:\n" + "\n".join(f"  {d['name']:28s} 1 = {d['low'][:60]} | 5 = {d['high'][:60]}" for d in DIMS), flush=True)

    rf = f"{OUT}/ratings.csv"  # 2. ratings of A and GT, each alone
    if os.path.exists(rf):
        R = pd.read_csv(rf)
    else:
        with ThreadPoolExecutor(W) as ex:
            ra = list(ex.map(lambda r: rate(r[0], r[1], names, prompt_dims), zip(A.question, A.pred)))
            rg = list(ex.map(lambda r: rate(r[0], r[1], names, prompt_dims), zip(A.question, A.gold)))
        R = pd.concat([A[["qid", "question", "pred", "gold", "correct"]], pd.DataFrame(ra).add_prefix("A_"), pd.DataFrame(rg).add_prefix("G_")], axis=1)
        R.to_csv(rf, index=False)
    R["pred"] = R.pred.fillna("").astype(str); R["gold"] = R.gold.fillna("").astype(str)
    R = R.dropna(subset=[f"A_{k}" for k in names] + [f"G_{k}" for k in names]).reset_index(drop=True)
    if len(R) < 50:
        print(f"[{DS}] only {len(R)} answer/reference pairs have parsable ratings: stopping (check the judge replies in {rf})"); sys.exit(0)
    wrong = 1 - R.correct.to_numpy(); n = len(R)
    if min(wrong.sum(), n - wrong.sum()) < 10:
        print(f"[{DS}] too few right or wrong answers ({int(n - wrong.sum())} / {int(wrong.sum())}) to learn from: stopping"); sys.exit(0)
    print(f"\n{n} answer/reference pairs rated on all {len(names)} dimensions, wrong {100 * wrong.mean():.1f}%  (spent ${gem.spent()[0]:.2f})", flush=True)

    from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LogisticRegression
    from sklearn.decomposition import NMF
    X = np.column_stack([R[f"A_{k}"] for k in names] + [np.log1p(R.pred.str.len())]); Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    DEV, R2 = {}, {}  # 3. ShAD decoder per dimension
    for i, k in enumerate(names):
        y = R[f"G_{k}"].to_numpy()
        if y.std() == 0:
            DEV[k] = R[f"A_{k}"].to_numpy() - y; R2[k] = (np.nan, np.nan); continue
        m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xs, y); own = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xs[:, [i]], y)
        DEV[k] = R[f"A_{k}"].to_numpy() - m.predict(Xs); R2[k] = (own.score(Xs[:, [i]], y), m.score(Xs, y))
    cols = [(k, "too high") for k in names] + [(k, "too low") for k in names]
    Z = np.column_stack([np.clip(DEV[k], 0, None) for k in names] + [np.clip(-DEV[k], 0, None) for k in names])
    # scale by the dimension's mean absolute deviation (both sides share it): a rare deviation stays small instead of
    # being blown up by its tiny standard deviation
    scale = np.array([np.abs(DEV[k]).mean() + 1e-9 for k in names] * 2); Zs = Z / scale
    occurs = (Z > .5).mean(0)  # how often each deviation actually happens (A at least half a point off the decoded GT)
    # 4. which deviations matter: L1 logistic; strength = the sparsest C within one SE of the best 5-fold log loss
    Cs = np.logspace(-2, 1, 12)
    l1 = LogisticRegressionCV(Cs=Cs, penalty="l1", solver="liblinear", cv=5, scoring="neg_log_loss").fit(Zs, wrong)
    sc = l1.scores_[1]; m_, se_ = sc.mean(0), sc.std(0) / np.sqrt(sc.shape[0])
    c1 = Cs[np.argmax(m_ >= m_.max() - se_[np.argmax(m_)])]
    w = LogisticRegression(C=c1, penalty="l1", solver="liblinear").fit(Zs, wrong).coef_[0]
    sel = [j for j in range(len(cols)) if w[j] > 0]
    print(f"\nlearned dimensions (L1 logistic P(wrong) on deviations, C={c1:.2f}: sparsest within 1 SE of the best 5-fold log loss): "
          f"{len(sel)} of {len(cols)} deviations kept")
    for j in sorted(sel, key=lambda j: -w[j]):
        k, s_ = cols[j]; e = Z[:, j] > .5
        print(f"  {k:28s} {s_:9s} weight {w[j]:+.2f}  share {100 * e.mean():4.1f}%  P(wrong|it) {wrong[e].mean() if e.any() else float('nan'):.2f}"
              f"  decoder R2 own {R2[k][0]:.2f} -> all {R2[k][1]:.2f}")
    if not sel:
        print("no deviation predicts wrong answers"); sys.exit(0)

    # 5. shared axes: NMF on the selected deviations weighted by their L1 weight, rank by 5-fold held-out reconstruction
    Zsel = Zs[:, sel] * w[sel]; rng = np.random.default_rng(0); fold = rng.integers(0, 5, n); best = (np.inf, 1)
    for r in range(1, min(6, len(sel)) + 1):
        err = 0.0
        for f in range(5):
            nmf = NMF(n_components=r, init="nndsvda", max_iter=2000, random_state=0).fit(Zsel[fold != f])
            Ht = nmf.transform(Zsel[fold == f]); err += ((Zsel[fold == f] - Ht @ nmf.components_) ** 2).sum()
        best = min(best, (err * (1 + .02 * r), r))  # a small preference for fewer axes
    nmf = NMF(n_components=best[1], init="nndsvda", max_iter=2000, random_state=0).fit(Zsel); Hx = nmf.transform(Zsel); C = nmf.components_
    cost = LogisticRegression(max_iter=2000).fit((Hx - Hx.mean(0)) / (Hx.std(0) + 1e-9), wrong).coef_[0]
    print(f"\nshared axes (NMF rank {best[1]} by 5-fold reconstruction): loadings of the selected deviations")
    ranked = []
    for a in np.argsort(-cost):
        load = C[a] / (C[a].max() + 1e-9)
        top, seen = [], set()  # the axis' main deviations that actually occur, one side per dimension
        for jj in np.argsort(-load):
            j = sel[jj]
            if load[jj] > .3 and occurs[j] > 0 and cols[j][0] not in seen:
                top.append(j); seen.add(cols[j][0])
            if len(top) == 3:
                break
        share = float((Z[:, top] > .5).any(1).mean()) if top else 0.0  # answers showing at least one of its deviations
        print(f"  axis {a}: cost {cost[a]:+.2f}, its deviations occur in {100 * share:.1f}% of answers: "
              + ", ".join(f"{cols[j][0]} {cols[j][1]} ({load[sel.index(j)]:.2f}, {100 * occurs[j]:.1f}%)" for j in top))
        if cost[a] <= 0 or not top:
            continue
        acts = []  # 6. template: move each deviation of the axis back towards its anchor
        for j in top:
            k, s_ = cols[j]; d = next(x for x in DIMS if x["name"] == k)
            acts.append(f"less {k.replace('_', ' ')} (closer to: {d['low']})" if s_ == "too high" else f"more {k.replace('_', ' ')} (closer to: {d['high']})")
        ranked.append(dict(axis=int(a), dims=[f"{cols[j][0]} {cols[j][1]}" for j in top], weight=float(cost[a]), share=share,
                           score=float(cost[a]) * share, rule="- In the final answer, write " + "; ".join(acts) + "."))
    ranked.sort(key=lambda r: -r["score"])
    for r in ranked:
        r["side"] = ""; r["dim"] = f"axis {r['axis']}"
    json.dump(ranked, open(f"{OUT}/ranked.json", "w"), indent=1)
    print("\ntranslated rules (ranked by cost x share):\n" + "\n".join(f"  {r['score']:.3f}  {r['rule']}" for r in ranked))
    print(f"done, spent ${gem.spent()[0]:.2f}")

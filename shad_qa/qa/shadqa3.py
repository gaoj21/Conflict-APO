"""ShAD for QA, v3: the decoder predicts the reference answer's features from the model answer's features (as ShAD
predicts human scores from the judge's scores), only answer features the model can change without changing what the
answer says are used, and each feature's rule is weighted by how much a mismatch on it goes with a wrong answer.

  .venv/bin/python qa/shadqa3.py <tatqa|musique> [--workers 32]
Reuses round-1 base answers of qa/shadqa.py. Steps on the 600 fit questions:
  1. feature proposal (contrasting pairs), each feature flagged actionable or not; only actionable ones are kept,
     plus the two universal counts (words, listed items)
  2. the same probe rates every feature for the model answer and for the reference answer
  3. decoder per feature k: reference_k ~ mean deviation (shrunk by t^2/(1+t^2)) + anchored ridge on all model-answer
     features (lambda by 5-fold CV), i.e. ShAD with answers in place of scores
  4. relevance of k: P(wrong | model and reference differ on k) - P(wrong | they agree), smooth weight max(0, .)
  5. one rule per feature from a template (mean deviation + the strongest conditional deviation), ordered by
     relevance x deviation size; the LLM only rephrases each line; rules apply only to the final Answer line
  6. candidate sets = top 1, 2, 4, all; chosen on the 200 val questions (base kept if none beats it); test as in v2
"""
import json, math, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ds_arg = sys.argv[1]; w_arg = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "32"
sys.argv = ["shadqa.py", ds_arg, "--workers", w_arg]
sys.path.insert(0, ROOT)
import shadqa as S  # noqa: E402
import shadqa2 as S2  # noqa: E402  (scope text, rephrase prompt, fold logic)

import numpy as np, pandas as pd  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

gem = S.gem
OUT3 = f"{S.OUT}/v3"; os.makedirs(OUT3, exist_ok=True)
gem.TAG.v = f"qa/{ds_arg}/shadqa3"

PROPOSE = """Below are questions with a model's answer and the reference answer (some correct, mostly wrong).

{pairs}

Propose {k} features that describe how the model's answers differ from the reference answers. Each feature must be a question that can be asked about ANY (question, answer) pair without seeing the reference, so it can be measured for the model's answer and for the reference answer in the same way. For each feature also say whether the person answering could change it WITHOUT changing what the answer says (actionable: true), e.g. its length, wording, format or scope, or whether it depends on what the right answer is (actionable: false).
Reply with only a JSON list: [{{"name": short name, "question": the question to ask about an answer, "type": "binary" or "count", "actionable": true or false}}]"""

COUNTS = {"number of words": lambda s: len(s.split()),
          "number of listed items": lambda s: len([x for x in re.split(r"\s*\|\s*|;\s+|,\s+(?![0-9]{3})|\s+and\s+", s) if x.strip()])}


def clean(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else re.sub(r"\s*\((thousand|million|billion|percent)\)$", "", str(x))


def ridge_anchor(X, y, k, folds=5):
    fold = np.arange(len(X)) % folds

    def fit(Xa, ya, lam):
        mu = ya.mean(); w0 = np.zeros(Xa.shape[1]); w0[k] = (Xa[:, k] @ (ya - mu)) / max(Xa[:, k] @ Xa[:, k], 1e-9)
        return mu, np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ (ya - mu) + lam * w0)
    lams = [1, 10, 100, 1000, 1e4]
    cv = [np.mean([np.mean((y[fold == f] - (lambda m, w: m + X[fold == f] @ w)(*fit(X[fold != f], y[fold != f], l))) ** 2)
                   for f in range(folds)]) for l in lams]
    return fit(X, y, lams[int(np.argmin(cv))])


def pct(v):
    return f"{100 * v:.0f}%"


def rules_from_decoder(F, O, G, wrong):
    names = list(O.columns); X, Y = O.to_numpy(float), G.to_numpy(float)
    ok = ~(np.isnan(X).any(1) | np.isnan(Y).any(1)); X, Y, wrong = X[ok], Y[ok], wrong[ok]
    mu, sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1); Z = (X - mu) / sd
    kinds = {f["name"]: f["type"] for f in F} | {c: "count" for c in COUNTS}
    qs = {f["name"]: f["question"] for f in F}
    out = []
    for k, nm in enumerate(names):
        d = Y[:, k] - X[:, k]; t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12)
        bias = d.mean() * t * t / (1 + t * t)
        _, w = ridge_anchor(Z, Y[:, k], k)
        cross = [(j, w[j]) for j in np.argsort(-np.abs(w)) if j != k][:1]
        mism = np.abs(d) > 1e-9 if kinds[nm] == "binary" else np.abs(d) >= 1
        rel = (wrong[mism].mean() - wrong[~mism].mean()) if mism.any() and (~mism).any() else 0.0
        size = abs(bias) / (Y[:, k].std() + 1e-9) + sum(abs(v) for _, v in cross) / (Y[:, k].std() + 1e-9)
        if kinds[nm] == "binary":
            s = (f"Reference answers are '{qs[nm]}' -> yes for {pct(Y[:, k].mean())} of questions, the model's answers for "
                 f"{pct(X[:, k].mean())}; so the final answer should {'more often' if bias > 0 else 'less often'} have this property")
        else:
            s = (f"Reference answers have on average {Y[:, k].mean():.1f} for '{nm}', the model's answers {X[:, k].mean():.1f}; "
                 f"so the final answer should have a {'higher' if bias > 0 else 'lower'} {nm}")
        for j, v in cross:
            if abs(v) / (Y[:, k].std() + 1e-9) >= .1:
                s += (f", especially when the model's answer is high on '{names[j]}'" if (v < 0) == (bias < 0)
                      else f", but less so when the model's answer is high on '{names[j]}'")
        out.append(dict(feature=nm, bias=float(bias), t=float(t), relevance=float(rel), weight=float(max(rel, 0) * size),
                        statement=s + "."))
    return sorted(out, key=lambda r: -r["weight"])


if __name__ == "__main__":
    TR, TE = S.sample("train", S.a.ntr), S.sample("test", S.a.nte)
    Atr, Ate = S.run(S.BASE, TR, "train_base"), S.run(S.BASE, TE, "test_base")
    fit, val = np.arange(len(Atr)) < 600, np.arange(len(Atr)) >= 600
    Afit = Atr[fit].reset_index(drop=True)
    # both sides as plain answer text: the scale label the scorer appends (TAT-QA) is removed from model and reference alike
    Afit = Afit.assign(pred=[clean(x) for x in Afit.pred], gold=[clean(x) for x in Afit.gold])

    ff = f"{OUT3}/features.json"
    if not os.path.exists(ff):
        txt = S.call([dict(role="user", content=PROPOSE.format(pairs=S.pairs_text(Afit), k=10))], 2500)
        json.dump(json.loads(re.findall(r"\[.*\]", txt, re.S)[-1]), open(ff, "w"), indent=1)
    Fall = json.load(open(ff)); F = [f for f in Fall if f.get("actionable")]
    print("features (kept = actionable):\n" + "\n".join(f"  [{'x' if f.get('actionable') else ' '}] {f['name']}: {f['question']}" for f in Fall), flush=True)

    pf = f"{OUT3}/probes_fit.csv"
    if not os.path.exists(pf):
        with ThreadPoolExecutor(S.a.workers) as ex:
            o = list(ex.map(lambda x: S.probe(F, x.question, x.pred), Afit.itertuples()))
            g = list(ex.map(lambda x: S.probe(F, x.question, clean(x.gold)), Afit.itertuples()))
        P = pd.concat([pd.DataFrame(o).add_prefix("out:"), pd.DataFrame(g).add_prefix("gt:")], axis=1)
        P.insert(0, "qid", Afit.qid); P.to_csv(pf, index=False)
    P = pd.read_csv(pf)
    O = pd.concat([P[[f"out:{f['name']}" for f in F]].set_axis([f["name"] for f in F], axis=1),
                   pd.DataFrame([{c: fn(clean(x)) for c, fn in COUNTS.items()} for x in Afit.pred])], axis=1)
    G = pd.concat([P[[f"gt:{f['name']}" for f in F]].set_axis([f["name"] for f in F], axis=1),
                   pd.DataFrame([{c: fn(clean(x)) for c, fn in COUNTS.items()} for x in Afit.gold])], axis=1)

    lf = f"{OUT3}/rules.json"
    if not os.path.exists(lf):
        L = rules_from_decoder(F, O, G, 1 - Afit.correct.to_numpy())
        for r in L:
            r["rule"] = "- " + (S.call([dict(role="user", content=S2.REPHRASE.format(s=r["statement"]))], 200) or r["statement"]).strip().lstrip("- ")
        json.dump(L, open(lf, "w"), indent=1)
    L = json.load(open(lf))
    print("\ndecoder rules (ordered by relevance x deviation):")
    for r in L:
        print(f"  weight {r['weight']:.3f} (relevance {r['relevance']:+.2f}, bias {r['bias']:+.2f}, t {r['t']:+.1f}) {r['feature']}\n      {r['rule']}")

    VR = [r for r, v in zip(TR, val) if v]; base_val = Atr[val].correct.mean()
    rules = [r["rule"] for r in L if r["weight"] > 0]
    best = (base_val, None, 0)
    for n in sorted({1, 2, 4, len(rules)} - {0}):
        inst = S.BASE + S2.SCOPE + "\n".join(rules[:n])
        em = S.run(inst, VR, f"v3_val_top{n}").correct.mean()
        print(f"[val] shad v3 top {n}: EM {100 * em:.1f} (base {100 * base_val:.1f})", flush=True)
        if em > best[0]:
            best = (em, inst, n)
    print(f"[val] chosen {'top ' + str(best[2]) if best[1] else 'base prompt (no candidate beats it)'}", flush=True)

    rng = np.random.default_rng(0); b = Ate.set_index("qid").correct
    df = Ate if best[1] is None else S.run(best[1], TE, "v3_test")
    c = df.set_index("qid").correct.loc[b.index]; d = c - b
    se = np.std([d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    E2 = {arm: pd.read_csv(f"{S.OUT}/answers_v2_test_{arm}.csv") for arm in ("shad", "errsum") if os.path.exists(f"{S.OUT}/answers_v2_test_{arm}.csv")}
    print(f"\n== test: base EM {100 * b.mean():.1f} | ShAD v3 EM {100 * c.mean():.1f} ({100 * d.mean():+.1f}, SE {100 * se:.1f}; "
          f"right->wrong {int(((b == 1) & (c == 0)).sum())}, wrong->right {int(((b == 0) & (c == 1)).sum())}) | "
          + " | ".join(f"v2 {k} EM {100 * v.correct.mean():.1f}" for k, v in E2.items()))
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

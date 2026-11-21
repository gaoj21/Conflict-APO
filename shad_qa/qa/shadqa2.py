"""ShAD for QA, v2: the decoder targets the metric itself, rules are generated from its weights by a template and
apply only to the final Answer line, and every rule set is chosen on a validation split.

  .venv/bin/python qa/shadqa2.py <tatqa|musique> [--workers 32]
Reuses round-1 artefacts of qa/shadqa.py (base answers, proposed features, probes of the model's answers).
  1. split the 800 train questions into fit (600) and val (200)
  2. decoder: P(wrong | answer features, question type), ridge logistic on standardized features, lambda by 5-fold CV
     on fit; answer features = the proposed probe features + universal deterministic ones
  3. one rule per answer-feature weight, written by a template from the weight's sign and the feature's mean among
     right vs wrong answers, ordered by |weight|; the LLM only rephrases each line (one call per line, nothing added)
  4. error-summary baseline: the LLM writes up to 8 rules from contrasting pairs, ordered by importance
  5. both arms: candidate rule sets = the top 1, 2, 4 and all rules, scoped to the final Answer line; the one with the
     best val EM is kept, and the base prompt is kept when no candidate beats it on val
  6. test: base vs ShAD vs error summary (EM, F1, right->wrong and wrong->right counts)
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ds_arg = sys.argv[1]; w_arg = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "32"
sys.argv = ["shadqa.py", ds_arg, "--workers", w_arg]  # reuse round 1's module (adapters, cached answers, probes)
sys.path.insert(0, ROOT)
import shadqa as S  # noqa: E402

import numpy as np, pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

gem = S.gem
OUT2 = f"{S.OUT}/v2"; os.makedirs(OUT2, exist_ok=True)
gem.TAG.v = f"qa/{ds_arg}/shadqa2"
SCOPE = ("\n\nReason as usual. The following rules apply only to how you write the final Answer line:\n")
WH = ["how many", "how much", "percentage", "change", "average", "when", "what year", "who", "where", "which", "what"]


def qfeats(q):
    q = q.lower()
    return {f"question asks '{w}'": float(w in q) for w in WH} | {"question length (words)": float(len(q.split()))}


def logistic(X, y, lam):
    def f(p):
        b, w = p[0], p[1:]; z = b + X @ w
        ll = np.logaddexp(0, z) - y * z
        g = (1 / (1 + np.exp(-z)) - y) / len(y)
        return ll.mean() + lam * w @ w, np.r_[g.sum(), X.T @ g + 2 * lam * w]
    return minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B").x


def fit_decoder(Xa, y):
    fold = np.arange(len(y)) % 5; lams = [1e-3, 1e-2, 3e-2, .1, .3]
    def ll(p, X, y):
        z = p[0] + X @ p[1:]; return np.mean(np.logaddexp(0, z) - y * z)
    cv = [np.mean([ll(logistic(Xa[fold != k], y[fold != k], l), Xa[fold == k], y[fold == k]) for k in range(5)]) for l in lams]
    lam = lams[int(np.argmin(cv))]
    return logistic(Xa, y, lam), lam, min(cv), np.mean([ll(np.r_[np.log(y.mean() / (1 - y.mean())), np.zeros(Xa.shape[1])], Xa, y)])


def strength(r):
    return "strongly" if r >= .5 else "clearly" if r >= .25 else "somewhat" if r >= .1 else "slightly"


REPHRASE = ("Rewrite this statement as one short instruction for how to write the final answer line. Keep its meaning "
            "exactly; do not add anything. Reply with only the instruction.\n\nStatement: {s}")


def shad_rules(F, Do, Q, wrong):
    names = list(Do.columns); qn = list(Q.columns)
    X = np.c_[Do.to_numpy(float), Q.to_numpy(float)]
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0); sd = np.where(sd > 0, sd, 1)
    Z = np.nan_to_num((X - mu) / sd)
    p, lam, cvll, base_ll = fit_decoder(Z, wrong)
    w = p[1:1 + len(names)]
    desc = {f["name"]: f for f in F}
    lines = []
    for j in np.argsort(-np.abs(w)):
        nm = names[j]; col = Do[nm].to_numpy(float)
        mr, mw = np.nanmean(col[wrong == 0]), np.nanmean(col[wrong == 1])
        what = desc[nm]["question"] if nm in desc else f"what is the {nm} of the answer?"
        kind = desc.get(nm, {}).get("type", "count")
        if kind == "binary":
            s = (f"Answers for which the answer to \"{what}\" is yes are {strength(abs(w[j]))} more often judged wrong "
                 f"(yes for {100 * mw:.0f}% of wrong vs {100 * mr:.0f}% of right answers), so final answers should "
                 f"{'avoid' if w[j] > 0 else 'prefer'} this.")
        else:
            s = (f"Final answers with a higher {nm} are {strength(abs(w[j]))} more often judged "
                 f"{'wrong' if w[j] > 0 else 'right'} (wrong answers average {mw:.2f}, right answers {mr:.2f}), so final "
                 f"answers should have a {'lower' if w[j] > 0 else 'higher'} {nm}.")
        lines.append(dict(feature=nm, weight=float(w[j]), statement=s))
    return lines, dict(lam=lam, cv_logloss=cvll, null_logloss=base_ll,
                       question_weights={qn[i]: float(p[1 + len(names) + i]) for i in range(len(qn))})


def evaluate_set(inst, R, tag):
    return S.run(inst, R, tag)


if __name__ == "__main__":
    TR, TE = S.sample("train", S.a.ntr), S.sample("test", S.a.nte)
    Atr, Ate = S.run(S.BASE, TR, "train_base"), S.run(S.BASE, TE, "test_base")
    F = json.load(open(f"{S.OUT}/features.json"))
    P = pd.read_csv(f"{S.OUT}/probes_train.csv")
    Do = pd.concat([P[[f"out:{f['name']}" for f in F]].set_axis([f["name"] for f in F], axis=1),
                    pd.DataFrame([S.det(x) for x in Atr.pred])], axis=1)
    Do = Do.drop(columns=["is a number", "is negative", "size of the number (log10)"])  # what the answer is, not how it is written
    Q = pd.DataFrame([qfeats(q) for q in Atr.question])
    fit, val = np.arange(len(Atr)) < 600, np.arange(len(Atr)) >= 600
    wrong = 1 - Atr.correct.to_numpy()

    lf = f"{OUT2}/shad_lines.json"
    if not os.path.exists(lf):
        lines, info = shad_rules(F, Do[fit].reset_index(drop=True), Q[fit].reset_index(drop=True), wrong[fit])
        for L in lines:
            L["rule"] = "- " + (S.call([dict(role="user", content=REPHRASE.format(s=L["statement"]))], 200) or L["statement"]).strip().lstrip("- ")
        json.dump(dict(lines=lines, info=info), open(lf, "w"), indent=1)
    SH = json.load(open(lf))
    print(f"decoder: lambda {SH['info']['lam']}, CV log loss {SH['info']['cv_logloss']:.4f} vs no-feature {SH['info']['null_logloss']:.4f}")
    for L in SH["lines"]:
        print(f"  w {L['weight']:+.2f}  {L['feature']}\n      {L['rule']}")

    ef = f"{OUT2}/errsum_rules.txt"
    if not os.path.exists(ef):
        pt = S.pairs_text(Atr[fit])
        txt = S.call([dict(role="user", content=S.RULES.format(example=Atr.question.iloc[0], evidence="Examples of the model's answers and the reference answers:\n\n" + pt)
                      + "\nOrder the rules from most to least important. The rules are only about how to write the final answer line.")], 1200)
        open(ef, "w").write("\n".join(l for l in (txt or "").splitlines() if l.strip().startswith("-")))
    ER = open(ef).read().splitlines()
    print("error-summary rules:\n" + "\n".join("  " + r for r in ER))

    VR = [r for r, v in zip(TR, val) if v]
    base_val = Atr[val].correct.mean()
    chosen = {}
    for arm, rules in (("shad", [L["rule"] for L in SH["lines"]]), ("errsum", ER)):
        best = (base_val, None, 0)
        for n in sorted({1, 2, 4, len(rules)}):
            inst = S.BASE + SCOPE + "\n".join(rules[:n])
            em = evaluate_set(inst, VR, f"v2_val_{arm}_top{n}").correct.mean()
            print(f"[val] {arm} top {n}: EM {100 * em:.1f} (base {100 * base_val:.1f})", flush=True)
            if em > best[0]:
                best = (em, inst, n)
        chosen[arm] = best
        print(f"[val] {arm}: chosen {'top ' + str(best[2]) if best[1] else 'base prompt (no candidate beats it)'}", flush=True)

    rng = np.random.default_rng(0); b = Ate.set_index("qid").correct
    print("\n== test (rule sets chosen on val)")
    for arm in ("base", "shad", "errsum"):
        if arm == "base" or chosen[arm][1] is None:
            df = Ate
        else:
            df = evaluate_set(chosen[arm][1], TE, f"v2_test_{arm}")
        c = df.set_index("qid").correct.loc[b.index]; d = c - b
        se = np.std([d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"{arm:7s} EM {100 * c.mean():.1f}  F1 {100 * df.f1.mean():.1f}  vs base {100 * d.mean():+.1f} (SE {100 * se:.1f})  "
              f"right->wrong {int(((b == 1) & (c == 0)).sum())}, wrong->right {int(((b == 0) & (c == 1)).sum())}"
              + ("" if arm == "base" else f"  [{'top ' + str(chosen[arm][2]) if chosen[arm][1] else 'base kept'}]"))
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

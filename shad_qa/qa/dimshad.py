"""ShAD with evaluation dimensions on QA (offline analysis, no val / test).

  QA_DS=<drop|musique|hotpot> .venv/bin/python qa/dimshad.py [--workers 32]
Input: the 1000 pool answers of TextGrad's final prompt (qa/drop_textgrad.py tg) with their references.
1. judge: the same model rates the model's answer A and the reference GT, each alone with the question (never both
   together), on K evaluation dimensions of how an answer is written (1-5), and tags the question type
2. ShAD: per dimension k, a ridge decoder GT_k ~ all K dimensions of A + log length of A (as ShAD on rating tasks:
   attribute = dimension, human rating = the reference's rating). The decoder's weights show which dimensions of A
   move together (shared axes); A_k - decoded GT_k is how A should change on dimension k
3. how much each deviation costs: P(wrong) against the absolute deviation, per dimension; and the mean deviation per
   question type
4. translation by template: for the dimensions whose deviation raises P(wrong) most, a global rule and, where the
   mean deviation differs by question type, a rule for that question type
Prints everything; writes dimshad/ratings.csv, dimshad/rules.json.
"""
import hashlib, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE) + "/baselines")
import gem  # noqa: E402

DS = os.environ.get("QA_DS", "drop")
W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 32
BASE = f"{gem.OUT}_qa/{DS}/shadqa" + ("_official" if DS == "drop" else "") + ("_devval" if os.environ.get("QA_VAL") == "dev" else ""); OUT = f"{BASE}/dimshad"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = f"qa/{DS}/dimshad"

DIMS = {  # name: (what 1 means, what 5 means) -- how the answer is written, each changeable by the answerer
    "verbosity": ("only the words needed to answer", "many words beyond what the question asks for"),
    "specificity": ("a general or descriptive phrase", "a specific named entity or exact value"),
    "granularity": ("coarse (e.g. a year, a country, a rounded number)", "fine (e.g. a full date, a town, an exact number)"),
    "multiplicity": ("a single item", "several items listed"),
    "qualification": ("a bare answer", "hedges, ranges, approximations or parenthetical additions"),
    "type_fit": ("not the kind of thing the question asks for", "exactly the kind of thing the question asks for"),
}
QTYPES = ["person", "place", "date or time", "number or quantity", "list of items", "event or thing", "yes/no", "other"]
RATE = """Question: {q}
Answer: {a}

Rate how this answer is written, from 1 to 5, on each dimension (do not judge whether it is correct):
{dims}
Also say what the question asks for, one of: {qt}.
Reply with only one line of JSON: {{{keys}, "asks": "<one of the kinds>"}}"""
PROMPT_DIMS = "\n".join(f"- {k}: 1 = {a}; 5 = {b}" for k, (a, b) in DIMS.items())
KEYS = ", ".join(f'"{k}": <1-5>' for k in DIMS)
PHRASE = {  # template phrases for the rules: (lower the dimension, raise it)
    "verbosity": ("use only the words needed to answer", "write out the full answer phrase"),
    "specificity": ("give the general term the question asks for, not a more specific one",
                    "give the specific named entity or exact value, not a descriptive phrase"),
    "granularity": ("give the answer at a coarser level (e.g. the year, the country)",
                    "give the answer at a finer level (e.g. the full date, the exact place)"),
    "multiplicity": ("give a single item", "list all the items asked for"),
    "qualification": ("give a bare answer without hedges, ranges or parenthetical additions",
                      "keep the qualifiers that are part of the answer"),
    "type_fit": ("", "give the kind of thing the question asks for (a person, a place, a date, ...); never reply that it is "
                     "not mentioned or cannot be determined, give your best answer"),
}


def rate(q, a):
    txt = gem.chat([dict(role="user", content=RATE.format(q=q, a=a if str(a).strip() else "(no answer)", dims=PROMPT_DIMS,
                                                          qt=", ".join(QTYPES), keys=KEYS))], max_tokens=200)[0]
    try:
        d = json.loads(re.findall(r"\{.*\}", txt or "", re.S)[-1])
        return {k: float(d[k]) for k in DIMS} | dict(asks=str(d.get("asks", "other")).lower())
    except Exception:  # noqa: BLE001
        return {k: np.nan for k in DIMS} | dict(asks="other")


if __name__ == "__main__":
    sysp = json.load(open(f"{BASE}/textgrad/tg.json"))["sys"]
    A = pd.read_csv(f"{BASE}/textgrad/answers_shadfit_{hashlib.sha1((sysp + chr(0)).encode()).hexdigest()[:8]}.csv")
    A["pred"] = A.pred.fillna("").astype(str); A["gold"] = A.gold.astype(str)
    rf = f"{OUT}/ratings.csv"
    if os.path.exists(rf):
        R = pd.read_csv(rf)
    else:
        with ThreadPoolExecutor(W) as ex:
            ra = list(ex.map(lambda r: rate(r[0], r[1]), zip(A.question, A.pred)))
            rg = list(ex.map(lambda r: rate(r[0], r[1]), zip(A.question, A.gold)))
        R = pd.concat([A[["qid", "question", "pred", "gold", "correct"]],
                       pd.DataFrame(ra).add_prefix("A_"), pd.DataFrame(rg).add_prefix("G_")], axis=1)
        R.to_csv(rf, index=False)
    R["pred"] = R.pred.fillna("").astype(str); R["gold"] = R.gold.fillna("").astype(str)  # empty answers read back as NaN
    K = list(DIMS); R = R.dropna(subset=[f"A_{k}" for k in K] + [f"G_{k}" for k in K]).reset_index(drop=True)
    R["qtype"] = R.G_asks.where(R.G_asks.isin(QTYPES), "other")
    if len(R) < 50:
        print(f"[{DS}] only {len(R)} answer/reference pairs have parsable ratings: stopping (check the judge replies in {rf})"); sys.exit(0)
    wrong = 1 - R.correct.to_numpy()
    print(f"[{DS}] {len(R)} rated answer/reference pairs, wrong {100 * wrong.mean():.1f}%  (spent ${gem.spent()[0]:.2f})")
    print("\nmean rating  A (model) vs GT (reference), and on wrong answers only:")
    for k in K:
        a, g = R[f"A_{k}"], R[f"G_{k}"]
        print(f"  {k:14s} A {a.mean():.2f}  GT {g.mean():.2f}  diff {a.mean() - g.mean():+.2f} | wrong: diff {(a - g)[wrong == 1].mean():+.2f}")

    # 2. ShAD decoder: GT_k ~ all dimensions of A + log length (standardized, ridge, alpha by CV)
    from sklearn.linear_model import RidgeCV, LogisticRegression
    X = np.column_stack([R[f"A_{k}"] for k in K] + [np.log1p(R.pred.str.len())]); names = K + ["log_len"]
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    print("\nShAD decoder  GT_k ~ A dimensions + log length  (standardized weights; R2 vs own-score-only R2):")
    print("  " + " " * 14 + "".join(f"{n[:9]:>10s}" for n in names))
    DEV = {}
    for k in K:
        y = R[f"G_{k}"].to_numpy(); m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xs, y)
        own = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xs[:, [K.index(k)]], y)
        print(f"  {k:14s}" + "".join(f"{w:10.2f}" for w in m.coef_) + f"   R2 {m.score(Xs, y):.2f} (own {own.score(Xs[:, [K.index(k)]], y):.2f})")
        DEV[k] = R[f"A_{k}"].to_numpy() - m.predict(Xs)  # how far A is from where the decoder puts the reference

    # 3. cost of each deviation: logistic P(wrong) ~ |deviation| above and below (standardized), all dimensions together
    Z = np.column_stack([np.clip(DEV[k], 0, None) for k in K] + [np.clip(-DEV[k], 0, None) for k in K])
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Zs, wrong)
    cost = {(k, "too high"): lr.coef_[0][i] for i, k in enumerate(K)} | {(k, "too low"): lr.coef_[0][len(K) + i] for i, k in enumerate(K)}
    print("\ncost of a deviation: logistic weight of P(wrong) on how far A is above / below the decoded reference")
    for (k, s), c in sorted(cost.items(), key=lambda x: -x[1]):
        e = DEV[k] > .5 if s == "too high" else DEV[k] < -.5
        print(f"  {k:14s} {s:9s} weight {c:+.2f}   share of answers {100 * e.mean():4.1f}%   P(wrong | it) {wrong[e].mean() if e.any() else float('nan'):.2f}")
    print("\nmean deviation A - decoded GT by question type (n):")
    tab = pd.DataFrame({k: DEV[k] for k in K}).assign(qtype=R.qtype).groupby("qtype")
    print(pd.concat([tab.mean().round(2), tab.size().rename("n")], axis=1).sort_values("n", ascending=False).to_string())

    # 4. translation: the 3 costliest deviations; a global rule and a rule for the question type where it is largest
    rules = []
    for (k, s), c in sorted(cost.items(), key=lambda x: -x[1])[:3]:
        if c <= 0:
            continue
        act = PHRASE[k][0] if s == "too high" else PHRASE[k][1]
        if not act:
            continue
        rules.append(dict(dim=k, side=s, kind="global", rule=f"- In the final answer, {act}."))
        m = tab.mean()[k]; m = m[tab.size() >= 30]
        qt = (m.idxmax() if s == "too high" else m.idxmin()) if len(m) else None
        if qt is not None:
            rules.append(dict(dim=k, side=s, kind="question type", qtype=qt,
                              rule=f"- For questions that ask for a {qt}: in the final answer, {act}. For other questions, write the final answer as before."))
    json.dump(rules, open(f"{OUT}/rules.json", "w"), indent=1)
    # all deviations ranked by how much error they explain (cost weight x share of answers), as global rules: the
    # candidates of the round-by-round combination (qa/drop_dimshad_eval.py --combine)
    ranked = []
    for (k, s), c in cost.items():
        act = PHRASE[k][0] if s == "too high" else PHRASE[k][1]
        share = float((DEV[k] > .5).mean() if s == "too high" else (DEV[k] < -.5).mean())
        if act and c > 0 and share > 0:
            ranked.append(dict(dim=k, side=s, weight=float(c), share=share, score=float(c) * share, rule=f"- In the final answer, {act}."))
    ranked.sort(key=lambda r: -r["score"])
    json.dump(ranked, open(f"{OUT}/ranked.json", "w"), indent=1)
    print("\nranked global rules (cost weight x share):\n" + "\n".join(f"  {r['score']:.3f}  [{r['dim']} {r['side']}] {r['rule']}" for r in ranked))
    print("\ntranslated rules:\n" + "\n".join(f"  [{r['dim']} {r['side']}, {r['kind']}] {r['rule']}" for r in rules))
    print(f"done, spent ${gem.spent()[0]:.2f}")

"""Edit-operation features of answers vs references -> question n-gram conditions -> templated rules (no LLM).
Used by qa/drop_editfeat.py (one step on TextGrad's final prompt) and qa/drop_textgrad.py arm tg+edit (every few steps)."""
import re

import numpy as np, pandas as pd

DESC = {"pct": ("a % sign", "the % sign"), "dollar": ("a $ sign", "the $ sign"), "num": ("extra numbers", "all the numbers asked for"),
        "commanum": ("commas inside numbers", "commas inside numbers"), "unit": ("a unit word after the number (e.g. 'yards', 'years')",
        "the unit word after the number (e.g. 'yards', 'years')"), "cap": ("extra capitalized words such as names or titles",
        "the capitalized words such as full names or titles"), "word": ("extra words", "the descriptive words"),
        "paren": ("parentheses", "parentheses"), "sep": ("several items separated by commas, 'and' or '|'", "all the items asked for")}


def tokens(s):
    toks = re.findall(r"\d[\d,]*\.?\d*|[A-Za-z][A-Za-z'\-]*|[%$()|;]|,(?!\d)", str(s))
    out, prev_num = [], False
    for t in toks:
        if t == "%": c = "pct"
        elif t == "$": c = "dollar"
        elif t in "()": c = "paren"
        elif t in ("|", ";", ",") or t.lower() == "and": c = "sep"
        elif re.match(r"\d", t): c = "commanum" if "," in t else "num"
        elif prev_num: c = "unit"
        elif t[0].isupper(): c = "cap"
        else: c = "word"
        out.append((c, t.lower())); prev_num = c in ("num", "commanum")
    return out


def edits(pred, gold):
    from collections import Counter
    p, g = Counter(tokens(pred)), Counter(tokens(gold)); f = {}
    for c in DESC:
        add = sum(max(n - g.get(k, 0), 0) for k, n in p.items() if k[0] == c)
        drop = sum(max(n - p.get(k, 0), 0) for k, n in g.items() if k[0] == c)
        # one-directional edits are about how the answer is written; a class both added and dropped is a replaced value
        f[f"added {c}"], f[f"dropped {c}"] = float(add > 0 and drop == 0), float(drop > 0 and add == 0)
        f[f"replaced {c}"] = float(add > 0 and drop > 0)
    return f


def candidates(A, log=print):
    """Answers A (pred, gold, correct, question) -> templated rules for the 3 top-ranked form edits: a global rule and an
    "only for questions containing ..." rule each."""
    A = A.assign(pred=A.pred.fillna("").astype(str), gold=A.gold.astype(str))
    E = pd.DataFrame([edits(p, g) for p, g in zip(A.pred, A.gold)]); wrong = 1 - A.correct.to_numpy()
    rows = []
    for c in E.columns:
        e = E[c].to_numpy() > 0
        if e.sum() == 0 or (~e).sum() == 0:
            continue
        rows.append(dict(edit=c, freq_wrong=float(e[wrong == 1].mean()), relevance=float(wrong[e].mean() - wrong[~e].mean()), n=int(e.sum())))
    R = pd.DataFrame(rows).assign(score=lambda d: d.freq_wrong * d.relevance.clip(lower=0)).sort_values("score", ascending=False)
    R = R.assign(form=~R.edit.str.startswith("replaced"))  # a replaced value is what the answer says, not how it is written
    log("edit features (freq among wrong answers, relevance = P(wrong|edit) - P(wrong|no edit)):\n" + R.round(3).to_string(index=False), flush=True)

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegressionCV
    cv = CountVectorizer(ngram_range=(1, 2), min_df=10, binary=True, stop_words=None)
    Xq = cv.fit_transform(A.question.str.lower()); vocab = np.array(cv.get_feature_names_out())
    rules = []
    for _, r in R[R.form].head(3).iterrows():
        y = E[r.edit].to_numpy() > 0
        lr = LogisticRegressionCV(Cs=[.03, .1, .3, 1], penalty="l1", solver="liblinear", cv=5, scoring="neg_log_loss").fit(Xq, y)
        w = lr.coef_[0]; top = [vocab[i] for i in np.argsort(-w)[:2] if w[i] > 0]
        kind, c = r.edit.split()
        what = DESC[c][0] if kind == "added" else DESC[c][1]
        act = f"do not add {what} to the final answer" if kind == "added" else f"keep {what} in the final answer"
        rules.append(dict(edit=r.edit, kind="global", rule=f"- {act[0].upper() + act[1:]}."))
        if top:
            q = " or ".join(f"'{t}'" for t in top)
            rules.append(dict(edit=r.edit, kind="only when", cond=top,
                              rule=f"- Only for questions containing {q}: {act}. For all other questions, write the final answer as before."))
        log(f"  {r.edit}: condition n-grams {top}", flush=True)
    return rules

"""Mixed QA with conflicting answer conventions: one prompt answers DROP, FinQA, TAT-QA and MuSiQue questions, each
scored by its own official rule. The same answer property (a % sign, a short answer) is right for some sources and wrong
for others, so any global rule helps some and hurts others; rules conditioned on the question/context can resolve it.

  .venv/bin/python qa/mixqa.py [--ntr 300] [--nte 200] [--workers 32]
1. base answers ("Think step by step.") on train (per source) and test (per source)
2. features, proposed automatically from mixed contrasting pairs:
   answer features (how the answer is written, actionable only) + universal counts (words, items, % sign)
   context features (what about the question / context decides how the reference answer is written)
   the same probe prompts rate them for every train question and model answer
3. decoder (v2 style): P(wrong) ~ ridge logistic on answer features, context features and answer x context products,
   lambda by 5-fold CV on fit (3/4 of train); weights -> one templated statement each (global for main effects,
   conditional for products), ordered by |weight|; the LLM only rephrases each line; rules apply to the Answer line
4. error-summary baseline from the same mixed pairs; both arms choose top 1/2/4/8/all rules on val (1/4 of train)
5. test: overall EM and per source change vs base, with paired bootstrap SE; conflict = a source significantly up and
   another significantly down
"""
import argparse, importlib.util, json, math, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, f"{ROOT}/baselines"); sys.path.insert(0, HERE)
import gem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ntr", type=int, default=300); ap.add_argument("--nte", type=int, default=200)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--docctx", action="store_true", help="context features must also describe the document itself")
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/mixqa" + ("/docctx" if a.docctx else ""); os.makedirs(OUT, exist_ok=True)
gem.TAG.v = "qa/mixqa"
BASE = "Think step by step."
SCOPE = "\n\nReason as usual. The following rules apply only to how you write the final Answer line:\n"


def adapter(ds):
    """The per-source adapter of qa/shadqa.py, loaded as its own module instance (its argparse reads sys.argv)."""
    argv = sys.argv; sys.argv = ["shadqa.py", ds]
    spec = importlib.util.spec_from_file_location(f"shadqa_{ds}", f"{HERE}/shadqa.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); sys.argv = argv
    return m


# ---------------- FinQA: official answer = the executed program value (percentages as fractions)
def finqa_rows(split):
    d = json.load(open(f"{ROOT}/data/finqa/{'train' if split == 'train' else 'test'}.json"))
    out = []
    for x in d:
        if not str(x["qa"]["answer"]).strip():
            continue
        table = "\n".join(" | ".join(c for c in r) for r in x["table"])
        ctx = " ".join(x["pre_text"]) + "\n\n" + table + "\n\n" + " ".join(x["post_text"])
        out.append(dict(qid=x["id"], question=x["qa"]["question"], ctx=ctx, exe=x["qa"]["exe_ans"], gold_text=str(x["qa"]["answer"])))
    return out


def finqa_prompt(r, inst):  # same text as qa/conv_screen.py, so its calls are cached
    return f"## Context\n{r['ctx']}\n\n## Question\n{r['question']}\n\n{inst.strip()}\n\nEnd with exactly one line: Answer: <the answer>"


def finqa_value(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", str(s))
    if not m:
        return None
    v = float(m[0].replace(",", ""))
    if re.search(r"\(\s*\$?\s*\d", str(s)) and not str(s).strip().startswith("-"):
        v = -abs(v)
    return v / 100 if "%" in str(s) else v


def finqa_score(txt, r):
    m = re.findall(r"Answer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
    if not m:
        return 0, 0.0, None
    p = m[-1].strip()
    if isinstance(r["exe"], str):
        c = int(p.lower().strip(" .") .startswith(r["exe"].lower()))
    else:
        v, g = finqa_value(p), float(r["exe"])
        c = int(v is not None and abs(v - g) <= max(.01 * abs(g), 1e-4))
    return c, float(c), p


SRC = {}
for ds in ("drop", "tatqa", "musique"):
    m = adapter(ds); SRC[ds] = dict(m=m, rows=m.rows, prompt=m.make_prompt, score=m.score)
SRC["finqa"] = dict(rows=finqa_rows, prompt=finqa_prompt, score=finqa_score)


def sample(ds, split, n):
    R = SRC[ds]["rows"](split)
    k = 800 if (split == "train" and ds != "finqa") else (400 if ds != "finqa" else (300 if split == "test" else n))
    idx = np.random.default_rng(0).permutation(len(R))[:k]  # the same draw as the earlier runs (their calls are cached)
    return [dict(R[i], source=ds) for i in idx[:n]]


def call(content, mt=3000, temperature=0.0):
    try:
        return gem.chat([dict(role="user", content=content)], max_tokens=mt, temperature=temperature)[0]
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001
        return None


def plain(p):
    return "" if p is None or (isinstance(p, float) and math.isnan(p)) else re.sub(r"\s*\((thousand|million|billion|percent)\)$", "", str(p))


def run(inst, R, tag):
    import hashlib
    f = f"{OUT}/answers_{tag}_{hashlib.sha1(inst.encode()).hexdigest()[:8]}.csv"  # the prompt is part of the name: no stale reuse
    if os.path.exists(f):
        return pd.read_csv(f)

    def one(r):
        s = SRC[r["source"]]
        txt = call(s["prompt"](r, inst))
        c, f1, pred = s["score"](txt, r)
        return dict(qid=r["qid"], source=r["source"], question=r["question"], pred=pred, gold=r["gold_text"], correct=c)
    with ThreadPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(ex.map(one, R))
    df.to_csv(f, index=False)
    print(f"[{tag}] EM {100 * df.correct.mean():.1f}  " + ", ".join(f"{s} {100 * g.correct.mean():.1f}" for s, g in df.groupby("source")), flush=True)
    return df


def pairs(df, n=48):
    rng = np.random.default_rng(0); out = []
    for s, g in df.groupby("source"):
        w = g[g.correct == 0]; r = g[g.correct == 1]
        out += [w.iloc[i] for i in rng.permutation(len(w))[:n * 3 // 16]] + [r.iloc[i] for i in rng.permutation(len(r))[:n // 16]]
    return "\n\n".join(f"Question: {x.question[:300]}\nModel answer: {plain(x.pred)}\nReference answer: {x.gold}\n({'correct' if x.correct else 'wrong'})"
                       for x in [out[i] for i in rng.permutation(len(out))])


PROP_A = """Below are questions from several kinds of documents with a model's answer and the reference answer.

{pairs}

Propose 8 features of HOW AN ANSWER IS WRITTEN (its form: length, units or symbols, sentence or fragment, extent of the span, list or single item, wording) on which the model's answers differ from the reference answers. Each must be a question that can be asked about any answer without seeing the reference. Only features the answerer can change without changing what the answer says.
Reply with only a JSON list: [{{"name": short name, "question": the question to ask about an answer, "type": "binary" or "count"}}]"""

PROP_C = """Below are questions from several kinds of documents with a model's answer and the reference answer.

{pairs}

How the reference answers are written depends on the question and the document (for example whether a percentage keeps its % sign, or how long a quoted span is). Propose 6 features of THE QUESTION AND ITS DOCUMENT (not of the answer) that decide how the reference answer is written. Each must be a yes/no question that can be asked about any question with its document.{doc}
Reply with only a JSON list: [{{"name": short name, "question": the question to ask"}}]"""
DOC = (" At least three of them must be about the document itself rather than the question: what kind of document it is "
       "(its genre, source or domain), whether it contains tables of numbers, and how numbers are presented in it.")

PROBE = """{item}

Rate it on each feature below: 1 or 0 for a yes/no feature, a non-negative integer for a count. Reply with only one line of JSON mapping each name to its value.
{feats}"""


def probe(F, item):
    feats = "\n".join(f'- "{f["name"]}": {f["question"]}' + (f" ({f['type']})" if "type" in f else " (yes/no)") for f in F)
    txt = call(PROBE.format(item=item, feats=feats), 400)
    try:
        d = json.loads(re.findall(r"\{.*\}", txt or "", re.S)[-1])
        return {f["name"]: float(d.get(f["name"], np.nan)) for f in F}
    except Exception:  # noqa: BLE001
        return {f["name"]: np.nan for f in F}


def doc(r):
    return r.get("ctx") or (r.get("table", "") + "\n\n" + r.get("paras", ""))  # TAT-QA keeps table and text apart


def counts(p):
    s = plain(p)
    return {"number of words": float(len(s.split())), "has a % sign": float("%" in s),
            "number of listed items": float(len([x for x in re.split(r"\s*\|\s*|;\s+|,\s+(?![0-9]{3})|\s+and\s+", s) if x.strip()]))}


def logistic(X, y, lam):
    from scipy.optimize import minimize

    def f(p):
        z = p[0] + X @ p[1:]; g = (1 / (1 + np.exp(-z)) - y) / len(y)
        return np.mean(np.logaddexp(0, z) - y * z) + lam * p[1:] @ p[1:], np.r_[g.sum(), X.T @ g + 2 * lam * p[1:]]
    return minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B").x


def decoder(A, C, wrong):
    an, cn = list(A.columns), list(C.columns)
    Xa = A.to_numpy(float); Xc = C.to_numpy(float)
    Xa = np.where(np.isnan(Xa), np.nanmean(Xa, 0), Xa); Xc = np.where(np.isnan(Xc), np.nanmean(Xc, 0), Xc)
    za = (Xa - Xa.mean(0)) / np.maximum(Xa.std(0), 1e-3); zc = (Xc - Xc.mean(0)) / np.maximum(Xc.std(0), 1e-3)
    inter = np.einsum("ni,nj->nij", za, zc).reshape(len(za), -1)
    X = np.c_[za, zc, inter]
    names = [("main", i, None) for i in range(len(an))] + [("ctx", None, j) for j in range(len(cn))] + \
            [("inter", i, j) for i in range(len(an)) for j in range(len(cn))]
    fold = np.arange(len(X)) % 5; lams = [1e-3, 3e-3, 1e-2, 3e-2, .1]
    ll = lambda p, X, y: np.mean(np.logaddexp(0, p[0] + X @ p[1:]) - y * (p[0] + X @ p[1:]))
    cv = [np.mean([ll(logistic(X[fold != k], wrong[fold != k], l), X[fold == k], wrong[fold == k]) for k in range(5)]) for l in lams]
    lam = lams[int(np.argmin(cv))]; p = logistic(X, wrong, lam)
    null = ll(np.r_[np.log(wrong.mean() / (1 - wrong.mean())), np.zeros(X.shape[1])], X, wrong)
    return [(names[i], p[1 + i]) for i in range(len(names))], dict(lam=lam, cv=min(cv), null=null)


def statements(W, FA, FC, A, C, wrong):
    qa = {f["name"]: f for f in FA}; qc = {f["name"]: f for f in FC}; an, cn = list(A.columns), list(C.columns)
    out = []
    for (kind, i, j), w in sorted(W, key=lambda t: -abs(t[1])):
        if kind == "ctx":
            continue
        nm = an[i]
        DET = {"number of words": "the answer has more words", "has a % sign": "the answer contains a % sign",
               "number of listed items": "the answer lists more items"}
        prop = DET.get(nm) or (f"the answer to \"{qa[nm]['question']}\" is yes" if qa[nm].get("type") == "binary"
                               else f"the answer's {nm} is higher")
        verb = "avoid answers where" if w > 0 else "prefer answers where"
        if kind == "main":
            s = f"In general, {verb} {prop} (such answers are {'more' if w > 0 else 'less'} often judged wrong)."
        else:
            cond = qc[cn[j]]["question"]
            s = (f"Only when the answer to \"{cond}\" is yes for the document: {verb} {prop} (there such answers are "
                 f"{'more' if w > 0 else 'less'} often judged wrong). Say nothing about the other case.")  # one-sided
        out.append(dict(kind=kind, answer_feature=nm, context=None if j is None else cn[j], weight=float(w), statement=s))
    return out


REPHRASE = ("Rewrite this statement as one short, clear instruction for how to write the final answer line. Keep its meaning "
            "and its condition exactly; do not add anything. Reply with only the instruction.\n\nStatement: {s}")
RULES = """A model answers questions about several kinds of documents (narrative passages, financial reports with tables, multi-paragraph reading). Below are examples of its answers and the reference answers.

{pairs}

Write at most 8 short rules for how to write the final answer line so that it matches the reference answers better. Order them from most to least important. Reply with only the rules, one per line, each starting with "- "."""


def per_source(df, base):
    rng = np.random.default_rng(0); out = {}
    m = df.merge(base, on=["qid", "source"], suffixes=("", "_b"))
    for s, g in m.groupby("source"):
        d = (g.correct - g.correct_b).to_numpy()
        se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        out[s] = (100 * d.mean(), 100 * se, bool(se > 0 and abs(d.mean()) > 2 * se))
    return out


if __name__ == "__main__":
    srcs = ["drop", "finqa", "tatqa", "musique"]
    TR = [r for s in srcs for r in sample(s, "train", a.ntr)]; TE = [r for s in srcs for r in sample(s, "test", a.nte)]
    Atr, Ate = run(BASE, TR, "train_base"), run(BASE, TE, "test_base")
    per = Atr.groupby("source").cumcount(); fit, val = (per < a.ntr * 3 // 4).to_numpy(), (per >= a.ntr * 3 // 4).to_numpy()
    Af = Atr[fit].reset_index(drop=True); pt = pairs(Af)
    rows_by = {(r["source"], r["qid"]): r for r in TR}

    ff = f"{OUT}/features.json"
    if not os.path.exists(ff):
        FA = json.loads(re.findall(r"\[.*\]", call(PROP_A.format(pairs=pt), 2000), re.S)[-1])[:8]
        FC = json.loads(re.findall(r"\[.*\]", call(PROP_C.format(pairs=pt, doc=DOC if a.docctx else ""), 2000), re.S)[-1])[:6]
        json.dump(dict(answer=FA, context=FC), open(ff, "w"), indent=1)
    FF = json.load(open(ff)); FA, FC = FF["answer"], FF["context"]
    FA = [f for f in FA if "reference" not in f["question"].lower()]  # a feature must be measurable without the reference
    print("answer features:\n" + "\n".join(f"  {f['name']}: {f['question']}" for f in FA))
    print("context features:\n" + "\n".join(f"  {f['name']}: {f['question']}" for f in FC), flush=True)

    pf = f"{OUT}/probes_fit.csv"
    if not os.path.exists(pf):
        with ThreadPoolExecutor(a.workers) as ex:
            pa = list(ex.map(lambda x: probe(FA, f"Question: {x.question}\nAnswer: {plain(x.pred)}"), Af.itertuples()))
            pc = list(ex.map(lambda x: probe(FC, f"Document (beginning):\n{doc(rows_by[(x.source, x.qid)])[:1500]}\n\nQuestion: {x.question}"),
                             Af.itertuples()))
        P = pd.concat([pd.DataFrame(pa).add_prefix("a:"), pd.DataFrame(pc).add_prefix("c:")], axis=1)
        P.insert(0, "qid", Af.qid); P.to_csv(pf, index=False)
    P = pd.read_csv(pf)
    A = pd.concat([P[[f"a:{f['name']}" for f in FA]].set_axis([f["name"] for f in FA], axis=1),
                   pd.DataFrame([counts(p) for p in Af.pred])], axis=1)
    C = P[[f"c:{f['name']}" for f in FC]].set_axis([f["name"] for f in FC], axis=1)
    wrong = 1 - Af.correct.to_numpy()

    lf = f"{OUT}/shad_rules.json"
    if not os.path.exists(lf):
        W, info = decoder(A, C, wrong)
        L = statements(W, FA, FC, A, C, wrong)
        for x in L[:12]:
            x["rule"] = "- " + (call(REPHRASE.format(s=x["statement"]), 200) or x["statement"]).strip().lstrip("- ")
        json.dump(dict(info=info, rules=L[:12]), open(lf, "w"), indent=1)
    SH = json.load(open(lf))
    print(f"\ndecoder: lambda {SH['info']['lam']}, CV log loss {SH['info']['cv']:.4f} vs no-feature {SH['info']['null']:.4f}")
    for x in SH["rules"]:
        print(f"  {x['weight']:+.2f} [{x['kind']}] {x['answer_feature']}" + (f" x {x['context']}" if x['context'] else "") + f"\n      {x['rule']}")

    ef = f"{OUT}/errsum_rules.txt"
    if not os.path.exists(ef):
        txt = call(RULES.format(pairs=pt), 1200)
        open(ef, "w").write("\n".join(l for l in (txt or "").splitlines() if l.strip().startswith("-")))
    ER = open(ef).read().splitlines()
    print("error-summary rules:\n" + "\n".join("  " + r for r in ER), flush=True)

    VR = [r for r, v in zip(TR, val) if v]; bv = Atr[val].correct.mean(); chosen = {}
    for arm, rules in (("shad", [x["rule"] for x in SH["rules"]]), ("errsum", ER)):
        best = (bv, None, 0)
        for n in sorted({1, 2, 4, 8, len(rules)}):
            if n > len(rules):
                continue
            em = run(BASE + SCOPE + "\n".join(rules[:n]), VR, f"val_{arm}_top{n}").correct.mean()
            print(f"[val] {arm} top {n}: EM {100 * em:.1f} (base {100 * bv:.1f})", flush=True)
            if em > best[0]:
                best = (em, BASE + SCOPE + "\n".join(rules[:n]), n)
        chosen[arm] = best
        print(f"[val] {arm}: chosen {'top ' + str(best[2]) if best[1] else 'base prompt'}", flush=True)

    print("\n== test (per source: change vs base x100, SE; * beyond 2 SE)")
    for arm in ("base", "shad", "errsum"):
        df = Ate if arm == "base" or chosen[arm][1] is None else run(chosen[arm][1], TE, f"test_{arm}")
        ps = per_source(df, Ate)
        up = [s for s, (d, se, sig) in ps.items() if sig and d > 0]; dn = [s for s, (d, se, sig) in ps.items() if sig and d < 0]
        print(f"{arm:7s} EM {100 * df.correct.mean():.1f}  " + "  ".join(f"{s} {d:+.1f}({se:.1f}){'*' if sig else ''}" for s, (d, se, sig) in ps.items())
              + f"  | up {up or '-'} down {dn or '-'}{'  CONFLICT' if up and dn else ''}")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

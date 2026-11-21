"""ShAD for QA, dataset-agnostic: compare named features of the model's answers with the same features of the gold
answers, fit a small decoder of the systematic deviations, translate its weights into rules for the prompt.

Round 1 (this script):
  A. answer train (N_TR) and test (N_TE) questions with the base instruction
  B. feature proposal: the LLM reads contrasting (model answer, gold answer) pairs from train and proposes K named,
     checkable features, each a question that can be asked of any (question, answer)
  C. one probe prompt rates every feature for the model answer and, separately, for the gold answer (same prompt, so
     both sides are measured the same way); plus universal deterministic features (words, items, number, sign, size)
  D. decoder: per feature k, gold_k ~ ridge on all model-answer features (anchored to the own feature, lambda by
     5-fold CV); mean bias shrunk smoothly by its t statistic t^2/(1+t^2); cross weights = conditional deviations
  E. the decoder is described in words (deterministic), the LLM writes rules from the description
     baseline: the LLM writes rules directly from the same contrasting pairs (error summarization, ExpeL-style)
  F. test: base vs + ShAD rules vs + error-summary rules (EM, F1)
  .venv/bin/python qa/shadqa.py <tatqa|musique> [--ntr 800] [--nte 400] [--workers 32]
"""
import argparse, json, math, os, re, string, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/baselines"); sys.path.insert(0, f"{ROOT}/qa")
import gem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds", choices=["tatqa", "musique", "drop", "hotpot"]); ap.add_argument("--ntr", type=int, default=800)
ap.add_argument("--nte", type=int, default=400); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--probes-only", action="store_true", help="stop after the probes (later rounds use them)")
a = ap.parse_args()
OUT = f"{gem.OUT}_qa/{a.ds}/shadqa"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = f"qa/{a.ds}/shadqa"
BASE = "Think step by step."


# ---------------- dataset adapters: rows, prompt, score (evaluation protocol only, not part of the method)
def snorm(s):
    s = "".join(ch for ch in str(s).lower() if ch not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def sf1(p, g):
    p, g = snorm(p).split(), snorm(g).split()
    c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if not c else 2 * c / (len(p) + len(g))


if a.ds == "tatqa":
    import tatqa as T

    def rows(split):
        R = T.load("train" if split == "train" else "dev")
        for r in R:
            g = r["gold"] if isinstance(r["gold"], list) else [r["gold"]]
            r["gold_text"] = " | ".join(map(str, g)) + (f" ({r['scale']})" if r["scale"] else "")
        return R

    def make_prompt(r, inst):
        return T.prompt(r, inst)

    def score(txt, r):
        c, pred, f = T.score(txt, r)
        sc = re.findall(r"Scale\s*[:：]\s*([A-Za-z]+)", (txt or "").replace("**", ""))
        sc = sc[-1].lower() if sc else "none"
        return c, f, None if pred is None else pred + ("" if sc in ("none", "") else f" ({sc})")
elif a.ds == "drop":
    def dnorm(text):
        """Official DROP answer normalization: per token (split on spaces and hyphens) lowercase, drop punctuation
        unless the token is a number, numbers as floats, drop articles."""
        def isnum(t):
            try:
                float(t.replace(",", "")); return True
            except ValueError:
                return False
        out = []
        for t in re.split(r" |-", str(text)):
            t = t.lower()
            if isnum(t):
                t = str(float(t.replace(",", "")))
            else:
                t = "".join(ch for ch in t if ch not in string.punctuation)
            t = re.sub(r"\b(a|an|the)\b", " ", t).strip()
            if t:
                out.append(t)
        return " ".join(out)

    def rows(split):
        from huggingface_hub import hf_hub_download
        d = pd.read_parquet(hf_hub_download("ucinlp/drop", f"data/{'train' if split == 'train' else 'validation'}-00000-of-00001.parquet",
                                            repo_type="dataset"))
        return [dict(qid=r.query_id, question=r.question, ctx=r.passage, golds=list(dict.fromkeys(r.answers_spans["spans"])),
                     gold_text=r.answers_spans["spans"][0]) for r in d.itertuples() if len(r.answers_spans["spans"])]

    def make_prompt(r, inst):
        return f"## Passage\n{r['ctx']}\n\n## Question\n{r['question']}\n\n{inst.strip()}\n\nEnd with exactly one line: Answer: <the answer>"

    def score(txt, r):
        m = re.findall(r"Answer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
        if not m:
            return 0, 0.0, None
        p = m[-1].strip()
        return int(any(dnorm(p) == dnorm(g) for g in r["golds"])), max(sf1(dnorm(p), dnorm(g)) for g in r["golds"]), p
elif a.ds == "hotpot":
    def rows(split):  # distractor setting: 2 gold + 8 distractor Wikipedia paragraphs; official EM normalization = snorm
        from datasets import load_dataset
        d = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train" if split == "train" else "validation")
        return [dict(qid=x["id"], question=x["question"], golds=[x["answer"]], gold_text=x["answer"], qtype=x["type"],
                     ctx="\n\n".join(f"# {t}\n{''.join(s)}" for t, s in zip(x["context"]["title"], x["context"]["sentences"])))
                for x in d]

    def make_prompt(r, inst):
        return (f"## Paragraphs\n{r['ctx']}\n\n## Question\n{r['question']}\n\n{inst.strip()}\n\n"
                "End with exactly one line: Answer: <a short answer>")

    def score(txt, r):
        m = re.findall(r"Answer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
        if not m:
            return 0, 0.0, None
        p = m[-1].strip()
        return int(any(snorm(p) == snorm(g) for g in r["golds"])), max(sf1(p, g) for g in r["golds"]), p
else:
    def rows(split):
        from datasets import load_dataset
        d = load_dataset("dgslibisey/MuSiQue", split="train" if split == "train" else "validation")
        out = []
        for x in d:
            if not x["answerable"]:
                continue
            ctx = "\n\n".join(f"# {p['title']}\n{p['paragraph_text']}" for p in x["paragraphs"])
            out.append(dict(qid=x["id"], question=x["question"], ctx=ctx, golds=[x["answer"]] + list(x["answer_aliases"]),
                            gold_text=x["answer"]))
        return out

    def make_prompt(r, inst):
        return (f"## Paragraphs\n{r['ctx']}\n\n## Question\n{r['question']}\n\n{inst.strip()}\n\n"
                "End with exactly one line: Answer: <a short answer>")

    def score(txt, r):
        m = re.findall(r"Answer\s*[:：]\s*(.+)", (txt or "").replace("**", ""))
        if not m:
            return 0, 0.0, None
        p = m[-1].strip()
        return int(any(snorm(p) == snorm(g) for g in r["golds"])), max(sf1(p, g) for g in r["golds"]), p


KEEP = dict(train=4100, test=1000)  # rows kept in data/qa/: enough for every pool of the QA experiments


def sample(split, n):
    """The first n rows of a fixed (seed-0) permutation of the split. For DROP / MuSiQue / HotpotQA the first KEEP[split]
    rows are stored in data/qa/<ds>_<split>.jsonl.gz the first time, so later runs (and machines without HuggingFace
    access) read them from there."""
    import gzip
    f = f"{ROOT}/data/qa/{a.ds}_{split}.jsonl.gz"
    if a.ds in ("drop", "musique", "hotpot") and os.path.exists(f):
        with gzip.open(f, "rt") as fh:
            R = [json.loads(l) for l in fh]
        if len(R) >= n:
            return R[:n]
    R = rows(split)
    R = [R[i] for i in np.random.default_rng(0).permutation(len(R))[:max(n, KEEP.get(split, 0))]]
    if a.ds in ("drop", "musique", "hotpot"):
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with gzip.open(f, "wt") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in R)
    return R[:n]


def call(msgs, mt, temperature=0.0):
    try:
        return gem.chat(msgs, max_tokens=mt, temperature=temperature)[0]
    except gem.Abort:
        raise
    except Exception:  # noqa: BLE001
        return None


def run(inst, R, tag):
    f = f"{OUT}/answers_{tag}.csv"
    if os.path.exists(f):
        return pd.read_csv(f)

    def one(r):
        txt = call([dict(role="user", content=make_prompt(r, inst))], 3000)
        c, f1, pred = score(txt, r)
        return dict(qid=r["qid"], question=r["question"], pred=pred, gold=r["gold_text"], correct=c, f1=f1)
    with ThreadPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(ex.map(one, R))
    df.to_csv(f, index=False)
    print(f"[{tag}] EM {100 * df.correct.mean():.1f}  F1 {100 * df.f1.mean():.1f}  (n={len(df)})", flush=True)
    return df


def pairs_text(df, n_wrong=30, n_right=10):
    w = df[df.correct == 0].sample(min(n_wrong, (df.correct == 0).sum()), random_state=0)
    r = df[df.correct == 1].sample(min(n_right, (df.correct == 1).sum()), random_state=0)
    return "\n\n".join(f"Question: {x.question}\nModel answer: {x.pred}\nReference answer: {x.gold}\n({'correct' if x.correct else 'wrong'})"
                       for x in pd.concat([w, r]).sample(frac=1, random_state=0).itertuples())


# ---------------- universal deterministic features
def num(s):
    s = re.sub(r"(?i)\b(million|thousand|billion|percent)\b|\([a-z]+\)\s*$", "", str(s)).replace(",", "").replace("$", "").replace("%", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    try:
        v = float(s.strip("()"))
    except ValueError:
        return None
    return -v if neg else v


def det(ans):
    s = "" if ans is None or (isinstance(ans, float) and math.isnan(ans)) else str(ans)
    v = num(s)
    return {"number of words": len(s.split()), "number of listed items": len([x for x in re.split(r"\s*\|\s*|;\s+|,\s+(?![0-9]{3})|\s+and\s+", s) if x.strip()]),
            "is a number": float(v is not None), "is negative": float(v is not None and v < 0),
            "size of the number (log10)": math.log10(abs(v) + 1) if v is not None else 0.0}


# ---------------- B. feature proposal, C. probes
PROPOSE = """Below are questions with a model's answer and the reference answer (some correct, mostly wrong).

{pairs}

Propose {k} features that describe how the model's answers systematically differ from the reference answers. Each feature must be a question that can be asked about ANY (question, answer) pair without seeing the reference, so it can be measured for the model's answer and for the reference answer in the same way. Prefer features about the form and content of the answer (what it contains, its unit, its scope, its specificity, how much it includes), not about correctness.
Reply with only a JSON list: [{{"name": short name, "question": the question to ask about an answer, "type": "binary" or "count"}}]"""

PROBE = """Question: {q}
Answer: {ans}

Rate this answer on each feature below. For a binary feature give 1 or 0; for a count give a non-negative integer. Reply with only one line of JSON mapping each name to its value.
{feats}"""


def probe(F, q, ans):
    feats = "\n".join(f'- "{f["name"]}": {f["question"]} ({f["type"]})' for f in F)
    txt = call([dict(role="user", content=PROBE.format(q=q, ans=ans, feats=feats))], 400)
    try:
        d = json.loads(re.findall(r"\{.*\}", txt or "", re.S)[-1])
        return {f["name"]: float(d.get(f["name"], np.nan)) for f in F}
    except Exception:  # noqa: BLE001
        return {f["name"]: np.nan for f in F}


# ---------------- D. decoder
def ridge_anchor(X, y, k, folds=5):
    fold = np.arange(len(X)) % folds

    def fit(Xa, ya, lam):
        mu = ya.mean(); w0 = np.zeros(Xa.shape[1]); w0[k] = (Xa[:, k] @ (ya - mu)) / max(Xa[:, k] @ Xa[:, k], 1e-9)
        return mu, np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ (ya - mu) + lam * w0)
    lams = [1, 10, 100, 1000, 1e4]
    cv = [np.mean([np.mean((y[fold == f] - (lambda m, w: m + X[fold == f] @ w)(*fit(X[fold != f], y[fold != f], l))) ** 2)
                   for f in range(folds)]) for l in lams]
    return fit(X, y, lams[int(np.argmin(cv))])


def strength(r):
    return ("strongly" if r >= .8 else "clearly" if r >= .5 else "somewhat" if r >= .25 else "slightly")


def decode(Do, Dg, names):
    O, G = Do[names].to_numpy(float), Dg[names].to_numpy(float)
    ok = ~(np.isnan(O).any(1) | np.isnan(G).any(1)); O, G = O[ok], G[ok]
    mo, so = O.mean(0), np.where(O.std(0) > 0, O.std(0), 1)
    Z = (O - mo) / so; lines, info = [], []
    for k, nm in enumerate(names):
        d = G[:, k] - O[:, k]; t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12)
        bias = d.mean() * t * t / (1 + t * t)  # smooth shrinkage of the mean deviation
        mu, w = ridge_anchor(Z, G[:, k], k)
        sg = np.where(G[:, k].std() > 0, G[:, k].std(), 1)
        rel = bias / sg
        info.append(dict(feature=nm, model_mean=O[:, k].mean(), gold_mean=G[:, k].mean(), t=t, bias=bias))
        parts = []
        if abs(rel) >= .05:
            parts.append(f"the reference answers are {strength(abs(rel))} {'higher' if bias > 0 else 'lower'} than yours on "
                         f"'{nm}' (yours {O[:, k].mean():.2f} vs reference {G[:, k].mean():.2f} on average)")
        for j in np.argsort(-np.abs(w)):
            if j == k:
                continue
            r = abs(w[j]) / sg
            if r < .05:
                break
            parts.append(f"when your answer is higher on '{names[j]}', the reference tends to be {'higher' if w[j] > 0 else 'lower'} "
                         f"on '{nm}' ({strength(r)})")
        if parts:
            lines.append(f"- {nm}: " + "; ".join(parts) + ".")
    return "\n".join(lines), pd.DataFrame(info)


RULES = """You are improving the instruction a model follows when answering questions of this kind:
{example}

{evidence}

Write at most 8 short, general rules for how to answer (not tied to particular questions) that would make the model's answers match the reference answers better. Do not mention features or statistics. Reply with only the rules, one per line, each starting with "- "."""


if __name__ == "__main__":
    TR, TE = sample("train", a.ntr), sample("test", a.nte)
    Atr, Ate = run(BASE, TR, "train_base"), run(BASE, TE, "test_base")
    ex = Atr.question.iloc[0]
    pt = pairs_text(Atr)

    ff = f"{OUT}/features.json"
    if not os.path.exists(ff):
        txt = call([dict(role="user", content=PROPOSE.format(pairs=pt, k=a.k))], 2000)
        F = json.loads(re.findall(r"\[.*\]", txt, re.S)[-1])[:a.k]
        json.dump(F, open(ff, "w"), indent=1)
    F = json.load(open(ff))
    print("proposed features:\n" + "\n".join(f"  {f['name']} ({f['type']}): {f['question']}" for f in F), flush=True)

    pf = f"{OUT}/probes_train.csv"
    if not os.path.exists(pf):
        with ThreadPoolExecutor(a.workers) as ex_:
            o = list(ex_.map(lambda x: probe(F, x.question, x.pred), Atr.itertuples()))
            g = list(ex_.map(lambda x: probe(F, x.question, x.gold), Atr.itertuples()))
        P = pd.concat([pd.DataFrame(o).add_prefix("out:"), pd.DataFrame(g).add_prefix("gt:")], axis=1)
        P.insert(0, "qid", Atr.qid); P.to_csv(pf, index=False)
    P = pd.read_csv(pf)
    if a.probes_only:
        sys.exit(0)
    names = [f["name"] for f in F] + list(det("x"))
    Do = pd.concat([P[[f"out:{f['name']}" for f in F]].set_axis([f["name"] for f in F], axis=1),
                    pd.DataFrame([det(x) for x in Atr.pred])], axis=1)
    Dg = pd.concat([P[[f"gt:{f['name']}" for f in F]].set_axis([f["name"] for f in F], axis=1),
                    pd.DataFrame([det(re.sub(r"\s*\((thousand|million|billion|percent)\)$", "", str(x))) for x in Atr.gold])], axis=1)
    desc, info = decode(Do, Dg, names)
    print("\ndecoder (model answer vs reference, train):\n" + info.round(2).to_string(index=False), flush=True)
    print("\ndescription given to the rule writer:\n" + desc, flush=True)
    open(f"{OUT}/decoder_description.txt", "w").write(desc)

    rules = {}
    for arm, ev in (("shad", "How the model's answers differ from the reference answers, measured on many questions:\n" + desc),
                    ("errsum", "Examples of the model's answers and the reference answers:\n\n" + pt)):
        rf = f"{OUT}/rules_{arm}.txt"
        if not os.path.exists(rf):
            txt = call([dict(role="user", content=RULES.format(example=ex, evidence=ev))], 1200)
            open(rf, "w").write("\n".join(l for l in (txt or "").splitlines() if l.strip().startswith("-")))
        rules[arm] = open(rf).read()
        print(f"\nrules ({arm}):\n{rules[arm]}", flush=True)

    res = {"base": Ate}
    for arm, rl in rules.items():
        res[arm] = run(BASE + "\n\nGuidelines:\n" + rl, TE, f"test_{arm}")
    rng = np.random.default_rng(0); b = Ate.set_index("qid").correct
    print("\n== test")
    for arm, df in res.items():
        d = df.set_index("qid").correct.loc[b.index] - b
        se = np.std([d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"{arm:7s} EM {100 * df.correct.mean():.1f}  F1 {100 * df.f1.mean():.1f}  vs base {100 * d.mean():+.1f} (SE {100 * se:.1f})")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

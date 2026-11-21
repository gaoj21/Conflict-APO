"""Learned features -> small network -> rules, on DROP, one step on top of the round-0 error-summary prompt.

  .venv/bin/python qa/drop_netfeat.py [--k 24] [--workers 32]
1. data: the answers of the start prompt (round-0 error-summary rules) on the 3000-question pool of qa/drop_rounds.py
2. features learned from the data: embeddings of "Question + model answer" (OpenRouter text-embedding-3-small),
   non-negative sparse dictionary learning (k atoms); each atom is named by the LLM from its top-activating vs inactive
   examples, and the name is scored by how well the LLM predicts held-out activations from it (interpretability)
3. small network: atoms -> non-negative sparse hidden concepts (softplus) -> P(wrong); hidden size and L1 by 5-fold CV
4. translation: each hidden concept = its named atoms; its output weight says whether answers high on it are more or
   less often wrong; plus the strongest single atoms; the LLM rephrases each statement as one rule for the answer line
5. each rule is tested alone on val (500) on top of the start prompt, next to the round-0 error-summary candidates; the
   best of each is evaluated on test (1000)
"""
import hashlib, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]; opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
K = int(opt("--k", "24")); W = opt("--workers", "32")
sys.argv = ["drop_rounds.py", "--workers", W, "--nshad", "3000"]; sys.path.insert(0, HERE)
import drop_rounds as D  # noqa: E402  (DROP adapter via shadqa, run(), prompt(), pools, start rules)
sys.path.insert(0, HERE)
from net import fit as net_fit, logloss  # noqa: E402  (non-negative sparse concept network)

S, gem = D.S, D.S.gem
OUT = f"{S.OUT}/netfeat"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = "qa/drop/netfeat"


def embed(texts, f):
    if os.path.exists(f):
        return np.load(f)
    out = []
    for i in range(0, len(texts), 256):
        r = requests.post("https://openrouter.ai/api/v1/embeddings", headers={"Authorization": "Bearer " + open(os.path.expanduser("~/.openrouter_key")).read().strip()},
                          json={"model": "openai/text-embedding-3-small", "input": texts[i:i + 256]}, timeout=120)
        out += [d["embedding"] for d in r.json()["data"]]
    E = np.array(out, dtype=np.float32); np.save(f, E)
    return E


NAME = """Below are question-answer pairs. Group A shares a property that group B does not have. The property is about how the ANSWER is written given the question (its form, length, content type, wording, scope).

Group A:
{a}

Group B:
{b}

Describe the property of group A in one short phrase that could be checked for any new answer. Reply with only the phrase."""

CHECK = """Property: {name}

For each numbered question-answer pair, say whether the answer has this property. Reply with only one line of JSON mapping each number to 1 or 0.

{items}"""


def qa_text(x):
    return f"Question: {x.question}\nAnswer: {S.plain(x.pred) if hasattr(S, 'plain') else x.pred}"


if __name__ == "__main__":
    TRall = S.rows("train"); idx = np.random.default_rng(0).permutation(len(TRall))
    FS = [TRall[i] for i in idx[D.NFIT + D.NVAL:D.NFIT + D.NVAL + 3000]]; VR = [TRall[i] for i in idx[D.NFIT:D.NFIT + D.NVAL]]
    TE = S.sample("test", D.NTE)
    ST = json.load(open(f"{D.OUT}/start.json")); start = D.prompt(ST["block"])
    A = D.run(start, FS, "errsum_shad3000_r0_shadfit")  # cached: the start prompt on the 3000 pool
    A["pred"] = A.pred.fillna("")
    texts = [f"Question: {q}\nAnswer: {p}" for q, p in zip(A.question, A.pred)]
    E = embed(texts, f"{OUT}/emb.npy"); E = E - E.mean(0)
    print(f"{len(A)} answers, wrong {100 * (1 - A.correct.mean()):.1f}%, embeddings {E.shape}", flush=True)

    from sklearn.decomposition import MiniBatchDictionaryLearning
    cf = f"{OUT}/codes.npy"
    if not os.path.exists(cf):
        dl = MiniBatchDictionaryLearning(n_components=K, alpha=.1, positive_code=True, fit_algorithm="cd", transform_algorithm="lasso_cd", transform_alpha=.1, random_state=0, batch_size=256, max_iter=50)
        np.save(cf, dl.fit_transform(E))
    Cd = np.load(cf); print(f"codes: mean active atoms per answer {(Cd > 0).sum(1).mean():.1f}", flush=True)

    nf = f"{OUT}/names.json"
    if not os.path.exists(nf):
        rng = np.random.default_rng(0); names = []
        for j in range(K):
            order = np.argsort(-Cd[:, j]); top = order[:40]; off = np.where(Cd[:, j] == 0)[0]
            off = rng.permutation(off)[:40] if len(off) else order[-40:]
            fmt = lambda ii: "\n".join(f"- {texts[i]}".replace("\n", " | ") for i in ii)
            nm = (S.call([dict(role="user", content=NAME.format(a=fmt(top[:15]), b=fmt(off[:15])))], 100) or "").strip().strip('"')
            held = list(top[15:25]) + list(off[15:25]); lab = [1] * 10 + [0] * 10; perm = rng.permutation(20)
            items = "\n".join(f"{n + 1}. {texts[held[p]]}".replace("\n", " | ") for n, p in enumerate(perm))
            txt = S.call([dict(role="user", content=CHECK.format(name=nm, items=items))], 300)
            try:
                d = json.loads(re.findall(r"\{.*\}", txt or "", re.S)[-1]); pred = [int(d.get(str(n + 1), 0)) for n in range(20)]
                score = np.mean([pred[n] == lab[p] for n, p in enumerate(perm)])
            except Exception:  # noqa: BLE001
                score = .5
            names.append(dict(atom=j, name=nm, interp=float(score), active=float((Cd[:, j] > 0).mean()),
                              wrong_if_active=float(1 - A.correct[Cd[:, j] > 0].mean()) if (Cd[:, j] > 0).any() else None))
        json.dump(names, open(nf, "w"), indent=1)
    names = json.load(open(nf))
    print("learned features (interpretability = held-out accuracy of the name):")
    for n in names:
        print(f"  atom {n['atom']:2d} interp {n['interp']:.2f} active {100 * n['active']:4.1f}% wrong-if-active {100 * (n['wrong_if_active'] or 0):4.1f}%  {n['name']}")

    X = (Cd - Cd.mean(0)) / np.maximum(Cd.std(0), 1e-6)
    X = X * np.array([max(n["interp"] - .5, 0) * 2 for n in names])  # features whose names do not hold are smoothly down-weighted
    y = (1 - A.correct.to_numpy(float))[:, None]
    folds = np.random.default_rng(0).integers(0, 5, len(X)); cv = {}
    for H in (1, 2, 3, 4):
        for l1 in (1e-3, 3e-3, 1e-2):
            cv[(H, l1)] = np.mean([logloss(net_fit(X[folds != k], y[folds != k], H, l1)["prob"](X[folds == k]), y[folds == k]) for k in range(5)])
    H, l1 = min(cv, key=cv.get); net = net_fit(X, y, H, l1)
    p0 = y.mean(); null = -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    print(f"\nnetwork: H={H}, l1={l1}, CV log loss {cv[(H, l1)]:.4f} vs no-feature {null:.4f}", flush=True)

    st = []
    for h in range(H):
        w1 = net["W1"][:, h]; top = [j for j in np.argsort(-w1) if w1[j] > .1 * w1.max()][:3]
        v = float(net["V"][h, 0])
        desc = " and ".join(f"'{names[j]['name']}'" for j in top)
        st.append(dict(kind="concept", weight=abs(v) * float(w1[top].sum()), statement=
                       f"Final answers that show {desc} are {'more' if v > 0 else 'less'} often judged wrong, so final answers should "
                       f"{'avoid' if v > 0 else 'prefer'} this.", atoms=[int(j) for j in top]))
        print(f"  concept {h + 1} (output weight {v:+.2f}): " + ", ".join(f"atom {j} {w1[j]:.2f} [{names[j]['name']}]" for j in top))
    sf = f"{OUT}/rules.json"
    if not os.path.exists(sf):
        st = sorted(st, key=lambda r: -r["weight"])
        for r in st:
            r["rule"] = "- " + (S.call([dict(role="user", content=D.M.REPHRASE.format(s=r["statement"]))], 200) or r["statement"]).strip().lstrip("- ")
        json.dump(st, open(sf, "w"), indent=1)
    st = json.load(open(sf))
    print("\nrules from the network:\n" + "\n".join(f"  {r['rule']}" for r in st), flush=True)

    cur = D.run(start, VR, "val_start").correct.mean() if False else ST["val"]
    res = []
    for i, r in enumerate(st):
        v = D.run(D.prompt(ST["block"] + [r["rule"]]), VR, f"netfeat_val_{i}").correct.mean(); res.append((v, r["rule"]))
        print(f"[val] net rule {i}: EM {100 * v:.1f} (start {100 * cur:.1f})  {r['rule']}", flush=True)
    E0 = json.load(open(f"{D.OUT}/errsum.json"))["log"][0]["errsum_tested"]
    print("[val] round-0 error-summary candidates (same start, same val): " + " || ".join(E0))
    bv, br = max(res)
    rng = np.random.default_rng(0); Tst = D.run(start, TE, "test_start")
    print("\n== test (n=%d)" % len(TE))
    for name, block in (("start", ST["block"]), ("start + best net rule", ST["block"] + [br]),
                        ("start + best errsum r0 rule", ST["block"] + [json.load(open(f"{D.OUT}/errsum.json"))["log"][0]["errsum_kept"]])):
        df = D.run(D.prompt(block), TE, "test_" + hashlib.sha1(name.encode()).hexdigest()[:6])
        d = (df.correct - Tst.correct).to_numpy(); se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"{name:30s} EM {100 * df.correct.mean():.1f}  vs start {100 * d.mean():+.1f} (SE {100 * se:.1f})")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

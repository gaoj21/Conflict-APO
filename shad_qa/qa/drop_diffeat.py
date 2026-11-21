"""Deviation features learned from data -> small network -> rules, on DROP, on top of the round-0 error-summary prompt.

  .venv/bin/python qa/drop_diffeat.py [--k 16] [--workers 32]
1. data: the start prompt's answers on the 3000-question pool; the wrong ones are the training signal
2. features learned from data: embedding(model answer) - embedding(reference answer) for every wrong answer; what the
   two share (the kind of answer the question asks for) cancels, what remains is how the model's answer deviates;
   non-negative sparse dictionary learning gives k deviation directions; each is named by the LLM from its
   top-activating (model -> reference) pairs vs inactive pairs, and the name is scored on held-out pairs
3. small network: question type (wh-word indicators) -> how strongly each deviation occurs (ridge, one output per
   deviation), i.e. in which kinds of questions each deviation happens
4. translation: one rule per deviation: avoid it, and name the question types where it is most frequent;
   ordered by frequency x interpretability; the LLM only rephrases
5. every rule tested alone on val (500) on top of the start prompt; the best one on test (1000), next to the best
   round-0 error-summary rule
"""
import hashlib, json, os, re, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]; opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
K = int(opt("--k", "16")); W = opt("--workers", "32")
V2 = "--v2" in argv  # form/content split by answer overlap; each deviation as a global and an "only when" rule
sys.argv = ["drop_netfeat.py", "--workers", W]; sys.path.insert(0, HERE)
import drop_netfeat as NF  # noqa: E402  (embedding helper, DROP rounds module, pools)

D, S, gem = NF.D, NF.S, NF.gem
OUT = f"{S.OUT}/diffeat"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = "qa/drop/diffeat"

NAME = """Each line shows a model's answer and the reference answer to the same question. In group A the model's answer differs from the reference in a shared, systematic way; group B does not share it.

Group A:
{a}

Group B:
{b}

In one short phrase, describe how the model's answer differs from the reference in group A (what it adds, drops, or writes differently). Reply with only the phrase."""

CHECK = """Deviation: the model's answer {name} (compared with the reference).

For each numbered pair, say whether the model's answer shows this deviation. Reply with only one line of JSON mapping each number to 1 or 0.

{items}"""

RULE = ("Rewrite this as one short instruction for how to write the final answer line. Keep its meaning; do not add "
        "anything. Reply with only the instruction.\n\nStatement: {s}")

if __name__ == "__main__":
    TRall = S.rows("train"); idx = np.random.default_rng(0).permutation(len(TRall))
    FS = [TRall[i] for i in idx[D.NFIT + D.NVAL:D.NFIT + D.NVAL + 3000]]; VR = [TRall[i] for i in idx[D.NFIT:D.NFIT + D.NVAL]]
    TE = S.sample("test", D.NTE)
    ST = json.load(open(f"{D.OUT}/start.json")); start = D.prompt(ST["block"])
    A = D.run(start, FS, "errsum_shad3000_r0_shadfit"); A["pred"] = A.pred.fillna("").astype(str); A["gold"] = A.gold.astype(str)
    Wr = A[A.correct == 0].reset_index(drop=True)
    nz = lambda xs: [x if x.strip() else "(no answer)" for x in xs]  # the embedding API rejects empty strings
    Ep = NF.embed(nz(Wr.pred), f"{OUT}/emb_pred.npy"); Eg = NF.embed(nz(Wr.gold), f"{OUT}/emb_gold.npy")
    Dv = Ep - Eg
    print(f"{len(Wr)} wrong answers (of {len(A)}); deviation vectors {Dv.shape}", flush=True)

    from sklearn.decomposition import MiniBatchDictionaryLearning
    cf = f"{OUT}/codes.npy"
    if not os.path.exists(cf):
        dl = MiniBatchDictionaryLearning(n_components=K, alpha=.1, positive_code=True, fit_algorithm="cd", transform_algorithm="lasso_cd",
                                         transform_alpha=.1, random_state=0, batch_size=128, max_iter=100)
        np.save(cf, dl.fit_transform(Dv))
    Cd = np.load(cf); print(f"codes: mean active deviations per wrong answer {(Cd > 0).sum(1).mean():.1f}", flush=True)
    pair = lambda i: f"model: {Wr.pred[i][:80]!r} -> reference: {Wr.gold[i][:80]!r}"

    nf = f"{OUT}/names.json"
    if not os.path.exists(nf):
        rng = np.random.default_rng(0); names = []
        for j in range(K):
            order = np.argsort(-Cd[:, j]); act = int((Cd[:, j] > 0).sum()); top = order[:min(25, act)]
            off = rng.permutation(np.where(Cd[:, j] == 0)[0])[:25]
            nm = (S.call([dict(role="user", content=NAME.format(a="\n".join(pair(i) for i in top[:15]),
                                                                b="\n".join(pair(i) for i in off[:15])))], 100) or "").strip().strip('"').rstrip(".")
            held = list(top[15:25]) + list(off[15:25]); lab = [1] * len(top[15:25]) + [0] * len(off[15:25]); perm = rng.permutation(len(held))
            txt = S.call([dict(role="user", content=CHECK.format(name=nm, items="\n".join(f"{n + 1}. {pair(held[p])}" for n, p in enumerate(perm))))], 300)
            try:
                d = json.loads(re.findall(r"\{.*\}", txt or "", re.S)[-1])
                score = float(np.mean([int(d.get(str(n + 1), 0)) == lab[p] for n, p in enumerate(perm)])) if len(held) else .5
            except Exception:  # noqa: BLE001
                score = .5
            names.append(dict(dev=j, name=nm, interp=score, freq=act / len(Wr)))
        json.dump(names, open(nf, "w"), indent=1)
    names = json.load(open(nf))
    print("learned deviations (freq = share of wrong answers showing it; interp = held-out accuracy of the name):")
    for n in sorted(names, key=lambda n: -n["freq"] * n["interp"]):
        print(f"  dev {n['dev']:2d} freq {100 * n['freq']:4.1f}% interp {n['interp']:.2f}  the model's answer {n['name']}")

    Q = np.array([[float(w in q.lower()) for w in D.WH] for q in Wr.question])
    Qs = (Q - Q.mean(0)) / np.maximum(Q.std(0), 1e-6); Y = (Cd > 0).astype(float)
    Bw = np.linalg.solve(Qs.T @ Qs + 50 * np.eye(Qs.shape[1]), Qs.T @ (Y - Y.mean(0)))  # ridge, one output per deviation
    if V2:
        # form vs content: on a deviation's top examples, does the model's answer overlap the reference (token F1 > 0)?
        for n in names:
            top = np.argsort(-Cd[:, n["dev"]])[:25]
            n["form"] = float(np.mean([S.sf1(S.dnorm(Wr.pred[i]), S.dnorm(Wr.gold[i])) > 0 for i in top]))
        print("form share (top examples overlapping the reference): " + ", ".join(f"dev {n['dev']} {n['form']:.2f}" for n in names))
        rf = f"{OUT}/rules_v2.json"
        if not os.path.exists(rf):
            rules = []
            for n in sorted(names, key=lambda n: -n["freq"] * max(n["interp"] - .5, 0) * n["form"])[:4]:
                j = n["dev"]; cond = [D.WH[i] for i in np.argsort(-Bw[:, j]) if Bw[i, j] > 0][:2]
                base = f"Compared with the reference answers, the model's final answers often {n['name']}."
                variants = [("global", base + " The final answer should not do this.")]
                if cond:
                    q = " or ".join(f"'{c}'" for c in cond)
                    variants.append(("only when", base + f" This happens in questions that ask {q}. Only for questions that ask {q}, "
                                     "the final answer should not do this; for all other questions keep writing the answer as before."))
                for kind, st in variants:
                    rules.append(dict(dev=j, kind=kind, statement=st,
                                      rule="- " + (S.call([dict(role="user", content=RULE.format(s=st) + " Keep any 'only for' condition exactly.")], 200) or st).strip().lstrip("- ")))
            json.dump(rules, open(rf, "w"), indent=1)
        rules = json.load(open(rf))
        print("\nrules (v2):\n" + "\n".join(f"  [dev {r['dev']}, {r['kind']}] {r['rule']}" for r in rules), flush=True)
    rf = f"{OUT}/rules.json"
    if not V2 and not os.path.exists(rf):
        rules = []
        for n in sorted(names, key=lambda n: -n["freq"] * max(n["interp"] - .5, 0)):
            if n["interp"] <= .5:
                continue
            j = n["dev"]; cond = [D.WH[i] for i in np.argsort(-Bw[:, j]) if Bw[i, j] > 0][:2]
            s = (f"Compared with the reference answers, the model's final answers often {n['name']}"
                 + (f", most of all for questions that ask '{cond[0]}'" + (f" or '{cond[1]}'" if len(cond) > 1 else "") if cond else "")
                 + ". The final answer should not do this.")
            rules.append(dict(dev=j, statement=s, rule="- " + (S.call([dict(role="user", content=RULE.format(s=s))], 200) or s).strip().lstrip("- ")))
        json.dump(rules, open(rf, "w"), indent=1)
    if not V2:
        rules = json.load(open(rf))
        print("\nrules:\n" + "\n".join(f"  [dev {r['dev']}] {r['rule']}" for r in rules), flush=True)

    res = []
    for i, r in enumerate(rules if V2 else rules[:6]):
        v = D.run(D.prompt(ST["block"] + [r["rule"]]), VR, f"diffeat{'2' if V2 else ''}_val_{i}").correct.mean(); res.append((v, r["rule"]))
        print(f"[val] rule {i}: EM {100 * v:.1f} (start {100 * ST['val']:.1f})  {r['rule']}", flush=True)
    bv, br = max(res)
    E0 = json.load(open(f"{D.OUT}/errsum.json"))["log"][0]
    print("[val] round-0 error-summary candidates: " + " || ".join(E0["errsum_tested"]))
    rng = np.random.default_rng(0); Tst = D.run(start, TE, "test_start")
    print("\n== test (n=%d)" % len(TE))
    for name, block in (("start + best deviation rule", ST["block"] + [br]), ("start + best errsum r0 rule", ST["block"] + [E0["errsum_kept"]])):
        df = D.run(D.prompt(block), TE, "test_" + hashlib.sha1(name.encode()).hexdigest()[:6])
        d = (df.correct - Tst.correct).to_numpy(); se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"{name:30s} EM {100 * df.correct.mean():.1f}  vs start ({100 * Tst.correct.mean():.1f}) {100 * d.mean():+.1f} (SE {100 * se:.1f})")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

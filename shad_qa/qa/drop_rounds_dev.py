"""Round-by-round error summarization + ShAD with learned deviation and question features, on DROP.

  .venv/bin/python qa/drop_rounds_dev.py [--rounds 5] [--kd 16] [--kq 16] [--workers 32]
Same start, fit / val / test pools and error-summary step as qa/drop_rounds.py (its "errsum" arm is the control).
ShAD step of every round, on the 3000-question pool answered with the current prompt:
  deviation features : embedding(model answer) - embedding(reference) of the wrong answers -> non-negative sparse
                       dictionary (kd atoms), named from (model -> reference) pairs, name checked on held-out pairs;
                       form share = how often the top pairs overlap the reference (how it is written vs what it says)
  question features  : embeddings of the questions -> non-negative sparse dictionary (kq atoms), named once from
                       top-activating vs inactive questions, checked on held-out questions
  network            : question atoms -> each deviation (ridge, one output per deviation)
  translation        : for the 3 deviations ranked by frequency x interpretability x form share: a global rule and an
                       "only for questions that <the 2 question atoms that most raise it>" rule
  each rule is tested alone on val; the best is kept if val EM does not drop
"""
import hashlib, json, os, re, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]; opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
ROUNDS, KD, KQ, W = int(opt("--rounds", "5")), int(opt("--kd", "16")), int(opt("--kq", "16")), opt("--workers", "32")
sys.argv = ["drop_netfeat.py", "--workers", W]; sys.path.insert(0, HERE)
import drop_netfeat as NF  # noqa: E402
import drop_diffeat as DF  # noqa: E402  (naming prompts)

D, S, gem = NF.D, NF.S, NF.gem
OUT = f"{S.OUT}/rounds_dev"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = "qa/drop/rounds_dev"
from sklearn.decomposition import MiniBatchDictionaryLearning  # noqa: E402

QNAME = """Group A questions share a property that group B questions do not have (about what kind of answer they ask for).

Group A:
{a}

Group B:
{b}

Describe the property of group A in a short phrase that completes "questions that ...". Reply with only the phrase."""
QCHECK = """Property: questions that {name}

For each numbered question, say whether it has this property. Reply with only one line of JSON mapping each number to 1 or 0.

{items}"""


def dictionary(X, k, f):
    if os.path.exists(f):
        return np.load(f)
    C = MiniBatchDictionaryLearning(n_components=k, alpha=.1, positive_code=True, fit_algorithm="cd", transform_algorithm="lasso_cd",
                                    transform_alpha=.1, random_state=0, batch_size=128, max_iter=100).fit_transform(X)
    np.save(f, C); return C


def name_atoms(C, show, name_prompt, check_prompt, f, fmt_name=lambda s: s):
    if os.path.exists(f):
        return json.load(open(f))
    rng = np.random.default_rng(0); out = []
    for j in range(C.shape[1]):
        order = np.argsort(-C[:, j]); act = int((C[:, j] > 0).sum()); top = order[:min(25, act)]
        off = rng.permutation(np.where(C[:, j] == 0)[0])[:25]
        nm = (S.call([dict(role="user", content=name_prompt.format(a="\n".join(show(i) for i in top[:15]), b="\n".join(show(i) for i in off[:15])))], 100)
              or "").strip().strip('"').rstrip(".")
        held = list(top[15:25]) + list(off[15:25]); lab = [1] * len(top[15:25]) + [0] * len(off[15:25]); perm = rng.permutation(len(held))
        txt = S.call([dict(role="user", content=check_prompt.format(name=fmt_name(nm), items="\n".join(f"{n + 1}. {show(held[p])}" for n, p in enumerate(perm))))], 300)
        try:
            d = json.loads(re.findall(r"\{.*\}", txt or "", re.S)[-1])
            sc = float(np.mean([int(d.get(str(n + 1), 0)) == lab[p] for n, p in enumerate(perm)])) if len(held) else .5
        except Exception:  # noqa: BLE001
            sc = .5
        out.append(dict(atom=j, name=nm, interp=sc, freq=act / len(C)))
    json.dump(out, open(f, "w"), indent=1); return out


def shad_candidates(Fd, Cq, qnames, t, pfx="r"):
    Fd = Fd.assign(pred=Fd.pred.fillna("").astype(str), gold=Fd.gold.astype(str)); wi = np.where(Fd.correct.to_numpy() == 0)[0]
    Wr = Fd.iloc[wi].reset_index(drop=True); nz = lambda xs: [x if x.strip() else "(no answer)" for x in xs]
    Dv = NF.embed(nz(Wr.pred), f"{OUT}/{pfx}{t}_emb_pred.npy") - NF.embed(nz(Wr.gold), f"{OUT}/{pfx}{t}_emb_gold.npy")
    Cd = dictionary(Dv, KD, f"{OUT}/{pfx}{t}_dev_codes.npy")
    pair = lambda i: f"model: {Wr.pred[i][:80]!r} -> reference: {Wr.gold[i][:80]!r}"
    dn = name_atoms(Cd, pair, DF.NAME, DF.CHECK, f"{OUT}/{pfx}{t}_dev_names.json")
    for n in dn:
        top = np.argsort(-Cd[:, n["atom"]])[:25]
        n["form"] = float(np.mean([S.sf1(S.dnorm(Wr.pred[i]), S.dnorm(Wr.gold[i])) > 0 for i in top]))
    Q = Cq[wi]; Qs = (Q - Q.mean(0)) / np.maximum(Q.std(0), 1e-6) * np.array([max(q["interp"] - .5, 0) * 2 for q in qnames])
    Y = (Cd > 0).astype(float); B = np.linalg.solve(Qs.T @ Qs + 50 * np.eye(Qs.shape[1]), Qs.T @ (Y - Y.mean(0)))
    cands = []
    for n in sorted(dn, key=lambda n: -n["freq"] * max(n["interp"] - .5, 0) * n["form"])[:3]:
        j = n["atom"]; cond = [qnames[i]["name"] for i in np.argsort(-B[:, j]) if B[i, j] > 0][:2]
        base = f"Compared with the reference answers, the model's final answers often {n['name']}."
        vs = [base + " The final answer should not do this."]
        if cond:
            q = " or ".join(f"questions that {c}" for c in cond)
            vs.append(base + f" This happens mostly in {q}. Only for {q}, the final answer should not do this; for all other questions keep writing the answer as before.")
        for st in vs:
            cands.append("- " + (S.call([dict(role="user", content=DF.RULE.format(s=st) + " Keep any 'only for' condition exactly.")], 200) or st).strip().lstrip("- "))
    return cands, dn


if __name__ == "__main__":
    TRall = S.rows("train"); idx = np.random.default_rng(0).permutation(len(TRall))
    FR = [TRall[i] for i in idx[:D.NFIT]]; VR = [TRall[i] for i in idx[D.NFIT:D.NFIT + D.NVAL]]
    FS = [TRall[i] for i in idx[D.NFIT + D.NVAL:D.NFIT + D.NVAL + 3000]]; TE = S.sample("test", D.NTE)
    ST = json.load(open(f"{D.OUT}/start.json"))
    qs = [r["question"] for r in FS]
    Eq = NF.embed(qs, f"{OUT}/emb_questions.npy"); Cq = dictionary(Eq - Eq.mean(0), KQ, f"{OUT}/q_codes.npy")
    qnames = name_atoms(Cq, lambda i: f"- {qs[i]}", QNAME, QCHECK, f"{OUT}/q_names.json")
    print("learned question features:\n" + "\n".join(f"  q{q['atom']:2d} interp {q['interp']:.2f} active {100 * q['freq']:4.1f}%  questions that {q['name']}" for q in qnames), flush=True)

    stf = f"{OUT}/state.json"
    A_ = json.load(open(stf)) if os.path.exists(stf) else dict(block=list(ST["block"]), val=ST["val"], log=[])
    for t in range(len(A_["log"]), ROUNDS):
        tag = f"dev_r{t}"
        Fd = D.run(D.prompt(A_["block"]), FR, f"{tag}_fit")
        gem.SALT.v = f"errsum+shad{t}"  # the same error-summary proposals as the errsum+shad arm of qa/drop_rounds.py would draw
        more = D.rules_of(S.call([dict(role="user", content=D.MORE.format(rules="\n".join(A_["block"]), pairs=S.pairs_text(Fd)))], 800, temperature=0.7), 3)
        gem.SALT.v = ""
        kept, A_["val"], tested = D.best_single(A_["block"], more, VR, A_["val"], f"{tag}_es")
        if kept:
            A_["block"].append(kept)
        Fs = D.run(D.prompt(A_["block"]), FS, f"{tag}_shadfit")
        cands, dn = shad_candidates(Fs, Cq, qnames, t)
        cands = [c for c in dict.fromkeys(cands) if c not in A_["block"]]
        k2, A_["val"], t2 = D.best_single(A_["block"], cands, VR, A_["val"], f"{tag}_sh")
        if k2:
            A_["block"].append(k2)
        A_["log"].append(dict(round=t, errsum_kept=kept, errsum_tested=tested, shad_kept=k2, shad_tested=t2, val=100 * A_["val"]))
        json.dump(A_, open(stf, "w"), indent=1)
        print(f"[round {t}] errsum {'kept: ' + kept if kept else 'none kept'} | ShAD {'kept: ' + k2 if k2 else 'none kept'} [{' || '.join(t2)}] | val EM {100 * A_['val']:.1f}  (spent ${gem.spent()[0]:.2f})", flush=True)

    rng = np.random.default_rng(0)
    ctrl = json.load(open(f"{D.OUT}/errsum.json"))["block"]
    res = {k: D.run(D.prompt(b), TE, "test_" + hashlib.sha1(k.encode()).hexdigest()[:6]) for k, b in
           (("start", ST["block"]), ("errsum", ctrl), ("errsum+devshad", A_["block"]))}
    print("\n== test (n=%d)" % len(TE))
    for k, df in res.items():
        print(f"{k:16s} EM {100 * df.correct.mean():.1f}  F1 {100 * df.f1.mean():.1f}")
    for x, y in (("errsum", "start"), ("errsum+devshad", "start"), ("errsum+devshad", "errsum")):
        d = (res[x].correct - res[y].correct).to_numpy(); se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"{x} vs {y}: {100 * d.mean():+.1f} (SE {100 * se:.1f}){' *' if abs(d.mean()) > 2 * se else ''}")
    print("final block:\n" + "\n".join("  " + r for r in A_["block"]))
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

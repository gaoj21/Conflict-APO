"""Round-by-round error summarization with and without ShAD on DROP (official EM normalization).

  .venv/bin/python qa/drop_rounds.py [--rounds 5] [--nfit 600] [--nval 500] [--nte 1000] [--workers 32]
Prompt = "Think step by step." + scope ("rules apply only to the final Answer line") + a rule block.
Round 0 (shared start): the LLM writes up to 8 rules from contrasting pairs of the base answers; top 1/2/4/8 chosen on val.
Every later round, for both arms on their own state:
  1. answer the fit questions with the current prompt
  2. error-summary step: the LLM sees the current rules and new contrasting pairs and proposes up to 3 more rules;
     each is tested alone on val, the best is kept if val EM does not drop
  3. ShAD step (arm "errsum+shad" only): probe the answer features of the current answers, fit the decoder
     P(wrong | answer features, question type, answer x question type), translate the 3 strongest untried statements
     (one-sided conditions), test each alone on val, keep the best if val EM does not drop
Test: base, the shared start, both final prompts; paired differences with bootstrap SE.
"""
import hashlib, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]; opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
W = opt("--workers", "32")
DS = os.environ.get("QA_DS", "drop")
sys.argv = ["shadqa.py", DS, "--workers", W]; sys.path.insert(0, HERE)
import shadqa as S  # noqa: E402  (DROP adapter, probe, deterministic features, call, rule-writing prompt)
sys.argv = ["mixqa.py", "--workers", W]
import mixqa as M  # noqa: E402  (logistic decoder with products, one-sided statements, rephrase prompt)

import numpy as np, pandas as pd  # noqa: E402

gem = S.gem
ROUNDS, NFIT, NVAL, NTE = int(opt("--rounds", "5")), int(opt("--nfit", "600")), int(opt("--nval", "500")), int(opt("--nte", "1000"))
NSHAD = int(opt("--nshad", "0"))  # > 0: the decoder is fitted on its own larger pool (error summary still reads the fit pool)
SHARM = "errsum+shad" + (str(NSHAD) if NSHAD else "")
OUT = f"{S.OUT}/rounds"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = f"qa/{DS}/rounds"
SCOPE = M.SCOPE
WH = ["how many", "how many more", "how many yards", "percent", "who", "which", "what", "when", "how long", "difference"]
QF = [dict(name=f"asks '{w}'", question=f"Does the question ask '{w}'?") for w in WH]

MORE = """A model answers reading-comprehension questions about passages (sports reports, history articles). Its final answer line follows these rules:
{rules}

Below are its current answers with the reference answers (mostly wrong ones).

{pairs}

Propose up to 3 NEW short rules (not repeating the current ones) for how to write the final answer line so that it matches the reference answers better. Order them from most to least important. Reply with only the rules, one per line, each starting with "- "."""


def prompt(block):
    return S.BASE + (SCOPE + "\n".join(block) if block else "")


def run(inst, R, tag):
    f = f"{OUT}/answers_{tag}_{hashlib.sha1(inst.encode()).hexdigest()[:8]}.csv"
    if os.path.exists(f):
        return pd.read_csv(f)

    def one(r):
        txt = S.call([dict(role="user", content=S.make_prompt(r, inst))], 3000)
        c, f1, pred = S.score(txt, r)
        return dict(qid=r["qid"], question=r["question"], pred=pred, gold=r["gold_text"], correct=c, f1=f1)
    with ThreadPoolExecutor(int(W)) as ex:
        df = pd.DataFrame(ex.map(one, R))
    df.to_csv(f, index=False)
    return df


def rules_of(txt, k):
    return [l.strip() for l in (txt or "").splitlines() if l.strip().startswith("-")][:k]


def best_single(block, cands, VR, cur, tag):
    tests = [(run(prompt(block + [c]), VR, f"{tag}_{i}").correct.mean(), i) for i, c in enumerate(cands)]
    if not tests:
        return None, cur, []
    v, i = max(tests)
    return (cands[i] if v >= cur else None), max(v, cur), [f"{100 * t:.1f} {cands[j]}" for t, j in tests]


def shad_cands(F, Fdf, tried, k=3):
    with ThreadPoolExecutor(int(W)) as ex:
        pa = list(ex.map(lambda x: S.probe(F, x.question, x.pred), Fdf.itertuples()))
    A = pd.concat([pd.DataFrame(pa)[[f["name"] for f in F]],
                   pd.DataFrame([{k_: v for k_, v in S.det(x).items() if k_ in ("number of words", "number of listed items")}
                                 for x in Fdf.pred])], axis=1)
    C = pd.DataFrame([{f["name"]: float(w in q.lower()) for f, w in zip(QF, WH)} for q in Fdf.question])
    Wt, info = M.decoder(A, C, 1 - Fdf.correct.to_numpy())
    L = [x for x in M.statements(Wt, F, QF, A, C, None) if (x["answer_feature"], x["context"]) not in tried][:k]
    for x in L:
        x["rule"] = "- " + (S.call([dict(role="user", content=M.REPHRASE.format(s=x["statement"]))], 200) or x["statement"]).strip().lstrip("- ")
    return L, info


if __name__ == "__main__":
    TRall = S.rows("train"); idx = np.random.default_rng(0).permutation(len(TRall))
    FR = [TRall[i] for i in idx[:NFIT]]; VR = [TRall[i] for i in idx[NFIT:NFIT + NVAL]]
    FS = [TRall[i] for i in idx[NFIT + NVAL:NFIT + NVAL + NSHAD]] if NSHAD else FR
    TE = S.sample("test", NTE)
    F = json.load(open(f"{S.OUT}/features.json"))
    base_f, base_v = run(prompt([]), FR, "fit_base"), run(prompt([]), VR, "val_base")
    print(f"base: fit EM {100 * base_f.correct.mean():.1f}, val EM {100 * base_v.correct.mean():.1f}", flush=True)

    sf = f"{OUT}/start.json"
    if not os.path.exists(sf):
        R0 = rules_of(S.call([dict(role="user", content=S.RULES.format(example=FR[0]["question"],
             evidence="Examples of the model's answers and the reference answers:\n\n" + S.pairs_text(base_f))
             + "\nOrder the rules from most to least important. The rules are only about how to write the final answer line.")], 1200), 8)
        vs = {n: run(prompt(R0[:n]), VR, f"val_start{n}").correct.mean() for n in (1, 2, 4, 8) if n <= len(R0)}
        n = max(vs, key=vs.get)
        json.dump(dict(block=R0[:n], val=vs[n], all=R0, vals={str(k): v for k, v in vs.items()}), open(sf, "w"), indent=1)
    ST = json.load(open(sf))
    print(f"start (round 0 error summary, top {len(ST['block'])}): val EM {100 * ST['val']:.1f}\n" + "\n".join("  " + r for r in ST["block"]), flush=True)

    final = {}
    for arm in ("errsum", SHARM):
        stf = f"{OUT}/{arm}.json"
        A_ = json.load(open(stf)) if os.path.exists(stf) else dict(block=list(ST["block"]), val=ST["val"], tried=[], log=[])
        for t in range(len(A_["log"]), ROUNDS):
            tag = f"{arm.replace('+', '_')}_r{t}"
            Fd = run(prompt(A_["block"]), FR, f"{tag}_fit")
            gem.SALT.v = f"{arm}{t}"
            more = rules_of(S.call([dict(role="user", content=MORE.format(rules="\n".join(A_["block"]), pairs=S.pairs_text(Fd)))], 800, temperature=0.7), 3)
            gem.SALT.v = ""
            kept, A_["val"], tested = best_single(A_["block"], more, VR, A_["val"], f"{tag}_es")
            if kept:
                A_["block"].append(kept)
            rec = dict(round=t, fit=100 * Fd.correct.mean(), errsum_kept=kept, errsum_tested=tested)
            if arm != "errsum":
                Fd = run(prompt(A_["block"]), FS, f"{tag}_shadfit")
                L, info = shad_cands(F, Fd, {tuple(x) for x in A_["tried"]})
                k2, A_["val"], t2 = best_single(A_["block"], [x["rule"] for x in L], VR, A_["val"], f"{tag}_sh")
                A_["tried"] += [[x["answer_feature"], x["context"]] for x in L]
                if k2:
                    A_["block"].append(k2)
                rec |= dict(shad_kept=k2, shad_tested=t2)
            rec["val"] = 100 * A_["val"]; A_["log"].append(rec); json.dump(A_, open(stf, "w"), indent=1)
            print(f"[{arm}] round {t}: fit EM {rec['fit']:.1f} | errsum {'kept: ' + kept if kept else 'none kept'} [{' || '.join(tested)}]"
                  + (f" | ShAD {'kept: ' + rec['shad_kept'] if rec['shad_kept'] else 'none kept'} [{' || '.join(rec['shad_tested'])}]" if arm != "errsum" else "")
                  + f" | val EM {rec['val']:.1f}  (spent ${gem.spent()[0]:.2f})", flush=True)
        final[arm] = A_["block"]

    print("\n== test (n=%d)" % len(TE))
    res = {name: run(prompt(b), TE, f"test_{name.replace('+', '_').replace(' ', '_')}") for name, b in
           (("base", []), ("start", ST["block"]), ("errsum", final["errsum"]), (SHARM, final[SHARM]))}
    rng = np.random.default_rng(0)

    def diff(x, y):
        d = (res[x].correct - res[y].correct).to_numpy()
        return 100 * d.mean(), 100 * np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    for k, df in res.items():
        print(f"{k:12s} EM {100 * df.correct.mean():.1f}  F1 {100 * df.f1.mean():.1f}  rules {len({'base': [], 'start': ST['block']}.get(k, final.get(k, [])))}")
    for x, y in (("errsum", "start"), (SHARM, "start"), (SHARM, "errsum")):
        d, se = diff(x, y); print(f"{x} vs {y}: {d:+.1f} (SE {se:.1f}){' *' if abs(d) > 2 * se else ''}")
    print(f"final {SHARM} block:\n" + "\n".join("  " + r for r in final[SHARM]))
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

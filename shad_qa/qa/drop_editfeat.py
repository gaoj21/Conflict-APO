"""Edit-operation features (no embeddings, no LLM naming) -> conditions from question n-grams -> templated rules,
one step on top of TextGrad's final DROP prompt.

  .venv/bin/python qa/drop_editfeat.py [--npool 1000] [--workers 32]
1. TextGrad's final prompt (qa/drop_textgrad.py, arm tg) answers `npool` pool questions
2. features: model answer vs reference, token by token, per token class: what the model ADDED and what it DROPPED.
   Classes come from the token itself: %, $, number, number-with-comma, the word right after a number (unit-like),
   capitalized word, other word, parenthesis, list separator. A feature's relevance is
   P(wrong | the edit occurs) - P(wrong | it does not), over all pool answers (edits that the official normalization
   forgives, e.g. articles, get no relevance by themselves)
3. conditions: for each edit, an L1 logistic regression on the question's word n-grams (C by 5-fold CV) predicts where it
   occurs; the two n-grams with the largest positive weights are its condition
4. rules from a template, no LLM: global ("do not add X" / "keep X") and conditional ("only for questions containing
   'a' or 'b' ...") for the 3 edits ranked by frequency among wrong answers x relevance
5. each rule tested alone on val (500) on top of TextGrad's final prompt; the best one on test (1000)
"""
import json, os, re, sys

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]; opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
NPOOL, W = int(opt("--npool", "1000")), opt("--workers", "32")
sys.argv = ["drop_textgrad.py", "tg", "--workers", W, "--nshad", str(NPOOL)]; sys.path.insert(0, HERE)
import drop_textgrad as TG  # noqa: E402
import editfeat as EF  # noqa: E402

D, S, gem = TG.D, TG.S, TG.gem
gem.TAG.v = f"qa/{D.DS}/editfeat"
OUT = f"{S.OUT}/editfeat2"; os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    TRp = S.sample("train", D.NFIT + D.NVAL + NPOOL)  # the same seed-0 permutation as qa/drop_textgrad.py
    VR = S.val_rows(TRp, D.NFIT, D.NVAL, D.NTE); POOL = TRp[D.NFIT + D.NVAL:]
    TE = S.sample("test", D.NTE)
    H = json.load(open(f"{TG.OUT}/tg.json")); sysp = H["sys"]
    base_v = TG.run(sysp, [], VR, "val").correct.mean(); base_t = TG.run(sysp, [], TE, "test")
    print(f"TextGrad final prompt: val EM {100 * base_v:.1f}, test EM {100 * base_t.correct.mean():.1f}", flush=True)
    A = TG.run(sysp, [], POOL, "shadfit"); A["pred"] = A.pred.fillna("").astype(str); A["gold"] = A.gold.astype(str)
    print(f"pool {len(A)} answers, wrong {100 * (1 - A.correct.mean()):.1f}%", flush=True)
    rules = EF.candidates(A)
    json.dump(rules, open(f"{OUT}/rules.json", "w"), indent=1)
    res = []
    for i, r in enumerate(rules):
        v = TG.run(sysp, [r["rule"]], VR, "val").correct.mean(); res.append((v, r["rule"]))
        print(f"[val] {r['kind']:9s} {100 * v:.1f} (TextGrad final {100 * base_v:.1f})  {r['rule']}", flush=True)
    bv, br = max(res); json.dump(dict(rule=br, val=bv, base_val=base_v), open(f"{OUT}/best.json", "w"))
    rng = np.random.default_rng(0)
    if bv >= base_v:
        T = TG.run(sysp, [br], TE, "test"); d = (T.correct - base_t.correct).to_numpy()
        se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"\n== test: TextGrad final {100 * base_t.correct.mean():.1f} | + best edit rule {100 * T.correct.mean():.1f}  ({100 * d.mean():+.1f}, SE {100 * se:.1f})\n   rule: {br}")
    else:
        print("\nno rule beats TextGrad's final prompt on val; test not run")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

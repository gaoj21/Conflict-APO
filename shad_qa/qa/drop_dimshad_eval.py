"""Evaluation-dimension ShAD rules (qa/dimshad.py) on top of TextGrad's final prompt: each rule alone on val (500), the
best on test (1000); compared with the one-step edit-feature rule and the error-summary rule on the same prompt.

  QA_DS=<ds> .venv/bin/python qa/drop_dimshad_eval.py [--workers 32]
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "32"
sys.argv = ["drop_textgrad.py", "tg", "--workers", W]; sys.path.insert(0, HERE)
import drop_textgrad as TG  # noqa: E402

D, S, gem = TG.D, TG.S, TG.gem
gem.TAG.v = f"qa/{D.DS}/dimshad_eval"
OUT = f"{S.OUT}/dimshad"

if __name__ == "__main__":
    TRp = S.sample("train", D.NFIT + D.NVAL); VR = S.val_rows(TRp, D.NFIT, D.NVAL, D.NTE); TE = S.sample("test", D.NTE)
    sysp = json.load(open(f"{TG.OUT}/tg.json"))["sys"]
    base_v = TG.run(sysp, [], VR, "val").correct.mean(); base_t = TG.run(sysp, [], TE, "test")
    rules = json.load(open(f"{OUT}/rules.json")); res = []
    for r in rules:
        v = TG.run(sysp, [r["rule"]], VR, "val").correct.mean(); res.append((v, r["rule"]))
        print(f"[val] {100 * v:.1f} (TextGrad final {100 * base_v:.1f})  [{r['dim']} {r['side']}, {r['kind']}] {r['rule']}", flush=True)
    bv, br = max(res); json.dump(dict(rule=br, val=bv, base_val=base_v, candidates=[dict(rule=r, val=v) for v, r in res]), open(f"{OUT}/best.json", "w"), indent=1)
    rng = np.random.default_rng(0)
    def se(d): return np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    T = TG.run(sysp, [br], TE, "test") if bv >= base_v else base_t
    print(f"\n== test: TextGrad final {100 * base_t.correct.mean():.1f} | + best dimension rule {100 * T.correct.mean():.1f}"
          f"{'' if bv >= base_v else ' (no rule beat TextGrad on val: TextGrad kept)'}\n   rule: {br}")
    for name, f in (("TextGrad", None), ("error summary", f"{S.OUT}/errsum_ctrl/best.json"), ("edit features", f"{S.OUT}/editfeat2/best.json")):
        if f is None:
            O = base_t
        elif os.path.exists(f):
            b = json.load(open(f)); O = TG.run(sysp, [b["rule"]], TE, "test") if b["val"] >= b["base_val"] else base_t
        else:
            continue
        d = (T.correct - O.correct).to_numpy(float)
        print(f"   dimension rule vs {name:14s} {100 * d.mean():+.1f} (SE {100 * se(d):.1f})  [{name} test {100 * O.correct.mean():.1f}]")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

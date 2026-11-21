"""Control for qa/drop_editfeat.py: error summarization on the same TextGrad final prompt, same pool answers, same
number of candidates (6), each tested alone on the same val, the best on the same test.

  .venv/bin/python qa/drop_editfeat_ctrl.py [--workers 32]
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "32"
sys.argv = ["drop_editfeat.py", "--workers", W]; sys.path.insert(0, HERE)
import drop_editfeat as EF  # noqa: E402

TG, D, S, gem = EF.TG, EF.D, EF.S, EF.gem
gem.TAG.v = f"qa/{D.DS}/editfeat_ctrl"
OUT = f"{S.OUT}/errsum_ctrl"; os.makedirs(OUT, exist_ok=True)

ASK = """A model answers reading-comprehension {kind}. Below are its answers with the reference answers (mostly wrong ones).

{pairs}

Write up to 6 short rules for how to write the final answer line so that it matches the reference answers better. Order them from most to least important. Reply with only the rules, one per line, each starting with "- "."""
KIND = dict(drop="questions about passages (sports reports, history articles)",
            musique="multi-hop questions over Wikipedia paragraphs", hotpot="multi-hop questions over Wikipedia paragraphs")

if __name__ == "__main__":
    TRp = S.sample("train", D.NFIT + D.NVAL + EF.NPOOL)  # the same seed-0 permutation as qa/drop_textgrad.py
    VR = S.val_rows(TRp, D.NFIT, D.NVAL, D.NTE); POOL = TRp[D.NFIT + D.NVAL:]
    TE = S.sample("test", D.NTE)
    sysp = json.load(open(f"{TG.OUT}/tg.json"))["sys"]
    base_v = TG.run(sysp, [], VR, "val").correct.mean(); base_t = TG.run(sysp, [], TE, "test")
    A = TG.run(sysp, [], POOL, "shadfit")
    rules = D.rules_of(S.call([dict(role="user", content=ASK.format(kind=KIND[D.DS], pairs=S.pairs_text(A)))], 1000), 6)
    res = []
    for r in rules:
        v = TG.run(sysp, [r], VR, "val").correct.mean(); res.append((v, r))
        print(f"[val] {100 * v:.1f} (TextGrad final {100 * base_v:.1f})  {r}", flush=True)
    bv, br = max(res); rng = np.random.default_rng(0)
    json.dump(dict(rule=br, val=bv, base_val=base_v, candidates=[dict(rule=r, val=v) for v, r in res]), open(f"{OUT}/best.json", "w"), indent=1)
    if bv >= base_v:
        T = TG.run(sysp, [br], TE, "test"); d = (T.correct - base_t.correct).to_numpy()
        se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"\n== test: TextGrad final {100 * base_t.correct.mean():.1f} | + best error-summary rule {100 * T.correct.mean():.1f} "
              f"({100 * d.mean():+.1f}, SE {100 * se:.1f})\n   rule: {br}")
        import hashlib, pandas as pd
        er = json.load(open(f"{EF.OUT}/best.json"))["rule"]  # the edit-feature rule chosen on val by qa/drop_editfeat.py
        Te = TG.run(sysp, [er], TE, "test"); d2 = (Te.correct - T.correct).to_numpy()
        se2 = np.std([d2[rng.integers(0, len(d2), len(d2))].mean() for _ in range(2000)])
        print(f"   edit-feature rule vs error-summary rule: {100 * d2.mean():+.1f} (SE {100 * se2:.1f})")
    else:
        print("\nno error-summary rule beats TextGrad's final prompt on val")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

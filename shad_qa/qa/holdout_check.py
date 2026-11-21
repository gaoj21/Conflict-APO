"""Diagnostic: do the val gains of the rules hold on fresh train-split questions (train positions 2100-2599, never used)?
  QA_DS=<ds> .venv/bin/python qa/holdout_check.py [--workers 32]"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "32"
sys.argv = ["drop_textgrad.py", "tg", "--workers", W]; sys.path.insert(0, HERE)
import drop_textgrad as TG  # noqa: E402

D, S, gem = TG.D, TG.S, TG.gem
gem.TAG.v = f"qa/{D.DS}/holdout_check"
if __name__ == "__main__":
    TRp = S.sample("train", 2600); VR, HO = TRp[D.NFIT:D.NFIT + D.NVAL], TRp[2100:2600]; TE = S.sample("test", D.NTE)
    sysp = json.load(open(f"{TG.OUT}/tg.json"))["sys"]
    arms = {"TextGrad": []}
    for name, f in (("+ error summary", "errsum_ctrl/best.json"), ("+ edit features", "editfeat2/best.json"), ("+ dimension rule", "dimshad/best.json")):
        arms[name] = [json.load(open(f"{S.OUT}/{f}"))["rule"]]
    arms["+ 3 dimension rules"] = json.load(open(f"{S.OUT}/dimshad/combined.json"))["block"]
    rng = np.random.default_rng(0)
    def se(d): return np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    res = {k: (TG.run(sysp, b, VR, "val"), TG.run(sysp, b, HO, "holdout"), TG.run(sysp, b, TE, "test")) for k, b in arms.items()}
    b = res["TextGrad"]
    print(f"{'arm':22s} {'val (used to choose)':>22s} {'fresh train questions':>24s} {'test (dev split)':>20s}")
    for k, (v, h, t) in res.items():
        cell = lambda x, y: f"{100 * x.correct.mean():5.1f}" + ("" if k == "TextGrad" else f" ({100 * (x.correct - y.correct).mean():+.1f}±{100 * se((x.correct - y.correct).to_numpy(float)):.1f})")
        print(f"{k:22s} {cell(v, b[0]):>22s} {cell(h, b[1]):>24s} {cell(t, b[2]):>20s}")
    print(f"done, spent ${gem.spent()[0]:.2f}")

"""Several evaluation-dimension ShAD rules together, added one by one (as round-by-round APO): the global rules of
qa/dimshad.py ranked by cost weight x share of answers; each is added to the block if val EM does not drop; the final
block on test, against TextGrad, the single best dimension rule, the error-summary rule and the edit-feature rule.

  QA_DS=<ds> .venv/bin/python qa/drop_dimshad_combine.py [--workers 32]
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "32"
SRC = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else "dimshad"  # or dimlearn
sys.argv = ["drop_textgrad.py", "tg", "--workers", W]; sys.path.insert(0, HERE)
import drop_textgrad as TG  # noqa: E402

D, S, gem = TG.D, TG.S, TG.gem
gem.TAG.v = f"qa/{D.DS}/{SRC}_combine"
OUT = f"{S.OUT}/{SRC}"

if __name__ == "__main__":
    TRp = S.sample("train", D.NFIT + D.NVAL); VR = S.val_rows(TRp, D.NFIT, D.NVAL, D.NTE); TE = S.sample("test", D.NTE)
    sysp = json.load(open(f"{TG.OUT}/tg.json"))["sys"]
    cur = TG.run(sysp, [], VR, "val").correct.mean(); base_t = TG.run(sysp, [], TE, "test"); block = []
    print(f"start: TextGrad final, val {100 * cur:.1f}", flush=True)
    for r in json.load(open(f"{OUT}/ranked.json")):
        v = TG.run(sysp, block + [r["rule"]], VR, "val").correct.mean(); keep = v >= cur
        if keep:
            block.append(r["rule"]); cur = v
        print(f"[val] {100 * v:.1f} {'keep' if keep else 'drop'} (block val {100 * cur:.1f})  [{r['dim']} {r['side']}, score {r['score']:.3f}] {r['rule']}", flush=True)
    json.dump(dict(block=block, val=cur), open(f"{OUT}/combined.json", "w"), indent=1)
    T = TG.run(sysp, block, TE, "test")
    rng = np.random.default_rng(0)
    def se(d): return np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    print(f"\n== test: TextGrad final {100 * base_t.correct.mean():.1f} | + combined dimension rules ({len(block)}) {100 * T.correct.mean():.1f}\n   block:\n" + "\n".join("     " + b for b in block))
    for name, f in (("TextGrad", None), ("single dimension rule", f"{OUT}/best.json"), ("error summary", f"{S.OUT}/errsum_ctrl/best.json"),
                    ("edit features", f"{S.OUT}/editfeat2/best.json")):
        if f is None:
            O = base_t
        elif os.path.exists(f):
            b = json.load(open(f)); O = TG.run(sysp, [b["rule"]], TE, "test") if b["val"] >= b["base_val"] else base_t
        else:
            continue
        d = (T.correct - O.correct).to_numpy(float)
        print(f"   combined vs {name:22s} {100 * d.mean():+.1f} (SE {100 * se(d):.1f})  [{name} test {100 * O.correct.mean():.1f}]")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

"""Final table of the QA experiments (run_qa_llm.sh), read from the saved answers only (no model calls).

  python qa/qa_report.py [drop musique hotpot]
Per dataset, test EM (x100) of
  TextGrad            : official TextGrad, final system prompt (qa/drop_textgrad.py tg)
  + error summary     : TextGrad's final prompt + the best of 6 error-summary rules on val (qa/drop_editfeat_ctrl.py)
  + ShAD edit (1 step): TextGrad's final prompt + the best of 6 edit-feature rules on val (qa/drop_editfeat.py)
  TextGrad x ShAD edit: ShAD edit rules added every 3 TextGrad steps (qa/drop_textgrad.py tg+edit)
with the paired difference to TextGrad and to the error-summary arm (bootstrap SE over test questions).
"""
import hashlib, json, os, sys

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/baselines")
import gem  # noqa: E402

DSS = sys.argv[1:] or ["drop", "musique", "hotpot"]


def out_dir(ds):  # the results folder of qa/shadqa.py for this dataset
    return f"{gem.OUT}_qa/{ds}/shadqa"


def answers(tgdir, sysp, block):
    f = f"{tgdir}/answers_test_{hashlib.sha1((sysp + chr(0) + chr(10).join(block)).encode()).hexdigest()[:8]}.csv"
    return pd.read_csv(f).set_index("qid").correct if os.path.exists(f) else None


def diff(x, y):
    d = (x - y.reindex(x.index)).to_numpy(float); rng = np.random.default_rng(0)
    se = np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return f"{100 * d.mean():+5.1f} (SE {100 * se:.1f}){' *' if abs(d.mean()) > 2 * se else '  '}"


for ds in DSS:
    o = out_dir(ds); tgd = f"{o}/textgrad"
    print(f"\n== {ds}  ({o})")
    if not os.path.exists(f"{tgd}/tg.json"):
        print("   TextGrad not run yet"); continue
    H = json.load(open(f"{tgd}/tg.json")); base = answers(tgd, H["sys"], [])
    if base is None:
        print(f"   TextGrad: {len(H['log'])} steps done, test not run yet"); continue
    arms = {"TextGrad": base}; rules = {}
    for name, f in (("+ error summary", f"{o}/errsum_ctrl/best.json"), ("+ ShAD edit (1 step)", f"{o}/editfeat2/best.json")):
        if os.path.exists(f):
            b = json.load(open(f)); rules[name] = b["rule"]
            if b["val"] < b["base_val"]:
                arms[name] = base; rules[name] += "   (no rule beat TextGrad on val: TextGrad's prompt kept)"
            else:
                arms[name] = answers(tgd, H["sys"], [b["rule"]])
    if os.path.exists(f"{tgd}/tg+edit.json"):
        R = json.load(open(f"{tgd}/tg+edit.json")); arms["TextGrad x ShAD edit"] = answers(tgd, R["sys"], R["block"])
        rules["TextGrad x ShAD edit"] = " / ".join(R["block"]) or "(no rule kept)"
    es = arms.get("+ error summary")
    print(f"   {'arm':22s} {'test EM':>8s}   {'vs TextGrad':18s} {'vs error summary':18s}")
    for k, v in arms.items():
        if v is None:
            print(f"   {k:22s} {'running':>8s}"); continue
        print(f"   {k:22s} {100 * v.mean():8.1f}   {diff(v, base) if k != 'TextGrad' else '':18s} "
              f"{diff(v, es) if es is not None and k not in ('TextGrad', '+ error summary') else '':18s}")
    for k, r in rules.items():
        print(f"   rule [{k}]: {r}")

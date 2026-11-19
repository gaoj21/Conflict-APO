"""Which candidate edits are REPLICABLE conflicts (a significant gain and a significant loss on train AND on test)?

  select: python scripts/rb2_conflicts.py select <train_dir> <variants.json> <test_variants_out.json>
          -> candidates with any significant train gain go to the test set
  report: python scripts/rb2_conflicts.py report <train_dir> <test_dir> <variants.json>

Objectives: accuracy per kind (points) and output tokens (cheaper = gain). Significance: paired bootstrap 95% CI.
"""

import glob
import json
import os
import sys

import numpy as np

KINDS = ["focus", "factuality", "safety", "math", "precise_if"]
OBJ = KINDS + ["cost"]


def summ(d, ids):
    v = {k: np.mean([d[i]["pred"] == d[i]["gt"] for i in ids if d[i]["kind"] == k] or [np.nan]) for k in KINDS}
    v["cost"] = np.mean([d[i]["out"] or 0 for i in ids])
    return v


def effects(d, base, n_boot=400):
    ids = sorted(base)
    rng = np.random.default_rng(0)

    def diff(a, b):  # accuracy points; cost: % fewer tokens (positive = better)
        return {k: (100 * (a[k] - b[k]) if k != "cost" else 100 * (1 - a[k] / b[k])) for k in OBJ}

    point = diff(summ(d, ids), summ(base, ids))
    bs = []
    for _ in range(n_boot):
        sub = [ids[i] for i in rng.choice(len(ids), len(ids))]
        bs.append(diff(summ(d, sub), summ(base, sub)))
    out = {}
    for k in OBJ:
        lo, hi = np.nanpercentile([b[k] for b in bs], [2.5, 97.5])
        out[k] = {"d": round(float(point[k]), 1), "sig": int(np.sign(point[k])) if (lo > 0 or hi < 0) else 0}
    return out


def load(dirname):
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(dirname, "*.json"))}


def fmt(e):
    return " ".join(f"{k[:5]}{e[k]['d']:+6.1f}{'*' if e[k]['sig'] else ' '}" for k in OBJ)


def main():
    mode = sys.argv[1]
    if mode == "select":
        train, variants = load(sys.argv[2]), json.load(open(sys.argv[3]))
        sel = {"base": variants["base"]}
        stats = {}
        for lab in sorted(v for v in variants if v != "base" and v in train):
            e = effects(train[lab], train["base"])
            stats[lab] = e
            gains = [k for k in OBJ if e[k]["sig"] > 0]
            losses = [k for k in OBJ if e[k]["sig"] < 0]
            print(f"{lab}: {fmt(e)}   gains={gains} losses={losses}")
            if gains:
                sel[lab] = variants[lab]
        json.dump(stats, open(os.path.join(sys.argv[2], "_effects.json.txt"), "w"), indent=1)
        json.dump(sel, open(sys.argv[4], "w"), indent=1)
        print(f"\n{len(sel) - 1} of {len(stats)} candidates have a significant train gain -> test set")
    else:
        train, test, variants = load(sys.argv[2]), load(sys.argv[3]), json.load(open(sys.argv[4]))
        n_conf_train = n_rep = 0
        for lab in sorted(v for v in variants if v != "base" and v in train):
            et = effects(train[lab], train["base"])
            gt, lt = {k for k in OBJ if et[k]["sig"] > 0}, {k for k in OBJ if et[k]["sig"] < 0}
            n_conf_train += bool(gt and lt)
            if lab not in test:
                continue
            ee = effects(test[lab], test["base"])
            ge, le = {k for k in OBJ if ee[k]["sig"] > 0}, {k for k in OBJ if ee[k]["sig"] < 0}
            rep_gain, rep_loss = gt & ge, lt & le
            tag = "REPLICATED CONFLICT" if rep_gain and rep_loss else ("replicated gain" if rep_gain else "")
            n_rep += bool(rep_gain and rep_loss)
            print(f"{lab}\n  train {fmt(et)}\n  test  {fmt(ee)}   {tag} {sorted(rep_gain)} vs {sorted(rep_loss)}")
            print(f"  edit: {variants[lab][len(variants['base']) + 1:][:200]}")
        print(f"\nconflicts on train: {n_conf_train}; replicated conflicts (gain and loss significant on both): {n_rep}")


if __name__ == "__main__":
    main()

"""New roots with a ROUTE split that the optimizer never sees (used only to route attributes and pick the merged
prompt): results/claude_hs2r (the 1159 HS2 train rows reshuffled into fit/sel/route 386/386/387) and
results/claude_brr (fit = old fit[:400], sel = old sel[:400], route = 400 new BRIGHTER train rows 1200-1599).
Test is unchanged. v0's cached judgments are carried over by id (route rows of brr are new and get judged later).

  python scripts/claude_prep_route.py
"""
import json, os, shutil
import numpy as np, pandas as pd


def rows(root, split):
    return [json.loads(l) for l in open(f"{root}/chunks/{split}_00.jsonl")]


def write(root, parts, gold):
    for d in ("chunks", "prompts", "out"):
        os.makedirs(f"{root}/{d}", exist_ok=True)
    for sp, rs in parts.items():
        with open(f"{root}/chunks/{sp}_00.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gold.to_csv(f"{root}/gold.csv", index=False)


def carry_v0(old, new, assign):
    """Old v0 judgments (fit/sel/test) re-split by the new assignment {id: split}."""
    v = pd.concat([pd.read_csv(f"{old}/out/v0_{s}.csv") for s in ("fit", "sel")])
    for sp in set(assign.values()):
        ids = [i for i, s in assign.items() if s == sp]
        d = v[v.id.isin(ids)]
        if len(d):
            d.to_csv(f"{new}/out/v0_{sp}.csv", index=False)
    shutil.copy(f"{old}/out/v0_test.csv", f"{new}/out/v0_test.csv")


for ds, new in (("hs2", "results/claude_hs2r"), ("br", "results/claude_brr")):
    old = f"results/claude_{ds}"
    g = pd.read_csv(f"{old}/gold.csv")
    if ds == "hs2":
        pool = rows(old, "fit") + rows(old, "sel")
        perm = np.random.default_rng(0).permutation(len(pool))
        cut = {"fit": perm[:386], "sel": perm[386:772], "route": perm[772:]}
        parts = {sp: [pool[i] for i in sorted(ix)] for sp, ix in cut.items()}
    else:
        parts = {"fit": rows(old, "fit")[:400], "sel": rows(old, "sel")[:400]}
        tr = pd.read_parquet("data/BRIGHTER/BRIGHTER-train.parquet").iloc[1200:1600].reset_index(drop=True)
        A = ["anger", "fear", "joy", "sadness", "surprise"]
        parts["route"] = [{"id": f"route{i:04d}", "user": f"## Text:\n{r.text}\n\n## Your evaluation:"} for i, r in tr.iterrows()]
        extra = pd.DataFrame([dict(id=f"route{i:04d}", split="route", **{a: float(r[a]) for a in A},
                                   words=len(str(r.text).split()), z=float(np.log1p(len(str(r.text).split()))))
                              for i, r in tr.iterrows()])
        g = pd.concat([g, extra], ignore_index=True)
    parts["test"] = rows(old, "test")
    assign = {r["id"]: sp for sp, rs in parts.items() for r in rs}
    g = g[g.id.isin(assign)].copy()
    g["split"] = g.id.map(assign)
    write(new, parts, g)
    for f in ("attrs.txt", "format.txt", "prompts/v0.txt"):
        shutil.copy(f"{old}/{f}", f"{new}/{f}")
    carry_v0(old, new, assign)
    print(new, g.split.value_counts().to_dict(), {s: len(pd.read_csv(f"{new}/out/v0_{s}.csv")) for s in ("fit", "sel", "test")},
          "v0_route" if os.path.exists(f"{new}/out/v0_route.csv") else "v0_route: to judge")

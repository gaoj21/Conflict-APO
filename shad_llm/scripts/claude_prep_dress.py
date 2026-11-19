"""DREsS_New (Yoo et al., ACL 2025; access by consent form, no redistribution): EFL undergraduate essays scored by
English-education experts on content / organization / language (1-5, half points). Duplicate essays averaged; split by
writing prompt (no prompt in two splits), filling fit / sel / test to ~400 essays each (seed 0).
  python scripts/claude_prep_dress.py -> results/claude_dress/"""
import json, os
import numpy as np, pandas as pd

OUT = "results/claude_dress"
A = ["content", "organization", "language"]
d = pd.read_csv("data/DREsS/DREsS_New.tsv", sep="\t")
d["essay"] = d.essay.astype(str).str.strip()
d = d[d.essay.str.split().str.len() >= 20]
g = d.groupby(["prompt", "essay"])[A].mean().reset_index().dropna(subset=A)
P = g.prompt.unique()
perm = np.random.default_rng(0).permutation(len(P))
cnt = g.groupby("prompt").size()
split_of, n, cur = {}, {"fit": 0, "sel": 0, "test": 0}, iter(("fit", "sel", "test"))
sp = next(cur)
for j in perm:
    if n[sp] >= 400:
        sp = next(cur, None)
        if sp is None:
            break
    split_of[P[j]] = sp; n[sp] += cnt[P[j]]
g["split"] = g.prompt.map(split_of)
for x in ("chunks", "prompts", "out"):
    os.makedirs(f"{OUT}/{x}", exist_ok=True)
gold = []
for sp in ("fit", "sel", "test"):
    part = g[g.split == sp].sample(frac=1, random_state=0)
    with open(f"{OUT}/chunks/{sp}_00.jsonl", "w") as f:
        for i, (_, r) in enumerate(part.iterrows()):
            u = f"## Essay prompt:\n{str(r.prompt).strip()}\n\n## Essay:\n{r.essay}\n\n## Your evaluation:"
            f.write(json.dumps({"id": f"{sp}{i:03d}", "user": u}, ensure_ascii=False) + "\n")
            w = len(r.essay.split())
            gold.append(dict(id=f"{sp}{i:03d}", split=sp, **{a: float(r[a]) for a in A}, words=w, z=float(np.log1p(w))))
pd.DataFrame(gold).to_csv(f"{OUT}/gold.csv", index=False)
open(f"{OUT}/attrs.txt", "w").write("\n".join(A) + "\n")
open(f"{OUT}/format.txt", "w").write(
    "## Output format:\nThink briefly (at most a few sentences), then end your reply with exactly one line of JSON with an "
    "integer from 1 to 5 for each attribute:\n{" + ", ".join(f'"{a}": 1-5' for a in A) + "}\n")
open(f"{OUT}/prompts/v0.txt", "w").write("""You are a careful, calibrated evaluator. Your goal is to produce an accurate evaluation by following the Instructions below.

## Task
Evaluate the Essay, an argumentative essay written by an English-as-a-foreign-language university student in response to the Essay prompt, using the Instructions below.
1. Consider every strength and flaw you find when making your evaluation.
2. Based on the number and severity of the strengths and flaws, assign a value.

## Instructions:
- content: Evaluate whether the paragraphs are well developed and relevant to the argument, supported with strong reasons and examples. Rate from 1 to 5.
- organization: Evaluate whether the argument is effectively structured and easy to follow, with coherent paragraphs and transitions. Rate from 1 to 5.
- language: Evaluate the use of sophisticated vocabulary, correct grammar and spelling, and a range of sentence structures. Rate from 1 to 5.
""")
G = pd.DataFrame(gold)
print(G.split.value_counts().to_dict(), "mean words", round(G.words.mean()))
print(G[A].corr("spearman").round(2))

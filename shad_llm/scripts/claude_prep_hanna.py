"""HANNA (story generation, Chhun et al. 2022, MIT): 1056 stories (96 writing prompts x 11 systems incl. human), six
criteria rated 1-5 by three raters (gold = mean). Split by writing prompt so no prompt appears in two splits:
fit / sel / test = 32 / 32 / 32 prompts (352 stories each, seed 0).  python scripts/claude_prep_hanna.py -> results/claude_hanna/"""
import json, os
import numpy as np, pandas as pd

OUT = "results/claude_hanna"
A = ["relevance", "coherence", "empathy", "surprise", "engagement", "complexity"]
d = pd.read_csv("data/HANNA/hanna_stories_annotations.csv")
g = d.groupby("Story ID").agg(prompt=("Prompt", "first"), story=("Story", "first"),
                              **{a: (a.capitalize(), "mean") for a in A}).reset_index()
P = sorted(g.prompt.unique())
perm = np.random.default_rng(0).permutation(len(P))
split_of = {P[j]: ("fit", "sel", "test")[i // 32] for i, j in enumerate(perm)}
g["split"] = g.prompt.map(split_of)
for x in ("chunks", "prompts", "out"):
    os.makedirs(f"{OUT}/{x}", exist_ok=True)
gold = []
for sp in ("fit", "sel", "test"):
    part = g[g.split == sp].sample(frac=1, random_state=0)
    with open(f"{OUT}/chunks/{sp}_00.jsonl", "w") as f:
        for i, (_, r) in enumerate(part.iterrows()):
            p, s = str(r.prompt).strip(), str(r.story).strip()
            f.write(json.dumps({"id": f"{sp}{i:03d}", "user": f"## Writing prompt:\n{p}\n\n## Story:\n{s}\n\n## Your evaluation:"},
                               ensure_ascii=False) + "\n")
            w = len(s.split())
            gold.append(dict(id=f"{sp}{i:03d}", split=sp, **{a: float(r[a]) for a in A}, words=w, z=float(np.log1p(w))))
pd.DataFrame(gold).to_csv(f"{OUT}/gold.csv", index=False)
open(f"{OUT}/attrs.txt", "w").write("\n".join(A) + "\n")
open(f"{OUT}/format.txt", "w").write(
    "## Output format:\nThink briefly (at most a few sentences), then end your reply with exactly one line of JSON with an "
    "integer from 1 to 5 for each attribute:\n{" + ", ".join(f'"{a}": 1-5' for a in A) + "}\n")
open(f"{OUT}/prompts/v0.txt", "w").write("""You are a careful, calibrated evaluator. Your goal is to produce an accurate evaluation by following the Instructions below.

## Task
Evaluate the Story, written in response to the Writing prompt, using the Instructions below.
1. Consider every strength and flaw you find when making your evaluation.
2. Based on the number and severity of the strengths and flaws, assign a value.

## Instructions:
- relevance: Evaluate how well the story matches its writing prompt. Rate from 1 to 5.
- coherence: Evaluate how much the story makes sense. Rate from 1 to 5.
- empathy: Evaluate how well you understood the characters' emotions. Rate from 1 to 5.
- surprise: Evaluate how surprising the end of the story is. Rate from 1 to 5.
- engagement: Evaluate how much you engaged with the story. Rate from 1 to 5.
- complexity: Evaluate how elaborate the story is. Rate from 1 to 5.
""")
G = pd.DataFrame(gold)
print(G.split.value_counts().to_dict(), "mean words", round(G.words.mean()))
print("rho(human, len):", {a: round(G[[a, "words"]].corr("spearman").iloc[0, 1], 2) for a in A})
print("human inter-attr rho:\n", G[A].corr("spearman").round(2))

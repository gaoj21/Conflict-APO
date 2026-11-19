"""TL;DR axis evaluations (Stiennon et al. 2020, openai/summarize_from_feedback "axis", Reddit part = validation file):
summaries of Reddit posts rated 1-7 on overall / accuracy / coverage / coherence by trained labelers (mean over
duplicate ratings). Split by post (no post in two splits), filling fit / sel / test to ~400 summaries each (seed 0).
  python scripts/claude_prep_tldr.py -> results/claude_tldr/"""
import json, os
import numpy as np, pandas as pd

OUT = "results/claude_tldr"
A = ["overall", "accuracy", "coverage", "coherence"]
d = pd.read_parquet("data/TLDR_axis/validation.parquet")
rows = [dict(pid=i["id"], title=i["title"], post=i["post"], text=s["text"].strip(), **{a: s["axes"].get(a) for a in A})
        for i, s in zip(d["info"], d["summary"])]
g = pd.DataFrame(rows).groupby(["pid", "text"]).agg(title=("title", "first"), post=("post", "first"),
                                                    **{a: (a, "mean") for a in A}).reset_index().dropna(subset=A)
pids = g.pid.unique()
perm = np.random.default_rng(0).permutation(len(pids))
cnt = g.groupby("pid").size()
split_of, n, cur = {}, {"fit": 0, "sel": 0, "test": 0}, iter(("fit", "sel", "test"))
sp = next(cur)
for j in perm:
    if n[sp] >= 400:
        sp = next(cur, None)
        if sp is None:
            break
    split_of[pids[j]] = sp; n[sp] += cnt[pids[j]]
g["split"] = g.pid.map(split_of)
for x in ("chunks", "prompts", "out"):
    os.makedirs(f"{OUT}/{x}", exist_ok=True)
gold = []
for sp in ("fit", "sel", "test"):
    part = g[g.split == sp].sample(frac=1, random_state=0)
    with open(f"{OUT}/chunks/{sp}_00.jsonl", "w") as f:
        for i, (_, r) in enumerate(part.iterrows()):
            u = f"## Reddit post\nTitle: {r.title.strip()}\n{r.post.strip()}\n\n## Summary:\n{r.text}\n\n## Your evaluation:"
            f.write(json.dumps({"id": f"{sp}{i:03d}", "user": u}, ensure_ascii=False) + "\n")
            w = len(r.text.split())
            gold.append(dict(id=f"{sp}{i:03d}", split=sp, **{a: float(r[a]) for a in A}, words=w, z=float(np.log1p(w))))
pd.DataFrame(gold).to_csv(f"{OUT}/gold.csv", index=False)
open(f"{OUT}/attrs.txt", "w").write("\n".join(A) + "\n")
open(f"{OUT}/format.txt", "w").write(
    "## Output format:\nThink briefly (at most a few sentences), then end your reply with exactly one line of JSON with an "
    "integer from 1 to 7 for each attribute:\n{" + ", ".join(f'"{a}": 1-7' for a in A) + "}\n")
open(f"{OUT}/prompts/v0.txt", "w").write("""You are a careful, calibrated evaluator. Your goal is to produce an accurate evaluation by following the Instructions below.

## Task
Evaluate the Summary of the Reddit post using the Instructions below.
1. Consider every strength and flaw you find when making your evaluation.
2. Based on the number and severity of the strengths and flaws, assign a value.

## Instructions:
- overall: Evaluate how good the summary is overall as a summary of the post. Rate from 1 to 7.
- accuracy: Evaluate to what extent the statements in the summary are stated or implied by the post. Rate from 1 to 7.
- coverage: Evaluate how well the summary covers the important information in the post. Rate from 1 to 7.
- coherence: Evaluate how easy the summary is to understand on its own. Rate from 1 to 7.
""")
G = pd.DataFrame(gold)
print(G.split.value_counts().to_dict(), "mean summary words", round(G.words.mean(), 1))
print("rho(human, len):", {a: round(G[[a, "words"]].corr("spearman").iloc[0, 1], 2) for a in A})
print(G[A].corr("spearman").round(2))

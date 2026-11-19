"""MERLIN (Boyd et al. 2014, CC BY-SA 4.0), German part: learner texts from CEFR-oriented exams rated by trained raters
on six analytic criteria (CEFR A1..C2 -> 1..6). Texts with all six criteria rated; random fit / sel / test thirds
(seed 0).  python scripts/claude_prep_merlin.py -> results/claude_merlin/"""
import glob, json, os, re
import numpy as np, pandas as pd

OUT = "results/claude_merlin"
L = ["A1", "A2", "B1", "B2", "C1", "C2"]
CRIT = {"grammar": "Grammatical accuracy", "orthography": "Orthography", "vocab_range": "Vocabulary range",
        "vocab_control": "Vocabulary control", "coherence": "Coherence/Cohesion", "sociolinguistic": "Sociolinguistic appropriateness"}
A = list(CRIT)
rows = []
for f in sorted(glob.glob("data/MERLIN/merlin-text-v1.1/meta_ltext/german/*.txt")):
    t = open(f, encoding="utf-8").read()
    r = {}
    for a, k in CRIT.items():
        m = re.search(re.escape(k) + r":\s*(\S*)", t)
        r[a] = L.index(m.group(1)) + 1 if m and m.group(1) in L else None
    task = re.search(r"Task\s*:\s*(.*?)\s*- ID:", t)
    r["task"] = task.group(1).strip() if task else ""
    r["text"] = t.split("Learner text:", 1)[1].strip()
    rows.append(r)
d = pd.DataFrame(rows).dropna(subset=A).reset_index(drop=True)
perm = np.random.default_rng(0).permutation(len(d))
n3 = len(d) // 3
parts = {"fit": d.iloc[perm[:n3]], "sel": d.iloc[perm[n3:2 * n3]], "test": d.iloc[perm[2 * n3:3 * n3]]}
for x in ("chunks", "prompts", "out"):
    os.makedirs(f"{OUT}/{x}", exist_ok=True)
gold = []
for sp, part in parts.items():
    with open(f"{OUT}/chunks/{sp}_00.jsonl", "w") as f:
        for i, (_, r) in enumerate(part.iterrows()):
            u = f"## Writing task: {r.task}\n\n## Learner text:\n{r.text}\n\n## Your evaluation:"
            f.write(json.dumps({"id": f"{sp}{i:03d}", "user": u}, ensure_ascii=False) + "\n")
            w = len(r.text.split())
            gold.append(dict(id=f"{sp}{i:03d}", split=sp, **{a: float(r[a]) for a in A}, words=w, z=float(np.log1p(w))))
pd.DataFrame(gold).to_csv(f"{OUT}/gold.csv", index=False)
open(f"{OUT}/attrs.txt", "w").write("\n".join(A) + "\n")
open(f"{OUT}/format.txt", "w").write(
    "## Output format:\nThink briefly (at most a few sentences), then end your reply with exactly one line of JSON with an "
    "integer from 1 to 6 for each attribute (1 = CEFR A1, 2 = A2, 3 = B1, 4 = B2, 5 = C1, 6 = C2):\n{"
    + ", ".join(f'"{a}": 1-6' for a in A) + "}\n")
open(f"{OUT}/prompts/v0.txt", "w").write("""You are a careful, calibrated evaluator. Your goal is to produce an accurate evaluation by following the Instructions below.

## Task
Evaluate the Learner text, written in German by a learner of German for a language exam, using the Instructions below. Each value is a CEFR level from 1 (A1) to 6 (C2).
1. Consider every strength and flaw you find when making your evaluation.
2. Based on the number and severity of the strengths and flaws, assign a value.

## Instructions:
- grammar: Evaluate grammatical accuracy. Rate from 1 to 6.
- orthography: Evaluate spelling and punctuation. Rate from 1 to 6.
- vocab_range: Evaluate the breadth of the vocabulary used. Rate from 1 to 6.
- vocab_control: Evaluate how accurately and appropriately the vocabulary is used. Rate from 1 to 6.
- coherence: Evaluate coherence and cohesion: how well the text is organized and connected. Rate from 1 to 6.
- sociolinguistic: Evaluate sociolinguistic appropriateness: register, politeness and conventions suited to the task. Rate from 1 to 6.
""")
G = pd.DataFrame(gold)
print(G.split.value_counts().to_dict(), "mean words", round(G.words.mean()))

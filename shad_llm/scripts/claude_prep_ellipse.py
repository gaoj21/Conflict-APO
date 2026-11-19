"""ELLIPSE (English-language-learner essays, grades 8-12; Crossley et al. 2023, CC BY 4.0) for the per-call judge:
six analytic scores 1-5 (averages of two raters, half points). fit / sel = 400 / 400 random train essays, test = 480
random test essays (seed 0).  python scripts/claude_prep_ellipse.py  -> results/claude_ell/"""
import json, os
import numpy as np, pandas as pd

OUT = "results/claude_ell"
A = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
tr = pd.read_csv("data/ELLIPSE/ELLIPSE_Final_github_train.csv")
te = pd.read_csv("data/ELLIPSE/ELLIPSE_Final_github_test.csv")
rng = np.random.default_rng(0)
i_tr, i_te = rng.permutation(len(tr)), rng.permutation(len(te))
parts = {"fit": tr.iloc[i_tr[:400]], "sel": tr.iloc[i_tr[400:800]], "test": te.iloc[i_te[:480]]}
for d in ("chunks", "prompts", "out"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
gold = []
for sp, d in parts.items():
    with open(f"{OUT}/chunks/{sp}_00.jsonl", "w") as f:
        for i, (_, r) in enumerate(d.iterrows()):
            text = str(r.full_text).strip()
            f.write(json.dumps({"id": f"{sp}{i:03d}", "user": f"## Essay:\n{text}\n\n## Your evaluation:"}, ensure_ascii=False) + "\n")
            w = len(text.split())
            gold.append(dict(id=f"{sp}{i:03d}", split=sp, **{a: float(r[a.capitalize()]) for a in A}, words=w, z=float(np.log1p(w))))
pd.DataFrame(gold).to_csv(f"{OUT}/gold.csv", index=False)
open(f"{OUT}/attrs.txt", "w").write("\n".join(A) + "\n")
open(f"{OUT}/format.txt", "w").write(
    "## Output format:\nThink briefly (at most a few sentences), then end your reply with exactly one line of JSON with an "
    "integer from 1 to 5 for each attribute:\n{" + ", ".join(f'"{a}": 1-5' for a in A) + "}\n")
open(f"{OUT}/prompts/v0.txt", "w").write("""You are a careful, calibrated evaluator. Your goal is to produce an accurate evaluation by following the Instructions below.

## Task
Evaluate the Essay, written by an English language learner in grades 8-12, using the Instructions below.
1. Consider every strength and flaw you find when making your evaluation.
2. Based on the number and severity of the strengths and flaws, assign a value.

## Instructions:
- cohesion: Evaluate how well the text organization is controlled and how well ideas are connected across sentences and paragraphs. Rate from 1 to 5.
- syntax: Evaluate the range and accuracy of sentence structures. Rate from 1 to 5.
- vocabulary: Evaluate the range and precision of the vocabulary used. Rate from 1 to 5.
- phraseology: Evaluate the use of phrases such as idioms, collocations and lexical bundles. Rate from 1 to 5.
- grammar: Evaluate the accuracy of grammar and usage. Rate from 1 to 5.
- conventions: Evaluate spelling, capitalization and punctuation. Rate from 1 to 5.
""")
print(pd.DataFrame(gold).split.value_counts().to_dict(), "mean words", round(pd.DataFrame(gold).words.mean()))

"""Main table: test Spearman (mean over attributes) for the three judge/optimizer models. Per dataset: the judge's raw
scores and + ShAD (ours) with the gain over raw.
  python scripts/main_table_3judges.py   -> paper/tables/main_3judges_rows.tex
"""
import pandas as pd

RUNS = [("Gemma 4 26B", "results/gemma/results.csv"), ("GPT-5.6 luna", "wgc_gpt/results/gpt56luna/results.csv"),
        ("GPT-5 nano", "wgc_gpt/results/gpt5nano/results.csv")]
DS = ["hs2", "hanna", "ell", "merlin"]
METHODS = [("v0", r"$\vz$ (no optimization)"), ("dspy_zeroshot", "DSPy zero-shot"), ("fewshot", "Few-shot"),
           ("opro", "OPRO"), ("textgrad_ccc", "TextGrad (CCC)"), ("textgrad_single", "TextGrad (per attribute)"),
           ("dspy_gepa", "GEPA"), ("dspy_mipro", "MIPROv2")]
f = lambda v: f"{100 * v:.1f}"  # Spearman x 100
r1 = lambda v: round(100 * v, 1)
out = []
for name, path in RUNS:
    d = pd.read_csv(path)
    d = d[d.seed.astype(str).isin(["seed0", "0"])].set_index(["method", "ds"])
    row = lambda m, c: [d.loc[(m, x), c] for x in DS] + [sum(d.loc[(m, x), c] for x in DS) / 4]
    out += [r"\midrule", rf"\multicolumn{{11}}{{l}}{{\textit{{{name}}}}} \\"]
    for m, lab in METHODS:
        raw, dec = row(m, "raw"), row(m, "readout")
        cells = [f"{f(r)} & {f(o)}" + rf"\gain{{{r1(o) - r1(r):.1f}}}" for r, o in zip(raw, dec)]  # gain of the rounded values
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
open("paper/tables/main_3judges_rows.tex", "w").write("\n".join(out) + "\n")
print("\n".join(out[:4]))

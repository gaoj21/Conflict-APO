"""Figure 2 of the paper: (a) HS2 test, Spearman of each attribute with log length: humans vs judges (v0);
(b) every candidate edit of the TextGrad-style runs (5 runs x 12 candidates x attributes): per-attribute Spearman
change on sel, raw scores (x) vs decoded scores (y).  -> paper/figures/axis.pdf"""
import json
import matplotlib.pyplot as plt, numpy as np

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw=dict(width_ratios=[1, 1.15]))
att = ["help", "corr", "coh", "cplx", "verb"]
hum = [.07, .11, .01, .37, .65]; hai = [.08, -.09, -.17, .38, .29]; son = [.04, -.04, -.10, .40, .09]
x = np.arange(5)
for i, (v, lab, c) in enumerate([(hum, "humans", "#2b3a4a"), (hai, "Haiku judge", "#7fb4e6"), (son, "Sonnet judge", "#f29aa8")]):
    a1.bar(x + (i - 1) * .27, v, .27, label=lab, color=c, ec="k", lw=.4)
a1.axhline(0, color="k", lw=.6); a1.set_xticks(x, att); a1.set_ylabel(r"Spearman with $\log$ length")
a1.set_title("(a) HS2: who rewards length", fontsize=10); a1.legend(fontsize=8, frameon=False)
cols = {"hs2": "#7fb4e6", "hs2r": "#7fb4e6", "br": "#7cc9a6", "brr": "#7cc9a6", "se": "#f2b36b"}
names = {"hs2": "HS2", "br": "BRIGHTER", "se": "SummEval"}
seen = set()
for r, c in cols.items():
    A = open(f"results/claude_{r}/attrs.txt").read().split()
    for l in open(f"results/claude_{r}/opt/text/log.jsonl"):
        for cd in json.loads(l)["cands"]:
            for k, (dr, do) in enumerate(zip(cd["d_raw"], cd["d_ro"])):
                verb = A[k] == "verbosity"
                lab = None
                key = "verb" if verb else names.get(r)
                if key and key not in seen:
                    lab = "HS2 verbosity" if verb else names[r]; seen.add(key)
                a2.scatter(dr, do, s=26 if verb else 12, c="#d0473f" if verb else c, marker="^" if verb else "o",
                           ec="k", lw=.3, alpha=.85, label=lab, zorder=3 if verb else 2)
lim = [-.35, .45]
a2.plot(lim, lim, "k--", lw=.7); a2.axhline(0, color="k", lw=.5); a2.axvline(0, color="k", lw=.5)
a2.set_xlim(lim); a2.set_ylim(-.35, .2); a2.set_xlabel(r"$\Delta$ Spearman on sel, raw scores")
a2.set_ylabel(r"$\Delta$ Spearman, decoded"); a2.set_title("(b) every candidate edit, per attribute", fontsize=10)
a2.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("paper/figures/axis.pdf"); fig.savefig("paper/figures/axis.png", dpi=150)

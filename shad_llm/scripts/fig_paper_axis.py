"""Figure 2 of the paper: (a) HS2 test, Spearman of each attribute with log length: humans vs judges (v0);
(b) where the readout's gain comes from (v0, test): raw / own score + length / all scores / all scores + length;
(c) every candidate edit of the five optimization runs (HS2 x2, HANNA, ELLIPSE, MERLIN; 12 candidates each):
per-attribute Spearman change on sel, raw scores (x) vs decoded scores (y).  -> paper/figures/axis.pdf"""
import json
import matplotlib.pyplot as plt, numpy as np

fig, (a1, a3, a2) = plt.subplots(1, 3, figsize=(15, 3.5), gridspec_kw=dict(width_ratios=[1, 1.1, 1.1]))
att = ["help", "corr", "coh", "cplx", "verb"]
hum = [7, 11, 1, 37, 65]; hai = [8, -9, -17, 38, 29]; son = [4, -4, -10, 40, 9]  # Spearman x 100
x = np.arange(5)
for i, (v, lab, c) in enumerate([(hum, "humans", "#2b3a4a"), (hai, "Haiku judge", "#7fb4e6"), (son, "Sonnet judge", "#f29aa8")]):
    a1.bar(x + (i - 1) * .27, v, .27, label=lab, color=c, ec="k", lw=.4)
a1.axhline(0, color="k", lw=.6); a1.set_xticks(x, att); a1.set_ylabel(r"Spearman with $\log$ length")
a1.set_title("(a) HS2: who rewards length", fontsize=10); a1.legend(fontsize=8, frameon=False)

# (b) decoder ablation on v0, test mean Spearman (scripts/readout_ablation.py)
AB = {"HS2": [33.7, 44.1, 36.9, 45.2], "HANNA": [38.3, 46.0, 40.2, 47.5],
      "ELLIPSE": [37.6, 41.1, 52.7, 55.3], "MERLIN": [52.0, 84.8, 68.5, 84.6]}
labs = ["raw", "own score + length", "all scores", "all scores + length"]
cl = ["#c9c9c9", "#f2d06b", "#9fd6b4", "#2f9e5b"]
xs = np.arange(len(AB))
for j in range(4):
    a3.bar(xs + (j - 1.5) * .2, [v[j] for v in AB.values()], .2, color=cl[j], ec="k", lw=.4, label=labs[j])
a3.set_xticks(xs, list(AB)); a3.set_ylim(30, 90); a3.set_ylabel("test mean Spearman ($p_0$)")
a3.set_title("(b) what the decoder uses", fontsize=10); a3.legend(fontsize=7.5, frameon=False, loc="upper left")

cols = {"hs2": "#7fb4e6", "hs2r": "#7fb4e6", "hanna": "#b39ddb", "ell": "#f2b36b", "merlin": "#7cc9a6"}
names = {"hs2": "HS2", "hanna": "HANNA", "ell": "ELLIPSE", "merlin": "MERLIN"}
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
                a2.scatter(100 * dr, 100 * do, s=26 if verb else 12, c="#d0473f" if verb else c, marker="^" if verb else "o",
                           ec="k", lw=.3, alpha=.85, label=lab, zorder=3 if verb else 2)
lim = [-42, 45]
a2.plot(lim, lim, "k--", lw=.7); a2.axhline(0, color="k", lw=.5); a2.axvline(0, color="k", lw=.5)
a2.set_xlim(lim); a2.set_ylim(-32, 20); a2.set_xlabel(r"$\Delta$ Spearman on sel, raw scores")
a2.set_ylabel(r"$\Delta$ Spearman, decoded"); a2.set_title("(c) every candidate edit, per attribute", fontsize=10)
a2.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("paper/figures/axis.pdf"); fig.savefig("paper/figures/axis.png", dpi=150)

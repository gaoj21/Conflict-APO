"""Method figure for ShAD-APO (draft): APO and the decoder solve one problem by variable projection.
(a) post-hoc: APO fits the prompt to the raw labels, which carry each attribute's length dependence; the decoder is
    added afterwards, so APO optimizes a different objective than the system that is deployed.
(b) ShAD-APO: min_{p,W} ||y - W [s(p); z]||^2. The length part of the decoder does not depend on the prompt and is
    projected out of the labels in closed form (content labels y~ = y - b z); any APO method optimizes the prompt on
    y~; ShAD is fitted in closed form and puts each attribute's length dependence back.
(c) HS2 / ELLIPSE / HANNA test (Gemma 4 26B): APO on raw vs content labels, both + ShAD.
  python scripts/fig_shadapo.py  -> paper/figures/method_shadapo.{pdf,png}
"""
import os, sys

import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREY, DARK, RED, GREEN, BLUE, YEL, MUTED = "#b9b9b9", "#2b3a4a", "#d0473f", "#2f9e5b", "#3b7dd8", "#e0b000", "#666666"

# HS2 verbosity: human labels and content labels vs standardized log length (fit + sel)
gold = pd.read_csv(f"{ROOT}/results/claude_hs2/gold.csv").set_index("id")
cl = pd.read_csv(f"{ROOT}/results/gemma/hs2/content_labels.csv").set_index("id")
z = gold.loc[cl.index, "z"].to_numpy(float); z = (z - z.mean()) / z.std()
yv, cv = gold.loc[cl.index, "verbosity"].to_numpy(float), cl["verbosity"].to_numpy(float)
rng = np.random.default_rng(0)

# test Spearman x100 + ShAD: (raw labels, content labels)
RES = [("MIPROv2 · HS2", 42.6, 44.2), ("MIPROv2 · ELLIPSE", 62.6, 67.8), ("MIPROv2 · HANNA", 51.7, 52.3),
       ("TextGrad · HS2", 40.4, 41.4), ("TextGrad · ELLIPSE", 58.9, 60.7), ("TextGrad · HANNA", 44.4, 47.0)]

fig = plt.figure(figsize=(17, 7.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(0, 72); ax.axis("off")


def box(x, y, w, h, txt="", fc="white", ec=DARK, lw=1.5, fs=11, c=DARK, r=1.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=c, weight=weight)


def arrow(x0, y0, x1, y1, c=DARK, lw=1.5, rad=0.0, ms=13, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def doc(x, y, w=4.6, h=5.6):
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - 1.3), (x + w - 1.3, y + h), (x, y + h)], closed=True,
                         fc="#fff8dc", ec=DARK, lw=1.2))
    for i in range(3):
        ax.plot([x + .7, x + w - .9], [y + 1.1 + i * 1.2] * 2, color="#8a8a8a", lw=.8)


def scatter(rect, yy, title, col, slope_note):
    a = fig.add_axes(rect)
    a.scatter(z, yy + rng.uniform(-.18, .18, len(yy)), s=3, alpha=.25, color=col, lw=0)
    b = np.polyfit(z, yy, 1); xs = np.linspace(-2.5, 2.5, 10)
    a.plot(xs, np.polyval(b, xs), color=DARK, lw=1.6)
    a.set_xlim(-2.6, 2.6); a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("length z", fontsize=8.5, labelpad=1); a.set_title(title, fontsize=9.5, pad=2)
    a.text(.03, .93, slope_note, transform=a.transAxes, fontsize=8.5, va="top", color=DARK)
    for s in a.spines.values():
        s.set_color("#999999")


# ------------------------------------------------------------------ (a) post-hoc
ax.text(1, 69.5, "(a) APO, then ShAD (post-hoc)", fontsize=14, weight="bold", color=DARK)
box(1, 7, 44, 59, fc="#fafafa", ec="#cccccc", lw=1.0)
ax.text(23, 62.5, "labels y  (carry each attribute's length)", ha="center", fontsize=10.5, color=DARK)
scatter([0.025, .60, .11, .19], yv, "human verbosity", RED, "ρ(y, z) = .65")
box(27, 47, 16, 9, "APO\n(any optimizer)", fc="#eef3f8", ec=BLUE, fs=10.5)
arrow(23.5, 51.5, 27, 51.5)
box(3, 29, 13, 8, "prompt p", fc="#eef3f8", ec=BLUE, fs=11)
arrow(33, 47, 11, 37.2, rad=-.12)
ax.text(30.5, 40.5, "fits y: edits\nmove along length", fontsize=9.5, color=RED, ha="left")
doc(18.5, 30)
box(26.5, 29, 15.5, 8, "LLM judge\n1 call → s", fs=10.5)
arrow(16, 33, 18.3, 33); arrow(23.3, 33, 26.5, 33)
box(14, 14, 18, 9, "ShAD\n(added afterwards)", fc="#f3faf5", ec=GREEN, fs=10.5)
arrow(34, 29, 27, 23.2, rad=.1)
arrow(23, 14, 23, 11)
ax.text(23, 8.6, "ŷ", ha="center", fontsize=13, color=DARK)
ax.text(23, 3.3, "APO optimizes ‖y − s(p)‖, not the deployed judge:\nlength moves create spurious conflicts",
        ha="center", fontsize=9.8, color=RED)

# ------------------------------------------------------------------ (b) ShAD-APO
X0 = 49
ax.text(X0, 69.5, "(b) ShAD-APO: one problem, solved by variable projection", fontsize=14, weight="bold", color=DARK)
box(X0, 7, 76, 59, fc="#fcfdfc", ec="#cccccc", lw=1.0)
ax.text(X0 + 38, 62.3, r"$\min_{p,\;W}\ \Vert\, y - W\,[\,s(p);\ z\,]\,\Vert^2$", ha="center", fontsize=15, color=DARK)
ax.text(X0 + 38, 58.3, "W: closed form (ridge)      p: APO search", ha="center", fontsize=10, color=MUTED)

# step 1: project the length part out of the labels
ax.text(X0 + 2, 53.8, "①  project out what W does without p", fontsize=11, weight="bold", color=DARK)
scatter([.305, .515, .085, .17], yv, "human  y", RED, "ρ = .65")
arrow(X0 + 25.5, 46.5, X0 + 31, 46.5, lw=2)
ax.text(X0 + 28.2, 48.6, r"$-\,b_k z$", ha="center", fontsize=11, color=DARK)
scatter([.49, .515, .085, .17], cv, "content  ỹ", GREEN, "ρ = 0")
ax.text(X0 + 52, 51.5, r"$\tilde y_k = y_k - b_k\,(z-\bar z)$", fontsize=12, color=DARK)
ax.text(X0 + 52, 47.6, "no judge, no extra call;\nby FWL, the prompt's gain\nafter decoding depends on ỹ",
        fontsize=9.3, color=MUTED, va="top")

# step 2: APO on content labels
ax.text(X0 + 2, 33.8, "②  any APO, unchanged, on ỹ", fontsize=11, weight="bold", color=DARK)
box(X0 + 2, 20, 13, 8, "prompt p", fc="#eef3f8", ec=BLUE, fs=11)
doc(X0 + 17.5, 21)
box(X0 + 25, 20, 15, 8, "LLM judge\n1 call → s", fs=10.5)
arrow(X0 + 15, 24, X0 + 17.3, 24); arrow(X0 + 22.3, 24, X0 + 25, 24)
arrow(X0 + 32, 19.8, X0 + 9, 19.8, c=BLUE, rad=-.45, lw=1.4)
ax.text(X0 + 20.5, 10.5, "APO fits ỹ: length moves earn nothing", ha="center", fontsize=9.5, color=BLUE)

# step 3: ShAD puts length back
ax.text(X0 + 46, 33.8, "③  ShAD (closed form)", fontsize=11, weight="bold", color=DARK)
box(X0 + 46, 17, 23, 12, "", fc="#f3faf5", ec=GREEN, lw=1.6)
ax.text(X0 + 57.5, 25.5, r"$\hat y_k = w_k^\top [\,s;\ z\,]$", ha="center", fontsize=13, color=DARK)
ax.text(X0 + 57.5, 20.3, "anchored ridge per attribute:\nlength back, cross-attribute info", ha="center", fontsize=9,
        color=MUTED)
arrow(X0 + 40, 24, X0 + 46, 24, lw=1.8)
box(X0 + 53.5, 9, 8, 5, "z", fc="#fff3c4", ec=YEL, fs=12)
arrow(X0 + 57.5, 14, X0 + 57.5, 17, c=YEL, lw=1.6)
ax.text(X0 + 62.5, 11.5, "length\n(from the text)", ha="left", va="center", fontsize=9, color=MUTED)
arrow(X0 + 69, 23, X0 + 72, 23)
ax.text(X0 + 73.6, 23, "ŷ", ha="center", va="center", fontsize=13, color=DARK)
ax.text(X0 + 38, 3.3, "APO and the decoder optimize the same objective; the optimizer, the judge call and the cost are unchanged",
        ha="center", fontsize=9.8, color=GREEN)

# ------------------------------------------------------------------ (c) result
ax.text(129, 69.5, "(c) Same APO, same ShAD: only the target differs", fontsize=14, weight="bold", color=DARK)
a = fig.add_axes([.825, .2, .165, .72])
ys = np.arange(len(RES))[::-1]
for y, (name, r0, r1) in zip(ys, RES):
    a.plot([r0, r1], [y, y], color=GREY, lw=2.2, zorder=1)
    a.scatter([r0], [y], s=55, color=GREY, zorder=2, label="APO on raw labels + ShAD" if y == ys[0] else None)
    a.scatter([r1], [y], s=55, color=GREEN, zorder=3, label="ShAD-APO (content labels)" if y == ys[0] else None)
    a.text(r1 + 1.0, y, f"+{r1 - r0:.1f}", va="center", fontsize=9.5, color=GREEN)
a.set_yticks(ys); a.set_yticklabels([r[0] for r in RES], fontsize=9.5)
a.set_xlabel("test Spearman ×100", fontsize=10); a.set_xlim(38, 72)
a.legend(loc="upper center", bbox_to_anchor=(0.4, -0.1), fontsize=9, frameon=False)
for s in ("top", "right"):
    a.spines[s].set_visible(False)

out = f"{ROOT}/paper/figures/method_shadapo"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=160, bbox_inches="tight")
print("wrote", out + ".{pdf,png}")

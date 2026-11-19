"""Method figure: prompt = encoder, per-attribute readout = decoder.
(a) the conflict: with one shared prompt and the raw scores as output (identity decoder), an edit that makes the judge
    reward length raises the length-driven attribute and lowers the others; raw selection accepts it (HS2, step 1).
(b) the method: the optimizer is unchanged; one judge call gives all attribute scores; a per-attribute readout over
    those scores + standardized log length z, shrunk toward the attribute's own score (lambda by CV; raw score kept
    as an option) gives the final score.
(c) the effect (HS2 test): the readout recovers the length axis on v0 for free, so optimization is left with content.
  python scripts/fig_method.py  -> figures/method.{pdf,png}
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

GREY, DARK, RED, GREEN, MUTED, BLUE = "#b9b9b9", "#2b3a4a", "#d0473f", "#2f9e5b", "#555555", "#3b7dd8"
ATT = ["help", "corr", "coh", "cplx", "verb"]
D_S1 = [.013, -.045, .032, .094, .337]  # HS2 run 1, step 1 (accepted): raw sel Spearman change per attribute
# HS2 test mean Spearman (run 1 / run 2)
TEST = [("$p_0$\nraw", .337, .337), ("$p_0$\n+ShAD", .452, .442), ("$p_T$\nraw", .455, .418),
        ("$p_T$\n+ShAD", .483, .453)]

fig = plt.figure(figsize=(16, 7.8))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(-12, 66); ax.axis("off")


def box(x, y, w, h, fc="white", ec=DARK, lw=1.6, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, c=DARK, lw=1.6, rad=0.0, ms=14, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def doc(x, y, w=5.2, h=6.4, fc="#fff8dc", ec=DARK):
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - 1.5), (x + w - 1.5, y + h), (x, y + h)], closed=True,
                         fc=fc, ec=ec, lw=1.3))
    for i in range(4):
        ax.plot([x + .8, x + w - 1], [y + 1.1 + i * 1.15] * 2, color="#8a8a8a", lw=.9)


# ------------------------------------------------------------------ (a) conflict
box(1.5, 2, 46, 62, fc="#f6f8fb", ec="#c7d0da", lw=1, r=2)
ax.text(3.5, 61, "(a) Where conflicts come from", fontsize=14, weight="bold", color=DARK, va="center")
ax.text(3.5, 57.6, "one shared prompt, raw scores as output\n(identity decoder)", fontsize=10.5, color=MUTED, va="top")
doc(5, 40); ax.text(7.6, 39.2, "$p_0$", ha="center", va="top", fontsize=11, color=DARK)
arrow(11, 43.2, 16, 43.2)
ax.text(13.5, 45, "edit", ha="center", fontsize=9.5, color=MUTED)
doc(16.5, 40); ax.text(19.1, 39.2, "$p_1$", ha="center", va="top", fontsize=11, color=DARK)
# per-attribute change bars (real)
bx, by, sc = 9, 22, 30
ax.plot([bx - 1, bx + 34], [by, by], color=DARK, lw=.8)
for i, (a_, d) in enumerate(zip(ATT, D_S1)):
    x = bx + i * 6.8
    ax.add_patch(Rectangle((x, by), 4, d * sc, fc=RED if i == 4 else (GREEN if d > 0 else GREY), ec=DARK, lw=.6))
    ax.text(x + 2, by + d * sc + (.5 if d > 0 else -.5), f"{100 * d:+.1f}", ha="center", va="bottom" if d > 0 else "top",
            fontsize=8.5, color=DARK)
    ax.text(x + 2, by - 3.4, a_, ha="center", va="top", fontsize=9, color=DARK)
ax.text(24.5, 18.3 - 1.5, "", fontsize=1)
ax.text(24.5, 15.2, "$\\Delta$ Spearman of $p_1$ vs $p_0$ (HS2, sel)", ha="center", fontsize=9.5, color=MUTED, va="top")
ax.text(24.5, 11.0, "the biggest accepted edit is a move along\nthe length axis: verbosity up, correctness down",
        ha="center", fontsize=10.3, color=RED, va="top")
arrow(47.5, 33, 52, 33, lw=2.2, ms=18)

# ------------------------------------------------------------------ (b) method
box(52, 2, 60, 62, fc="white", ec="#c7d0da", lw=1, r=2)
ax.text(54, 61, "(b) Encoder + per-attribute decoder", fontsize=14, weight="bold", color=DARK, va="center")
ax.text(54, 57.6, "the prompt optimizer is unchanged; the output layer is learned", fontsize=10.5, color=MUTED, va="top")
# encoder
box(55, 36, 13, 14, fc="#eef3f8")
doc(58.9, 41.5)
ax.text(61.5, 38.2, "prompt $p$", ha="center", fontsize=10, color=DARK)
ax.text(61.5, 51.5, "encoder", ha="center", fontsize=10.5, weight="bold", color=BLUE)
ax.text(61.5, 33.5, "optimized by\nTextGrad / CCC", ha="center", va="top", fontsize=9, color=MUTED)
arrow(68, 43, 72, 43)
box(72, 38.5, 9, 9, fc="#f3f3f3")
ax.text(76.5, 43, "LLM\njudge\n1 call", ha="center", va="center", fontsize=9.5, color=DARK)
arrow(81, 43, 84.5, 43)
# scores + z
for i, a_ in enumerate(ATT):
    yy = 50 - i * 3.2
    box(84.5, yy - 1.2, 7.5, 2.6, fc="white", ec="#9aa5b1", lw=.8, r=.4)
    ax.text(88.25, yy + .1, f"$s_{{\\mathrm{{{a_}}}}}$", ha="center", va="center", fontsize=9.5, color=DARK)
box(84.5, 50 - 5 * 3.2 - 1.2, 7.5, 2.6, fc="#fff3c4", ec="#9aa5b1", lw=.8, r=.4)
ax.text(88.25, 50 - 5 * 3.2 + .1, "$z$ (length)", ha="center", va="center", fontsize=9, color=DARK)
arrow(92.5, 42, 96, 42)
# decoder
box(96, 34, 14, 17, fc="#f3faf5", ec=GREEN)
ax.text(103, 52.3, "decoder", ha="center", fontsize=10.5, weight="bold", color=GREEN)
ax.text(103, 42.5, "per attribute $k$:\n$\\hat y_k = w_k^\\top [s, z]$\nridge shrunk\ntoward $s_k$", ha="center",
        va="center", fontsize=9.5, color=DARK)
arrow(103, 34, 103, 30)
ax.text(103, 29.5, "$\\hat y_1 \\ldots \\hat y_K$", ha="center", va="top", fontsize=11, color=DARK)
ax.text(82, 22.5, "$\\lambda$ (and the raw score $s_k$ itself as an option)\nchosen by within-fold CV Spearman on fit",
        ha="center", fontsize=10, color=MUTED, va="top")
ax.text(82, 14.5, "length is absorbed by the decoder:\nafter decoding, a pure length move is worth nothing",
        ha="center", fontsize=11, color=GREEN, va="top", weight="bold")
ax.text(82, 6.6, "1 judge call per item;  smooth shrinkage, no gates;\nfalls back to the raw score when CV finds no gain",
        ha="center", fontsize=9.5, color=MUTED, va="top")
arrow(112.5, 33, 117, 33, lw=2.2, ms=18)

# ------------------------------------------------------------------ (c) effect
box(117, 2, 41.5, 62, fc="white", ec="#c7d0da", lw=1, r=2)
ax.text(119, 61, "(c) What is left for the prompt", fontsize=14, weight="bold", color=DARK, va="center")
ax.text(119, 57.6, "HS2 test, mean Spearman $\\times$100 (two runs)", fontsize=10.5, color=MUTED, va="top")
bx, by, sc = 122, 18, 60
ax.plot([bx - 1, bx + 34], [by, by], color=DARK, lw=.8)
for i, (lab, r1, r2) in enumerate(TEST):
    x = bx + i * 8.5
    col = [GREY, GREEN, GREY, GREEN][i]
    ax.add_patch(Rectangle((x, by), 3, (r1 - .25) * sc, fc=col, ec=DARK, lw=.6))
    ax.add_patch(Rectangle((x + 3.2, by), 3, (r2 - .25) * sc, fc=col, ec=DARK, lw=.6, alpha=.6))
    ax.text(x + 3.1, by + (max(r1, r2) - .25) * sc + .6, f"{100 * r1:.1f}\n{100 * r2:.1f}", ha="center", va="bottom",
            fontsize=8.3, color=DARK)
    ax.text(x + 3.1, by - .8, lab, ha="center", va="top", fontsize=9, color=DARK)
ax.text(bx - 1.5, by, "25", ha="right", va="center", fontsize=8, color=MUTED)
ax.text(137.75, 9.5, "the decoder alone gives $p_0$ the length gain\n(+11) that raw optimization spends its\n"
        "first step on; optimization then adds content", ha="center", fontsize=9.8, color=DARK, va="top")

# ------------------------------------------------------------------ data splits
segs = [("fit", "the optimizer sees its errors; the decoder is fitted", "#e8eef6", 62),
        ("sel", "accept or reject edits (optimizer's own rule, raw scores)", "#fff3c4", 62),
        ("test", "final evaluation only", "#e9f6ee", 31)]
x = 1.5
for name, use, fc, w in segs:
    ax.add_patch(Rectangle((x, -10.5), w, 7, fc=fc, ec=DARK, lw=1))
    ax.text(x + w / 2, -5.4, name, ha="center", va="center", fontsize=12, weight="bold", color=DARK)
    ax.text(x + w / 2, -8.6, use, ha="center", va="center", fontsize=9.8, color=MUTED)
    x += w + .5

fig.savefig("figures/method.pdf", bbox_inches="tight"); fig.savefig("figures/method.png", dpi=150, bbox_inches="tight")
print("saved figures/method.{pdf,png}")

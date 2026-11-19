"""Method figure: explore -> route -> merge.
(a) explore: an unchanged shared-prompt optimizer; every candidate, kept or rejected, goes to the archive (all are
    judged on fit/sel anyway). Conflicting candidates are specialists: each is best on some attribute.
(b) route: per attribute, the archive version whose readout (own score + length z) agrees best with humans by CV.
(c) merge: an LLM writes ONE prompt taking each attribute's criteria from its source; judge once + per-attribute readout.
  python scripts/fig_method.py  -> figures/method.{pdf,png}
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

C = ["#7cc9a6", "#f29aa8", "#7fb4e6"]  # attributes A, B, C
GREY, DARK, RED, GREEN, MUTED = "#b9b9b9", "#2b3a4a", "#d0473f", "#2f9e5b", "#555555"
# illustrative per-attribute agreement of each archived version (A, B, C) and whether the optimizer kept it
V = [("p0", [.55, .50, .45], True), ("c1", [.48, .66, .47], False), ("c2", [.58, .56, .52], True),
     ("c3", [.70, .45, .50], False), ("c4", [.52, .58, .64], True), ("c5", [.50, .54, .58], False)]
ROUTE = {0: 3, 1: 1, 2: 4}  # attribute -> archive index (c3 for A, c1 for B, c4 for C)

fig = plt.figure(figsize=(16, 7.8))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(-12, 66); ax.axis("off")


def box(x, y, w, h, fc="white", ec=DARK, lw=1.6, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, c=DARK, lw=1.6, rad=0.0, ms=14, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def doc(x, y, w=5.2, h=6.4, fc="#fff8dc", lines=None, ec=DARK):
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - 1.5), (x + w - 1.5, y + h), (x, y + h)], closed=True,
                         fc=fc, ec=ec, lw=1.3))
    for i in range(4):
        col = lines[i] if lines else "#8a8a8a"
        ax.plot([x + .8, x + w - 1], [y + 1.1 + i * 1.15] * 2, color=col, lw=2.2 if lines else .9)


def bars(x, y, q, scale=7, bw=1.3, gap=.4, base=None, hi=None, labels=False):
    for i, v in enumerate(q):
        ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, v * scale, fc=C[i], ec=DARK,
                               lw=2.0 if hi == i else .6))
        if base is not None:
            ax.plot([x + i * (bw + gap) - .2, x + i * (bw + gap) + bw + .2], [y + base[i] * scale] * 2, color=DARK,
                    lw=1, ls=(0, (2, 1)))
        if labels:
            ax.text(x + i * (bw + gap) + bw / 2, y - .7, "ABC"[i], ha="center", va="top", fontsize=8, color=DARK)


# ------------------------------------------------------------------ (a) explore
box(1.5, 2, 52, 62, fc="#f6f8fb", ec="#c7d0da", lw=1, r=2)
ax.text(3.5, 61, "(a) Explore", fontsize=14, weight="bold", color=DARK, va="center")
ax.text(3.5, 57.6, "unchanged shared-prompt optimizer (TextGrad / CCC) on\nfit + sel; every candidate is archived, kept or rejected",
        fontsize=10.5, color=MUTED, va="top")
pos = [(5, 40), (21, 40), (37, 40), (5, 22), (21, 22), (37, 22)]
for (name, q, kept), (x, y) in zip(V, pos):
    doc(x, y)
    bars(x + 6.3, y, q, base=V[0][1], labels=True)
    ax.text(x + 2.6, y - .8, f"${name[0]}_{name[1]}$", ha="center", va="top", fontsize=11, color=DARK, style="italic")
    if name != "p0":
        ax.text(x + 4.6, y + 7.4, "kept" if kept else "rejected", fontsize=8.5, color=GREEN if kept else RED,
                ha="center", weight="bold")
ax.text(27.5, 14.6, "rejected candidates are specialists:\n$c_3$ is best on A, $c_1$ on B", ha="center", fontsize=10.5,
        color=RED, va="top")
ax.text(27.5, 6.4, "bars: per-attribute agreement with humans;\ndashed = $p_0$", ha="center", fontsize=9.5, color=MUTED)
arrow(53.5, 33, 58, 33, lw=2.2, ms=18)

# ------------------------------------------------------------------ (b) route
box(58, 2, 44, 62, fc="white", ec="#c7d0da", lw=1, r=2)
ax.text(60, 61, "(b) Assign each attribute", fontsize=14, weight="bold", color=DARK, va="center")
ax.text(60, 57.6, "on sel, per attribute: each version's readout\n(own score + length $z$) vs. the final $p_T$", fontsize=10.5, color=MUTED, va="top")
x0, y0, cw, rh = 71, 18, 8.5, 4.6
for j in range(3):
    ax.add_patch(Rectangle((x0 + j * cw, y0 + len(V) * rh + .4), cw - .4, 3, fc=C[j], ec=DARK, lw=.6))
    ax.text(x0 + j * cw + cw / 2 - .2, y0 + len(V) * rh + 1.9, "ABC"[j], ha="center", va="center", fontsize=10.5,
            weight="bold", color=DARK)
for i, (name, q, kept) in enumerate(V):
    yy = y0 + (len(V) - 1 - i) * rh
    ax.text(x0 - 1.5, yy + rh / 2, f"${name[0]}_{name[1]}$", ha="right", va="center", fontsize=11, color=DARK)
    for j in range(3):
        best = ROUTE[j] == i
        ax.add_patch(Rectangle((x0 + j * cw, yy + .2), cw - .4, rh - .4, fc="#fff3c4" if best else "white",
                               ec=DARK if best else "#d5d9de", lw=2 if best else .8))
        ax.text(x0 + j * cw + cw / 2 - .2, yy + rh / 2, f".{int(round(q[j] * 100)):02d}", ha="center", va="center",
                fontsize=10.5, color=DARK, weight="bold" if best else "normal")
ax.text(80, 14.5, "advantage shrunk by empirical Bayes ($\\tau^2$ from the data);\nan extra source must pay one call; else stay with $p_T$",
        ha="center", fontsize=10, color=MUTED, va="top")
ax.text(80, 6.4, "A $\\leftarrow c_3$     B $\\leftarrow c_1$     C $\\leftarrow c_4$", ha="center", fontsize=12.5,
        color=DARK, weight="bold")
arrow(102.5, 33, 107, 33, lw=2.2, ms=18)

# ------------------------------------------------------------------ (c) merge
box(107, 2, 51.5, 62, fc="white", ec="#c7d0da", lw=1, r=2)
ax.text(109, 61, "(c) Merge into one prompt", fontsize=14, weight="bold", color=DARK, va="center")
ax.text(109, 57.6, "an LLM writes one prompt that carries each\nattribute's criteria over from its source",
        fontsize=10.5, color=MUTED, va="top")
srcs = [("c3", 0), ("c1", 1), ("c4", 2)]
for i, (n, j) in enumerate(srcs):
    doc(110, 42 - i * 9, lines=["#8a8a8a", C[j], C[j], "#8a8a8a"])
    ax.text(116.4, 45.2 - i * 9, f"${n[0]}_{n[1]}$: {'ABC'[j]}", fontsize=10.5, va="center", color=DARK)
    arrow(123, 45.2 - i * 9, 128, 36.5, lw=1.1, ms=10)
box(128, 31.5, 9, 10, fc="#eef3f8")
ax.text(132.5, 36.5, "merger\nLLM", ha="center", va="center", fontsize=10.5, color=DARK)
arrow(137, 36.5, 140.5, 36.5, lw=1.4)
doc(141, 32, w=7, h=9, lines=[C[2], C[1], C[0], "#8a8a8a"], fc="#fffbe9")
ax.text(144.5, 30.8, "$p^\\star$ (1 prompt)", ha="center", va="top", fontsize=10.5, color=DARK)
# deployment row
arrow(144.5, 27, 144.5, 23.5, lw=1.4)
box(111, 15, 44, 8, fc="#f3faf5", ec=GREEN, lw=1.6)
ax.text(143, 25.3, "deploy $p^\\star$ or $p_T$,\nwhichever is better on sel", ha="right", va="center", fontsize=9.5, color=MUTED)
ax.text(133, 19, "judge $p^\\star$ once $\\rightarrow$ per-attribute readout\nover its scores + length $z$",
        ha="center", va="center", fontsize=10.8, color=DARK)
arrow(133, 15, 133, 12.5, lw=1.4)
ax.text(133, 10.8, "1 call per item;  every attribute $\\geq p_0$", ha="center", va="top", fontsize=11.5,
        color=GREEN, weight="bold")
ax.text(133, 6.2, "length moves are absorbed by the readout;\nno gates, no change to the optimizer", ha="center",
        va="top", fontsize=9.8, color=MUTED)

# ------------------------------------------------------------------ data splits
segs = [("fit", "the optimizer sees its errors", "#e8eef6", 52),
        ("sel", "accept edits; assign attributes; keep $p_T$ or deploy $p^\\star$ (items never shown to the optimizer)", "#fff3c4", 72),
        ("test", "final evaluation only", "#e9f6ee", 31)]
x = 1.5
for name, use, fc, w in segs:
    ax.add_patch(Rectangle((x, -10.5), w, 7, fc=fc, ec=DARK, lw=1))
    ax.text(x + w / 2, -5.4, name, ha="center", va="center", fontsize=12, weight="bold", color=DARK)
    ax.text(x + w / 2, -8.6, use, ha="center", va="center", fontsize=9.8, color=MUTED)
    x += w + .5

fig.savefig("figures/method.pdf", bbox_inches="tight"); fig.savefig("figures/method.png", dpi=150, bbox_inches="tight")
print("saved figures/method.{pdf,png}")

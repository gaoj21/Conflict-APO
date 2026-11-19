"""Method figure: trajectory decoding. Top: an unchanged shared-prompt optimizer (TextGrad / CCC) leaves a trajectory of
prompt versions whose per-attribute quality trades off. Bottom left: the standard judge keeps the last version and
reads its scores directly (identity decoder) -> the sacrificed attribute stays lost. Bottom right: ours keeps a few
versions and fits one anchored readout per attribute over all their scores + the spurious feature (length).
  python scripts/fig_method.py  -> figures/method.{pdf,png}
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

C = ["#7cc9a6", "#f29aa8", "#7fb4e6"]  # attribute colors (A, B, C)
GREY, DARK, RED, GREEN = "#b9b9b9", "#2b3a4a", "#d0473f", "#2f9e5b"
# illustrative per-attribute quality of each prompt version (A, B, C)
Q = {"p0": [.62, .50, .45], "p1": [.55, .62, .50], "p2": [.66, .55, .60], "p3": [.52, .70, .63]}

fig = plt.figure(figsize=(15, 7.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 150); ax.set_ylim(0, 76); ax.axis("off")


def box(x, y, w, h, fc="white", ec=DARK, lw=1.6, r=1.2, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, **kw))


def arrow(x0, y0, x1, y1, c=DARK, lw=1.6, ls="-", rad=0.0, ms=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def doc(x, y, label, w=6.5, h=8, fc="#fff8dc"):
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - 1.8), (x + w - 1.8, y + h), (x, y + h)], closed=True,
                         fc=fc, ec=DARK, lw=1.4))
    for i in range(4):
        ax.plot([x + 1, x + w - 1.2], [y + 1.4 + i * 1.4] * 2, color="#8a8a8a", lw=0.9)
    ax.text(x + w / 2, y - 1.6, label, ha="center", va="top", fontsize=12, color=DARK, style="italic")


def bars(x, y, q, scale=9, bw=1.6, gap=0.5, base=None, labels=False):
    for i, v in enumerate(q):
        ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, v * scale, fc=C[i], ec=DARK, lw=0.6))
        if base is not None:
            ax.plot([x + i * (bw + gap) - .2, x + i * (bw + gap) + bw + .2], [y + base[i] * scale] * 2, color=DARK, lw=1.1, ls=(0, (2, 1)))
        if labels:
            ax.text(x + i * (bw + gap) + bw / 2, y - .9, "ABC"[i], ha="center", va="top", fontsize=8.5, color=DARK)


# ------------------------------------------------------------------ top: unchanged optimizer trajectory
box(2, 49, 146, 25, fc="#f6f8fb", ec="#c7d0da", lw=1.0, r=2)
ax.text(4, 71.6, "(a) Shared-prompt optimization, unchanged (TextGrad / CCC): one prompt judges all K attributes",
        fontsize=13, weight="bold", color=DARK, va="center")
xs = [14, 46, 78, 110]
for i, (k, x) in enumerate(zip(Q, xs)):
    doc(x, 56, f"${k[0]}_{k[1]}$")
    bars(x + 9, 55.5, Q[k], labels=True, base=Q["p0"])
    if i < 3:
        arrow(x + 17, 60, xs[i + 1] - 1.5, 60)
        ax.text((x + 17 + xs[i + 1]) / 2 - .7, 61.2, "edit", ha="center", fontsize=10.5, color="#555")
ax.text(127, 63.5, "each edit is kept or\nrejected for all\nattributes at once:\nA is traded for B, C", fontsize=10.5,
        color=RED, va="center")
ax.text(14, 52, "bars: per-attribute agreement with humans (dashed = $p_0$)", fontsize=10, color="#555")

# ------------------------------------------------------------------ bottom left: standard read-out
box(2, 2, 52, 44, fc="white", ec="#c7d0da", lw=1.0, r=2)
ax.text(4, 43.3, "(b) Standard: keep $p_T$, read its scores", fontsize=13, weight="bold", color=DARK, va="center")
doc(9, 24, "$p_3$")
arrow(17, 28, 24, 28)
box(24, 23.5, 9, 9, fc="#eef3f8")
ax.text(28.5, 28, "judge\nLLM", ha="center", va="center", fontsize=10.5, color=DARK)
arrow(33, 28, 39, 28)
ax.text(46, 38.5, "identity decoder\n$\\hat y_k = s_k$", ha="center", fontsize=11, color=DARK)
bars(40, 24, Q["p3"], base=Q["p0"], labels=True)
ax.annotate("", xy=(41, 24 + Q["p3"][0] * 9 - .2), xytext=(41, 24 + Q["p0"][0] * 9 + 3),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
ax.text(28, 12.5, "the attribute that was traded away\nstays lost (conflict), and length\nmoves count as gains",
        ha="center", fontsize=11.5, color=RED)

# ------------------------------------------------------------------ bottom right: ours
box(57, 2, 91, 44, fc="white", ec="#c7d0da", lw=1.0, r=2)
ax.text(59, 43.3, "(c) Ours: keep the trajectory, decode each attribute", fontsize=13, weight="bold", color=DARK, va="center")
keep = ["p0", "p2", "p3"]
ys = [33, 22.5, 12]
for k, y in zip(keep, ys):
    doc(61, y, f"${k[0]}_{k[1]}$", w=5.5, h=7)
    arrow(67.5, y + 3.5, 72, 28 if y == 22.5 else (30 if y > 22.5 else 26), lw=1.2, ms=10)
box(72, 23.5, 9, 9, fc="#eef3f8")
ax.text(76.5, 28, "judge\nLLM", ha="center", va="center", fontsize=10.5, color=DARK)
# score vectors -> a bus feeding every decoder
for j, (k, y) in enumerate(zip(keep, [36, 30.5, 25])):
    arrow(81, 28, 85, y + 1, lw=1.0, ms=9)
    for i in range(3):
        ax.add_patch(Rectangle((85.5 + i * 2.2, y), 2, 2, fc=C[i], ec=DARK, lw=.6))
    ax.text(92.6, y + 1, f"$s^{{({k[1]})}}$", fontsize=11, va="center", color=DARK)
    ax.plot([96, 99], [y + 1, y + 1], color=DARK, lw=1.0)
ax.add_patch(Rectangle((85.5, 18), 2, 2, fc=GREY, ec=DARK, lw=.6))
ax.text(88.3, 19, "$z$ = length", fontsize=10.5, va="center", color="#555")
ax.plot([96, 99], [19, 19], color=DARK, lw=1.0)
ax.plot([99, 99], [19, 37], color=DARK, lw=1.4)
for i, y in enumerate([33, 25.5, 18]):
    box(101.5, y, 26, 6, fc="white", ec=C[i], lw=2.2)
    a_ = "ABC"[i]
    ax.text(114.5, y + 3, f"$\\hat y_{{{a_}}} = w_{{{a_}}}^\\top[s^{{(0)}}, s^{{(2)}}, s^{{(3)}}, z]$",
            ha="center", va="center", fontsize=10.5, color=DARK)
    arrow(99, y + 3, 101.5, y + 3, lw=1.0, ms=8)
    arrow(127.5, y + 3, 131, 29, lw=1.0, ms=8)
ax.text(114.5, 15.6, "per attribute: ridge shrunk toward one version's\nown score; anchor and strength chosen by CV",
        ha="center", va="top", fontsize=10, color="#555")
final = [Q["p2"][0] + .02, Q["p3"][1], Q["p3"][2] + .01]
bars(132, 24, final, base=Q["p0"], labels=True)
ax.text(138.5, 38.5, "every attribute\n$\\geq p_0$", ha="center", fontsize=11.5, color=GREEN, weight="bold")
ax.text(102, 6.2, "conflicting versions carry complementary information: A is read from $p_2$, B and C from $p_3$;\n"
        "length is an input, so moves along it earn nothing.  No change to the optimizer, no gates.",
        fontsize=11, color=GREEN, ha="center", va="center")

fig.savefig("figures/method.pdf", bbox_inches="tight"); fig.savefig("figures/method.png", dpi=150, bbox_inches="tight")
print("saved figures/method.{pdf,png}")

"""Editable PowerPoint version of the method figure (shared encoder, attribute-specific decoder).
Top: inference. Bottom: how it is trained -- the prompt optimizer is unchanged (it selects by raw Spearman), and the
readouts are fitted once on the final prompt's scores.
  python scripts/fig_method_pptx.py  -> paper/figures/method.pptx"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

NAVY = RGBColor(0x1f, 0x2d, 0x4d)
TEAL = RGBColor(0xd4, 0xf0, 0xea)
TEAL_E = RGBColor(0x3a, 0x8f, 0x86)
BLUE = RGBColor(0xd6, 0xe9, 0xfb)
BLUE_E = RGBColor(0x3b, 0x7d, 0xd8)
SALMON = RGBColor(0xf6, 0xa8, 0x9a)
SALMON_E = RGBColor(0xd9, 0x5a, 0x48)
GREY = RGBColor(0xe6, 0xe6, 0xe6)
MUTED = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xff, 0xff, 0xff)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
sl = prs.slides.add_slide(prs.slide_layouts[6])
S = sl.shapes


def text(tf_owner, parts, size=12, bold=False, color=NAVY, align=PP_ALIGN.CENTER, italic=False):
    """parts: str or list of (text, 'sub'|'sup'|None); a list of such lists gives several paragraphs."""
    tf = tf_owner.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.04)
    tf.margin_top = tf.margin_bottom = Inches(.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    paras = parts if isinstance(parts, list) and parts and isinstance(parts[0], list) else [parts]
    for i, pp in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for item in ([pp] if isinstance(pp, str) else pp):
            t, kind = (item, None) if isinstance(item, str) else item
            r = p.add_run()
            r.text = t
            f = r.font
            f.size, f.bold, f.italic, f.color.rgb, f.name = Pt(size), bold, italic, color, "Calibri"
            if kind:
                r._r.get_or_add_rPr().set("baseline", "-25000" if kind == "sub" else "30000")


def box(x, y, w, h, fill=WHITE, line=NAVY, lw=1.25, shape=MSO_SHAPE.ROUNDED_RECTANGLE, parts=None, **kw):
    s = S.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = .12
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb, s.line.width = line, Pt(lw)
    s.shadow.inherit = False
    if parts is not None:
        text(s, parts, **kw)
    return s


def label(x, y, w, h, parts, **kw):
    t = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    text(t, parts, **kw)
    return t


def arrow(x0, y0, x1, y1, color=NAVY, lw=2.0):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb, c.line.width = color, Pt(lw)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def cells(x, y, names, w=.5, h=.42, z_last=False, size=12):
    for i, n in enumerate(names):
        last = z_last and i == len(names) - 1
        box(x + i * w, y, w, h, fill=SALMON if last else TEAL, line=SALMON_E if last else TEAL_E, lw=1,
            shape=MSO_SHAPE.RECTANGLE, parts=n, size=size, bold=True, italic=True)


def doc(x, y, w, h, parts, size=11):
    s = box(x, y, w, h, shape=MSO_SHAPE.FOLDED_CORNER, fill=WHITE, line=NAVY, lw=1.1)
    text(s, parts, size=size, align=PP_ALIGN.LEFT)
    return s


ATT = [("H", "Helpfulness"), ("C", "Correctness"), ("Coh", "Coherence"), ("V", "Verbosity")]
s_names = [[("s", None), (a, "sub")] for a, _ in ATT]

# ------------------------------------------------------------------ title
label(.3, .1, 9, .55, "Shared Encoder, Attribute-Specific Decoder", size=26, bold=True, align=PP_ALIGN.LEFT)
label(.3, .62, 4, .4, "INFERENCE", size=18, bold=True, align=PP_ALIGN.LEFT)

# ------------------------------------------------------------------ inference row
label(.45, 1.0, 2.8, .35, "INPUT", size=15, bold=True)
d = doc(.45, 1.4, 2.8, 1.55, [[("Why do we have seasons?", None)],
                              [("Earth's tilted axis changes sunlight angles and day lengths. In other words, "
                                "seasons happen because Earth is tilted.", None)]], size=11)
d.text_frame.paragraphs[0].runs[0].font.bold = True
box(.75, 3.15, 1.5, .36, fill=SALMON, line=SALMON_E, parts=[("log length ", None), ("z", None)], size=12, bold=True)
arrow(3.35, 2.15, 3.85, 2.15)

label(3.9, 1.0, 2.6, .35, "ENCODER", size=15, bold=True)
box(3.95, 1.45, 2.3, .8, fill=BLUE, line=BLUE_E, parts=[["Shared prompt p +"], ["LLM judge"]], size=14, bold=True)
label(3.95, 2.28, 2.3, .3, "one call per item", size=11, color=MUTED)
cells(4.1, 2.62, s_names, size=12)
label(3.6, 3.04, 3.0, .26, "optimized by a TextGrad-style optimizer", size=10, color=MUTED, italic=True)
arrow(6.35, 2.15, 6.85, 2.15)

label(6.9, 1.0, 3.6, .35, "DECODER", size=15, bold=True)
label(6.75, 1.4, 2.4, .3, "one linear readout per attribute", size=10.5, color=MUTED)
cells(6.95, 1.95, s_names + ["z"], w=.46, z_last=True, size=12)
# z arrow from input
c1 = arrow(2.25, 3.33, 9.03, 3.33, color=SALMON_E, lw=1.75)
arrow(9.03, 3.33, 9.03, 2.4, color=SALMON_E, lw=1.75)
# readout rows
for i, (a, full) in enumerate(ATT):
    y = 1.45 + i * .5
    arrow(9.3, 2.16, 9.6, y + .19, lw=1.5)
    box(9.62, y, .55, .38, fill=WHITE, line=NAVY, lw=1, shape=MSO_SHAPE.RECTANGLE,
        parts=[("w", None), (a, "sub")], size=12, bold=True, italic=True)
    for j in range(5):
        last = j == 4
        box(10.17 + j * .22, y, .22, .38, fill=SALMON if last else TEAL, line=SALMON_E if last else TEAL_E, lw=.75,
            shape=MSO_SHAPE.RECTANGLE)
    arrow(11.35, y + .19, 11.65, y + .19, lw=1.5)
    box(11.7, y, 1.35, .38, fill=TEAL, line=TEAL_E, parts=full, size=12, bold=True)
label(11.7, 1.0, 1.35, .35, "OUTPUT", size=15, bold=True)
label(9.4, 3.45, 2.6, .4, [("ŷ", None), ("k", "sub"), (" = w", None), ("k", "sub"), ("ᵀ [s, z]", None)], size=15, italic=True)
label(6.6, 3.82, 6.6, .32, "ridge shrunk toward the own score; falls back to the raw score",
      size=10.5, color=MUTED)

# ------------------------------------------------------------------ divider
ln = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(.3), Inches(4.25), Inches(13.0), Inches(4.25))
ln.line.color.rgb, ln.line.width = RGBColor(0xbb, 0xbb, 0xbb), Pt(1)
label(.3, 4.3, 12, .45, [("TRAINING", None), ("  —  the optimizer is unchanged; the decoder is fitted once", None)],
      size=18, bold=True, align=PP_ALIGN.LEFT)

# ------------------------------------------------------------------ step 1: optimize prompt (unchanged)
label(.35, 4.8, 4.0, .4, "1  Optimize the prompt", size=15, bold=True, align=PP_ALIGN.LEFT)
doc(.5, 5.35, 1.45, .75, [[("p", None)], [("Score the response.", None)]], size=11)
label(.5, 6.12, 1.45, .3, "current prompt", size=10, color=MUTED)
arrow(2.0, 5.72, 2.45, 5.72)
label(1.85, 5.25, .75, .35, "rewrite", size=10, color=MUTED)
doc(2.5, 5.35, 1.45, .75, [[("p′", None)], [("Judge each attribute independently.", None)]], size=11)
label(2.5, 6.12, 1.45, .3, "candidate", size=10, color=MUTED)
label(.4, 6.4, 3.8, .55, [["error feedback from fit (unchanged optimizer);"], ["accept p′ if raw Spearman on sel improves"]], size=10.5)
ln = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(4.35), Inches(4.85), Inches(4.35), Inches(6.95))
ln.line.color.rgb = RGBColor(0xcc, 0xcc, 0xcc)

# ------------------------------------------------------------------ step 2: fit readouts once
label(4.5, 4.8, 4.6, .4, "2  Fit the readouts once", size=15, bold=True, align=PP_ALIGN.LEFT)
label(4.5, 5.2, 2.2, .3, [("judge scores + z  ", None), ("Φ", None)], size=11)
for r in range(4):
    for c in range(5):
        last = c == 4
        box(4.75 + c * .3, 5.5 + r * .26, .3, .26, fill=SALMON if last else TEAL, line=SALMON_E if last else TEAL_E,
            lw=.6, shape=MSO_SHAPE.RECTANGLE)
label(6.4, 5.2, 1.4, .3, [("human labels  ", None), ("Y", None)], size=11)
for r in range(4):
    for c in range(4):
        box(6.6 + c * .26, 5.5 + r * .26, .26, .26, fill=WHITE, line=TEAL_E, lw=.6, shape=MSO_SHAPE.RECTANGLE)
arrow(7.75, 6.02, 8.15, 6.02)
for i, (a, _) in enumerate(ATT):
    y = 5.5 + i * .26
    label(8.15, y - .03, .45, .3, [("w", None), (a, "sub")], size=11, italic=True, bold=True)
label(4.5, 6.6, 4.4, .45, [["5-fold CV picks λ per attribute;"],
                           [("ridge toward β", None), ("k", "sub"), ("e", None), ("k", "sub"),
                            ("; the raw score s", None), ("k", "sub"), (" is a candidate", None)]], size=10.5)
ln = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(9.05), Inches(4.85), Inches(9.05), Inches(6.95))
ln.line.color.rgb = RGBColor(0xcc, 0xcc, 0xcc)

# ------------------------------------------------------------------ step 3: deploy
label(9.2, 4.8, 3.9, .4, "3  Deploy", size=15, bold=True, align=PP_ALIGN.LEFT)
box(9.45, 5.35, 1.5, .75, fill=BLUE, line=BLUE_E, parts=[[("final prompt", None)], [("p", None), ("T", "sub")]],
    size=12, bold=True)
label(10.95, 5.5, .4, .4, "+", size=20, bold=True)
box(11.35, 5.35, 1.5, .75, fill=TEAL, line=TEAL_E, parts=[["readouts"], ["W"]], size=12, bold=True)
label(9.3, 6.25, 3.7, .7, [["one judge call per item;"], ["no extra LLM calls for the decoder"]], size=10.5)

# ------------------------------------------------------------------ split legend
label(.3, 7.0, 12.7, .35, "fit: error feedback + readout fitting    |    sel: accept / reject edits; readout fitting    |"
      "    test: final evaluation only", size=11, color=MUTED)

prs.save("paper/figures/method.pptx")
print("saved paper/figures/method.pptx")

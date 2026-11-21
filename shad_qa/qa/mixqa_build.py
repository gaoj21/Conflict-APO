"""ShAD built on error summarization, on the mixed QA set (document-aware context features).

  .venv/bin/python qa/mixqa_build.py [--workers 32]
R = the error-summary rule set chosen on val in qa/mixqa.py --docctx (its answers are reused).
arm A, errsum + statistics: the rule writer sees the same contrasting pairs plus the decoder's templated statements of
       the base answers, and writes up to 8 rules; top 1/2/4/8 chosen on val
arm B, errsum then ShAD on the residual: answer the fit questions with R, fit the decoder on these answers (answer
       features x document/question features -> wrong), translate the strongest statements, and add the top 1/2/4/all
       of them after R; a candidate is kept only if it beats R on val
test: overall EM, per source change vs base and vs R, conflicts.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
w = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "32"
sys.argv = ["mixqa.py", "--docctx", "--workers", w]
sys.path.insert(0, HERE)
import mixqa as M  # noqa: E402

import numpy as np, pandas as pd  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

gem = M.gem
gem.TAG.v = "qa/mixqa_build"
OUTB = f"{M.OUT}/build"; os.makedirs(OUTB, exist_ok=True)

RULES_STATS = """A model answers questions about several kinds of documents (narrative passages, financial reports with tables, multi-paragraph reading). Below are examples of its answers and the reference answers, followed by statistics measured on many more questions about which ways of writing the answer are judged wrong, and in which kinds of documents.

{pairs}

Statistics (from all training questions):
{stats}

Write at most 8 short rules for how to write the final answer line so that it matches the reference answers better. Where the statistics show that the right way depends on the kind of document or question, make the rule conditional. Order them from most to least important. Reply with only the rules, one per line, each starting with "- "."""


def per_source(df, ref):
    return M.per_source(df, ref)


def report(name, df, Ate, AR):
    pb, pr = per_source(df, Ate), per_source(df, AR)
    up = [s for s, (d, se, sig) in pb.items() if sig and d > 0]; dn = [s for s, (d, se, sig) in pb.items() if sig and d < 0]
    print(f"{name:22s} EM {100 * df.correct.mean():.1f}  vs base: " + "  ".join(f"{s} {d:+.1f}({se:.1f}){'*' if sig else ''}" for s, (d, se, sig) in pb.items())
          + f"  | up {up or '-'} down {dn or '-'}{'  CONFLICT' if up and dn else ''}")
    print(f"{'':22s}          vs errsum: " + "  ".join(f"{s} {d:+.1f}({se:.1f}){'*' if sig else ''}" for s, (d, se, sig) in pr.items()), flush=True)


def choose(cands, VR, bv, tag):
    best = (bv, None, 0)
    for n, inst in cands:
        em = M.run(inst, VR, f"build_val_{tag}_{n}").correct.mean()
        print(f"[val] {tag} {n}: EM {100 * em:.1f} (reference {100 * bv:.1f})", flush=True)
        if em > best[0]:
            best = (em, inst, n)
    print(f"[val] {tag}: chosen {best[2] if best[1] else 'reference (none beats it)'}", flush=True)
    return best


if __name__ == "__main__":
    srcs = ["drop", "finqa", "tatqa", "musique"]
    TR = [r for s in srcs for r in M.sample(s, "train", M.a.ntr)]; TE = [r for s in srcs for r in M.sample(s, "test", M.a.nte)]
    Atr, Ate = M.run(M.BASE, TR, "train_base"), M.run(M.BASE, TE, "test_base")
    per = Atr.groupby("source").cumcount(); fit, val = (per < M.a.ntr * 3 // 4).to_numpy(), (per >= M.a.ntr * 3 // 4).to_numpy()
    FR = [r for r, f in zip(TR, fit) if f]; VR = [r for r, v in zip(TR, val) if v]
    Af = Atr[fit].reset_index(drop=True); pt = M.pairs(Af)
    FF = json.load(open(f"{M.OUT}/features.json")); FA = [f for f in FF["answer"] if "reference" not in f["question"].lower()]; FC = FF["context"]
    P = pd.read_csv(f"{M.OUT}/probes_fit.csv"); C = P[[f"c:{f['name']}" for f in FC]].set_axis([f["name"] for f in FC], axis=1)
    SH = json.load(open(f"{M.OUT}/shad_rules.json"))

    ER = open(f"{M.OUT}/errsum_rules.txt").read().splitlines()
    vals = {n: M.run(M.BASE + M.SCOPE + "\n".join(ER[:n]), VR, f"val_errsum_top{n}").correct.mean() for n in (1, 2, 4, 8) if n <= len(ER)}
    nR = max(vals, key=vals.get); R = ER[:nR]; instR = M.BASE + M.SCOPE + "\n".join(R)
    print(f"errsum reference: top {nR} (val EM {100 * vals[nR]:.1f})\n" + "\n".join("  " + r for r in R), flush=True)
    AR = M.run(instR, TE, f"test_errsum")  # same file name as qa/mixqa.py when its choice was top nR

    # arm A
    af = f"{OUTB}/rules_A.txt"
    if not os.path.exists(af):
        stats = "\n".join("- " + x["statement"] for x in SH["rules"])
        txt = M.call(RULES_STATS.format(pairs=pt, stats=stats), 1500)
        open(af, "w").write("\n".join(l for l in (txt or "").splitlines() if l.strip().startswith("-")))
    RA = open(af).read().splitlines()
    print("\narm A rules:\n" + "\n".join("  " + r for r in RA), flush=True)
    bA = choose([(n, M.BASE + M.SCOPE + "\n".join(RA[:n])) for n in (1, 2, 4, 8) if n <= len(RA)], VR, vals[nR], "A")

    # arm B: decoder on the answers written with R
    AfR = M.run(instR, FR, "build_fit_R")
    pf = f"{OUTB}/probes_fit_R.csv"
    if not os.path.exists(pf):
        with ThreadPoolExecutor(M.a.workers) as ex:
            pa = list(ex.map(lambda x: M.probe(FA, f"Question: {x.question}\nAnswer: {M.plain(x.pred)}"), AfR.itertuples()))
        pd.DataFrame(pa).to_csv(pf, index=False)
    A = pd.concat([pd.read_csv(pf)[[f["name"] for f in FA]], pd.DataFrame([M.counts(p) for p in AfR.pred])], axis=1)
    lf = f"{OUTB}/shad_residual.json"
    if not os.path.exists(lf):
        W, info = M.decoder(A, C, 1 - AfR.correct.to_numpy())
        L = M.statements(W, FA, FC, A, C, None)[:8]
        for x in L:
            x["rule"] = "- " + (M.call(M.REPHRASE.format(s=x["statement"]), 200) or x["statement"]).strip().lstrip("- ")
        json.dump(dict(info=info, rules=L), open(lf, "w"), indent=1)
    SB = json.load(open(lf))
    print(f"\narm B decoder on answers with R: CV log loss {SB['info']['cv']:.4f} vs no-feature {SB['info']['null']:.4f}")
    for x in SB["rules"]:
        print(f"  {x['weight']:+.2f} [{x['kind']}] {x['answer_feature']}" + (f" x {x['context']}" if x["context"] else "") + f"\n      {x['rule']}")
    RB = [x["rule"] for x in SB["rules"]]
    bB = choose([(n, instR + "\n" + "\n".join(RB[:n])) for n in (1, 2, 4, len(RB))], VR, vals[nR], "B")

    print("\n== test")
    report("errsum (reference)", AR, Ate, AR)
    for name, b in (("A errsum+statistics", bA), ("B errsum+ShAD residual", bB)):
        df = AR if b[1] is None else M.run(b[1], TE, f"build_test_{name[0]}")
        report(name, df, Ate, AR)
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

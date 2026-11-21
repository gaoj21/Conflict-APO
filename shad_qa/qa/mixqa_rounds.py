"""Round-by-round APO with ShAD rules on the mixed QA set, the same structure as TextGrad x ShAD rounds on rating tasks.

  .venv/bin/python qa/mixqa_rounds.py [--rounds 6] [--nfit 100] [--workers 32]
The prompt = an instruction (edited by APO) + a block of ShAD rules (outside what APO edits), scoped to the Answer line.
Every round:
  1. answer the fit questions (nfit per source) with the current prompt
  2. APO step: the optimizer reads errors of these answers (balanced over sources) and rewrites the instruction
     (error-summary / TextGrad style); kept if val EM does not drop
  3. ShAD step (arm "apo+shad" only): on the current answers, fit the decoder (answer features x document/question
     features -> wrong), translate the two strongest statements on features not yet covered, append them to the block
     (at most 8); kept if val EM does not drop
Control arm "apo": the same rounds without step 3. Test: both final prompts; per source change vs base; conflicts.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
sys.argv = ["mixqa.py", "--docctx", "--workers", opt("--workers", "32")]
sys.path.insert(0, HERE)
import mixqa as M  # noqa: E402

import numpy as np, pandas as pd  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

gem = M.gem
ROUNDS, NFIT = int(opt("--rounds", "6")), int(opt("--nfit", "100"))
DIMS = opt("--dims", "")  # "hs2": answer features = the HelpSteer2 rating dimensions instead of auto-proposed ones
ARM = "apo+shad" + (f"_{DIMS}" if DIMS else "")
HS2 = [dict(name=n, question=q + " Rate from 1 (lowest) to 5 (highest).", type="count") for n, q in [
    ("helpfulness", "How helpful is the answer for the question?"),
    ("correctness", "How likely is the answer to be factually correct and complete for the question?"),
    ("coherence", "How clear and well-formed is the answer?"),
    ("complexity", "How much expertise or how elaborate is the answer?"),
    ("verbosity", "How long or detailed is the answer relative to what the question asks for?")]]
OUTR = f"{M.OUT}/rounds"; os.makedirs(OUTR, exist_ok=True)

APO = """You are improving the instruction a model follows when answering questions about several kinds of documents (narrative passages, financial reports with tables, multi-paragraph reading). It is followed by a fixed block of answer-formatting rules that you cannot change.

Current instruction:
<instruction>{inst}</instruction>
{block}
Accuracy on a training sample: {acc}.
Questions it got wrong (source, question, the model's answer, the reference answer):
{errs}

Write an improved instruction that fixes these kinds of errors while staying general across all kinds of documents. Reply with only the new instruction inside <instruction></instruction>."""


def prompt(inst, block):
    return inst.strip() + (M.SCOPE + "\n".join(block) if block else "")


def errors(df, k=24):
    rng = np.random.default_rng(len(df)); out = []
    for s, g in df[df.correct == 0].groupby("source"):
        out += [g.iloc[i] for i in rng.permutation(len(g))[:k // 4]]
    return "\n\n".join(f"[{x.source}] {x.question[:300]}\nModel: {M.plain(x.pred)}\nReference: {x.gold}" for x in out)


def shad_step(df, FA, FC, Cmap, covered, k=3):
    with ThreadPoolExecutor(M.a.workers) as ex:
        pa = list(ex.map(lambda x: M.probe(FA, f"Question: {x.question}\nAnswer: {M.plain(x.pred)}"), df.itertuples()))
    A = pd.concat([pd.DataFrame(pa)[[f["name"] for f in FA]], pd.DataFrame([M.counts(p) for p in df.pred])], axis=1)
    FD = [f for f in FC if re.match(r"(does|is) the document", f["question"].strip().lower())]  # conditions: document only
    C = pd.DataFrame([{f["name"]: Cmap[q][f["name"]] for f in FD} for q in df.qid])
    W, info = M.decoder(A, C, 1 - df.correct.to_numpy())
    L = [x for x in M.statements(W, FA, FD, A, C, None) if (x["answer_feature"], x["context"]) not in covered][:k]
    for x in L:
        x["rule"] = "- " + (M.call(M.REPHRASE.format(s=x["statement"]), 200) or x["statement"]).strip().lstrip("- ")
    return L, info


if __name__ == "__main__":
    srcs = ["drop", "finqa", "tatqa", "musique"]
    TR = [r for s in srcs for r in M.sample(s, "train", M.a.ntr)]; TE = [r for s in srcs for r in M.sample(s, "test", M.a.nte)]
    Atr, Ate = M.run(M.BASE, TR, "train_base"), M.run(M.BASE, TE, "test_base")
    per = Atr.groupby("source").cumcount()
    FR = [r for r, p in zip(TR, per) if p < NFIT]; VR = [r for r, p in zip(TR, per) if p >= M.a.ntr * 3 // 4]
    FF = json.load(open(f"{M.OUT}/features.json")); FA = [f for f in FF["answer"] if "reference" not in f["question"].lower()]; FC = FF["context"]
    if DIMS == "hs2":
        FA = HS2
    P = pd.read_csv(f"{M.OUT}/probes_fit.csv"); Cmap = {q: {f["name"]: v for f, v in zip(FC, row)} for q, row in
                                                       zip(P.qid, P[[f"c:{f['name']}" for f in FC]].to_numpy())}
    final = {}
    for arm in ("apo", ARM):  # apo is resumed from its state file when already done
        gem.TAG.v = f"qa/mixqa_rounds/{arm}"
        st = f"{OUTR}/{arm}.json"
        S = json.load(open(st)) if os.path.exists(st) else dict(inst=M.BASE, block=[], covered=[], tried=[], log=[])
        cur = M.run(prompt(S["inst"], S["block"]), VR, f"r_{arm}_val_start").correct.mean() if not S["log"] else S["log"][-1]["val"]
        for t in range(len(S["log"]), ROUNDS):
            tag = f"r_{arm}_{t}"
            F = M.run(prompt(S["inst"], S["block"]), FR, f"{tag}_fit")
            acc = ", ".join(f"{s} {100 * g.correct.mean():.0f}%" for s, g in F.groupby("source"))
            blk = ("\nFixed formatting rules:\n" + "\n".join(S["block"]) + "\n") if S["block"] else ""
            gem.SALT.v = f"{arm}{t}"
            txt = M.call(APO.format(inst=S["inst"], block=blk, acc=acc, errs=errors(F)), 1500, temperature=0.7)
            gem.SALT.v = ""
            m = re.findall(r"<instruction>(.*?)</instruction>", txt or "", re.S); cand = m[-1].strip() if m else S["inst"]
            v = M.run(prompt(cand, S["block"]), VR, f"{tag}_val_apo").correct.mean()
            apo_keep = v >= cur
            if apo_keep:
                S["inst"], cur = cand, v
                F = M.run(prompt(S["inst"], S["block"]), FR, f"{tag}_fit_after")
            rec = dict(round=t, apo_val=100 * v, apo_keep=bool(apo_keep))
            if arm != "apo":
                L, info = shad_step(F, FA, FC, Cmap, {tuple(c) for c in S["covered"] + S["tried"]})  # rejected ones are not retried
                tests = []  # every candidate rule is tested on its own; the best one is kept if val does not drop
                for i, x in enumerate(L):
                    nb = (S["block"] + [x["rule"]])[-8:]
                    tests.append((M.run(prompt(S["inst"], nb), VR, f"{tag}_val_shad{i}").correct.mean(), i, nb))
                v2, i, nb = max(tests, key=lambda t: t[0])
                shad_keep = v2 >= cur
                for j, x in enumerate(L):
                    (S["covered"] if shad_keep and j == i else S["tried"]).append([x["answer_feature"], x["context"]])
                if shad_keep:
                    S["block"], cur = nb, v2
                rec |= dict(shad_val=100 * v2, shad_keep=bool(shad_keep), new_rules=[L[i]["rule"]],
                            tested=[f"{100 * t[0]:.1f} {L[t[1]]['rule']}" for t in tests])
            rec["val"] = cur; S["log"].append(rec); json.dump(S, open(st, "w"), indent=1)
            print(f"[{arm}] round {t}: APO val {rec['apo_val']:.1f} {'keep' if apo_keep else 'revert'}"
                  + (f" | ShAD best {rec['shad_val']:.1f} {'keep' if rec['shad_keep'] else 'revert'} [" + " || ".join(rec["tested"]) + "]" if arm != "apo" else "")
                  + f" | current val {100 * cur:.1f}  (spent ${gem.spent()[0]:.2f})", flush=True)
        final[arm] = S
        print(f"[{arm}] final instruction:\n{S['inst']}\n[{arm}] final ShAD block:\n" + ("\n".join(S["block"]) or "(none)"), flush=True)

    print("\n== test (per source change vs base x100, SE; * beyond 2 SE)")
    AP = None
    final["scope only"] = dict(inst=M.BASE, block=["- Write the final answer line as the question asks."])  # control: the wrapper without rules
    for arm in ("base", "apo", "scope only", ARM):
        df = Ate if arm == "base" else M.run(prompt(final[arm]["inst"], final[arm]["block"]), TE, f"r_{arm.replace(' ', '_')}_test")
        AP = df if arm == "apo" else AP
        ps = M.per_source(df, Ate)
        up = [s for s, (d, se, sig) in ps.items() if sig and d > 0]; dn = [s for s, (d, se, sig) in ps.items() if sig and d < 0]
        print(f"{arm:9s} EM {100 * df.correct.mean():.1f}  " + "  ".join(f"{s} {d:+.1f}({se:.1f}){'*' if sig else ''}" for s, (d, se, sig) in ps.items())
              + f"  | up {up or '-'} down {dn or '-'}{'  CONFLICT' if up and dn else ''}")
        if arm == ARM:
            pr = M.per_source(df, AP)
            print(f"{'':9s} vs apo: " + "  ".join(f"{s} {d:+.1f}({se:.1f}){'*' if sig else ''}" for s, (d, se, sig) in pr.items())
                  + f"  overall {100 * (df.correct.mean() - AP.correct.mean()):+.1f}")
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

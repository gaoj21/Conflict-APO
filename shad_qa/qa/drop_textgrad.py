"""TextGrad (official package, the same loop as baselines/run_textgrad.py) on DROP, with and without ShAD rules.

  .venv/bin/python qa/drop_textgrad.py <tg|tg+shad|tg+edit> [--steps 12] [--every 3] [--nshad 1000] [--workers 32]
The optimized variable is the system prompt (start: "Think step by step."). Each step: a batch of 3 fit questions, the
loss LLM compares the model's response with the reference answer, backward, TGD step; kept if val EM (500) does not
drop. The user message is the passage + question (+ the ShAD rule block, scoped to the final Answer line) + the frozen
answer format, so TextGrad cannot edit the rules.
tg+shad: every `every` steps, the ShAD step of qa/drop_rounds_dev.py on `nshad` pool questions answered with the
current prompt (learned deviation features of wrong answers x learned question features -> rules, global and "only
for"); candidates close to an already rejected rule (embedding cosine > .9) are skipped; each candidate is tested
alone on val and the best is kept if val EM does not drop.
tg+edit: the same, but the ShAD step is qa/editfeat.py (token-class edits of the answers vs the references, question
n-gram conditions, templated rules; all its candidates tested). Its TextGrad steps draw the same batches and samples as
arm tg, so until the first kept rule both arms are identical.
Test: the final prompts on 1000 DROP validation questions.
"""
import hashlib, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]; opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
ARM = argv[0]; STEPS, EVERY, NSHAD, W = int(opt("--steps", "12")), int(opt("--every", "3")), int(opt("--nshad", "1000")), opt("--workers", "32")
sys.argv = ["drop_rounds_dev.py", "--workers", W]; sys.path.insert(0, HERE)
import drop_rounds_dev as RD  # noqa: E402

NF, D, S, gem = RD.NF, RD.D, RD.S, RD.gem
import textgrad as tg  # noqa: E402
import editfeat as EF  # noqa: E402
from textgrad.engine import EngineLM  # noqa: E402

OUT = f"{S.OUT}/textgrad"; os.makedirs(OUT, exist_ok=True)
gem.TAG.v = f"qa/{D.DS}/textgrad/{ARM}"
FORMAT = "End with exactly one line: Answer: <the answer>"


class Engine(EngineLM):
    def __init__(self, temperature, max_tokens=2000):
        self.model_string, self.temperature, self.max_tokens = gem.JUDGE, temperature, max_tokens

    def generate(self, prompt, system_prompt=None, **kw):
        msgs = ([dict(role="system", content=system_prompt)] if system_prompt else []) + [dict(role="user", content=prompt)]
        # cached at every temperature: the step's salt (gem.SALT) keeps steps apart, and arm tg+edit replays arm tg's
        # TextGrad calls exactly while the two arms still have the same prompt
        return gem.chat(msgs, temperature=self.temperature, max_tokens=self.max_tokens, cache=True)[0]

    def __call__(self, prompt, **kw):
        return self.generate(prompt, **kw)


model_engine, loss_engine, grad_engine, opt_engine = Engine(0.0, 3000), Engine(0.3), Engine(0.3), Engine(0.7, 4000)
tg.set_backward_engine(grad_engine, override=True)


def user_msg(r, block):
    return f"## {'Paragraphs' if D.DS in ('musique', 'hotpot') else 'Passage'}\n{r['ctx']}\n\n## Question\n{r['question']}" + (RD.D.SCOPE + "\n".join(block) if block else "") + f"\n\n{FORMAT}"


def run(sysp, block, R, tag):
    key = hashlib.sha1((sysp + "\x00" + "\n".join(block)).encode()).hexdigest()[:8]
    f = f"{OUT}/answers_{tag}_{key}.csv"
    if os.path.exists(f):
        return pd.read_csv(f)

    def one(r):
        try:
            txt = gem.chat([dict(role="system", content=sysp), dict(role="user", content=user_msg(r, block))], max_tokens=3000)[0]
        except gem.Abort:
            raise
        except Exception:  # noqa: BLE001
            txt = None
        c, f1, pred = S.score(txt, r)
        return dict(qid=r["qid"], question=r["question"], pred=pred, gold=r["gold_text"], correct=c, f1=f1)
    with ThreadPoolExecutor(int(W)) as ex:
        df = pd.DataFrame(ex.map(one, R))
    df.to_csv(f, index=False)
    return df


if __name__ == "__main__":
    TRp = S.sample("train", D.NFIT + D.NVAL + NSHAD)  # seed-0 permutation of the train split
    FR, VR, FS = TRp[:D.NFIT], TRp[D.NFIT:D.NFIT + D.NVAL], TRp[D.NFIT + D.NVAL:]; TE = S.sample("test", D.NTE)
    if ARM == "tg+shad":
        Cq = np.load(f"{RD.OUT}/q_codes.npy")[:NSHAD]; qnames = json.load(open(f"{RD.OUT}/q_names.json"))

    hf = f"{OUT}/{ARM}.json"
    H = json.load(open(hf)) if os.path.exists(hf) else dict(sys=D.S.BASE, block=[], rejected=[], log=[])
    cur = run(H["sys"], H["block"], VR, "val").correct.mean()
    print(f"[{ARM}] start val EM {100 * cur:.1f}", flush=True)
    rng = random.Random(0)
    for _ in range(len(H["log"])):
        rng.sample(FR, 3)  # keep the batch sequence on resume
    for step in range(len(H["log"]) + 1, STEPS + 1):
        sp = tg.Variable(H["sys"], requires_grad=True, role_description="system prompt of a model answering reading-comprehension questions")
        model = tg.BlackboxLLM(model_engine, system_prompt=sp)
        optim = tg.TGD(engine=opt_engine, parameters=[sp], constraints=["Do not write an output-format section; the output format is appended separately."])
        eval_fn = tg.loss.MultiFieldEvaluation(
            tg.Variable("Below are a passage with a question, the reference answer, and a model's response. Compare the model's "
                        "final answer with the reference answer (exact match after normalization is required) and criticize the "
                        "response wherever it deviates: what did it misread, compute wrongly, or write differently from the "
                        "reference? Be concise and specific.", requires_grad=False, role_description="evaluation instruction"),
            role_descriptions=["passage and question", "reference answer", "model response"], engine=loss_engine)
        gem.SALT.v = f"tg{step}" if ARM == "tg+edit" else f"{ARM}{step}"
        optim.zero_grad(); losses = []; batch = rng.sample(FR, 3)
        try:
            for r in batch:
                x = tg.Variable(user_msg(r, H["block"]), requires_grad=False, role_description="passage and question")
                y = tg.Variable(r["gold_text"], requires_grad=False, role_description="reference answer")
                losses.append(eval_fn([x, y, model(x)]))
            tg.sum(losses).backward()
            sp.gradients = sorted(sp.gradients, key=lambda g: g.value)  # textgrad keeps them in a set (order varies by run)
            optim.step()
            cand = sp.value
        except gem.Abort:
            raise
        except Exception as e:  # noqa: BLE001  (a failed call inside TextGrad: this step proposes nothing, i.e. a revert)
            print(f"[{ARM}] step {step}: TextGrad call failed ({str(e)[:200]}); step counted as revert", flush=True)
            cand = H["sys"]
        gem.SALT.v = ""
        v = run(cand, H["block"], VR, "val").correct.mean(); kept = v >= cur and cand != H["sys"]
        if kept:
            H["sys"], cur = cand, v
        rec = dict(step=step, tg_val=100 * v, tg_kept=bool(kept))
        if ARM == "tg+shad" and step % EVERY == 0:
            Fs = run(H["sys"], H["block"], FS, "shadfit")
            cands, dn = RD.shad_candidates(Fs, Cq, qnames, step, pfx=f"tg{step}_")
            cands = [c for c in dict.fromkeys(cands) if c not in H["block"]]
            if H["rejected"] and cands:  # skip candidates that restate an already rejected rule
                Ec, Er = NF.embed(cands, f"{OUT}/emb_c{step}.npy"), NF.embed(H["rejected"], f"{OUT}/emb_r{step}.npy")
                sim = (Ec / np.linalg.norm(Ec, axis=1, keepdims=True)) @ (Er / np.linalg.norm(Er, axis=1, keepdims=True)).T
                cands = [c for c, s in zip(cands, sim.max(1)) if s <= .9]
            tests = [(run(H["sys"], H["block"] + [c], VR, "val").correct.mean(), c) for c in cands[:4]]
            best = max(tests) if tests else (0, None)
            for vv, c in tests:
                if not (best[1] == c and vv >= cur):
                    H["rejected"].append(c)
            if best[1] and best[0] >= cur:
                H["block"].append(best[1]); cur = best[0]
            rec |= dict(shad_tested=[f"{100 * vv:.1f} {c}" for vv, c in tests], shad_kept=best[1] if best[1] and best[0] >= cur else None)
        if ARM == "tg+edit" and step % EVERY == 0:
            Fs = run(H["sys"], H["block"], FS, "shadfit")
            cands = [c["rule"] for c in EF.candidates(Fs, log=lambda *a, **k: None)]
            cands = [c for c in dict.fromkeys(cands) if c not in H["block"] and c not in H["rejected"]]
            tests = [(run(H["sys"], H["block"] + [c], VR, "val").correct.mean(), c) for c in cands]
            best = max(tests) if tests else (0, None); keep = best[1] is not None and best[0] >= cur
            H["rejected"] += [c for _, c in tests if not (keep and c == best[1])]
            if keep:
                H["block"].append(best[1]); cur = best[0]
            rec |= dict(shad_tested=[f"{100 * vv:.1f} {c}" for vv, c in tests], shad_kept=best[1] if keep else None)
        rec["val"] = 100 * cur; H["log"].append(rec); json.dump(H, open(hf, "w"), indent=1)
        print(f"[{ARM}] step {step}: TextGrad val {rec['tg_val']:.1f} {'keep' if kept else 'revert'}"
              + (f" | ShAD {'kept: ' + rec['shad_kept'] if rec.get('shad_kept') else 'none kept'} [{' || '.join(rec['shad_tested'])}]" if 'shad_tested' in rec else "")
              + f" | val {rec['val']:.1f}  (spent ${gem.spent()[0]:.2f})", flush=True)

    T = run(H["sys"], H["block"], TE, "test")
    print(f"[{ARM}] test EM {100 * T.correct.mean():.1f}  F1 {100 * T.f1.mean():.1f}\nfinal system prompt:\n{H['sys']}\nfinal rule block:\n" + ("\n".join(H["block"]) or "(none)"))
    print(f"done, spent ${gem.spent()[0]:.2f}", flush=True)

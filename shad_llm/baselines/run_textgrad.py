"""TextGrad (Yuksekgonul et al., 2024; official `textgrad` package) in the two multi-objective modes of Darshan &
Divekar (2026), whose code is not released; we follow their description on top of the official library:

  ccc     one judge prompt for all attributes; the loss LLM critiques all attributes jointly, the gradient LLM and the
          optimizer see all of them (Combined-Combined-Combined).
  single  one prompt per attribute, each optimized in its own run with a loss on that attribute only (Single-Task);
          at inference the judge is called once per attribute.

As in their setup: separate loss / gradient / optimizer LLMs (temperatures 0.3 / 0.3 / 0.7, all the judge model Gemma 4 26B A4B), 12
steps, the output format frozen (only the instruction text is optimized), and val=mae: a step is kept only if the
mean absolute error on 100 validation items does not increase, otherwise the prompt is reverted. The training loop
is the official TextGrad prompt-optimization loop (batch of 3, losses summed with tg.sum, backward, TGD step,
validation revert). Train items: fit; validation: 100 items of sel.
--shad (ccc only): ShAD in the loop. D0 (shad.py) is fitted on v0's fit scores; validation MAE and the ratings the
loss LLM compares with the human ratings are the decoded ones D0(s, z). Method name gets "_shad".
  .venv/bin/python baselines/run_textgrad.py <ds> ccc|single [--seed 0] [--steps 12] [--shad]
"""
import argparse, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
import textgrad as tg  # noqa: E402
from textgrad.engine import EngineLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("mode", choices=["ccc", "single"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=12); ap.add_argument("--batch", type=int, default=3)
ap.add_argument("--nval", type=int, default=100); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--shad", action="store_true")
ap.add_argument("--name", default=None)  # method name override
ap.add_argument("--shadrules", action="store_true")  # ShAD rules updated every step (see update_rules)
ap.add_argument("--rstep", type=float, default=0.5)  # share of ShAD's remaining length correction added per step
a = ap.parse_args()
assert not (a.shad and a.mode == "single")
ds = gem.DS(a.ds)
METHOD = a.name or (f"textgrad_{a.mode}" + ("_shad" if a.shad else ""))
gem.TAG.v = f"{a.ds}/{METHOD}/seed{a.seed}"
OUTD = ds.out(METHOD, a.seed)


class GemmaEngine(EngineLM):
    """textgrad engine backed by OpenRouter (same accounting as every other call)."""

    def __init__(self, model, temperature, max_tokens=2000):
        self.model_string, self.temperature, self.max_tokens = model, temperature, max_tokens

    def generate(self, prompt, system_prompt=None, **kwargs):
        msgs = ([dict(role="system", content=system_prompt)] if system_prompt else []) + [dict(role="user", content=prompt)]
        return gem.chat(msgs, model=self.model_string, temperature=self.temperature, max_tokens=self.max_tokens,
                        cache=self.temperature == 0)[0]

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


judge_engine = GemmaEngine(gem.JUDGE, 0.0)
loss_engine = GemmaEngine(gem.OPTIM, 0.3)
grad_engine = GemmaEngine(gem.OPTIM, 0.3)
opt_engine = GemmaEngine(gem.OPTIM, 0.7, max_tokens=4000)
tg.set_backward_engine(grad_engine, override=True)


RULES = ""  # --shadrules: rules derived from ShAD, kept outside the optimized system prompt (TextGrad cannot edit them)


def user_msg(text):
    if not a.shadrules:
        return text.strip() + "\n\n" + ds.FORMAT  # the frozen output format travels with the item
    return (f"(Word count: {len(text.split())})\n\n" + text.strip() + ("\n\n" + RULES if RULES else "") + "\n\n" + ds.FORMAT
            + "\nHalf points and decimals are allowed.")


def judge(instruction, rows, strict=True):
    return ds.judge_msgs(lambda r: [dict(role="system", content=instruction), dict(role="user", content=user_msg(r["user"]))],
                         rows, a.workers, strict)


DEC = None


def mae(instruction, rows, attrs):
    df = ds.worst_case(judge(instruction, rows, strict=False))  # a step that breaks the format is reverted
    if DEC:
        df = DEC.df(df)
    P = df[attrs].to_numpy(float)
    H = ds.gold.loc[df.id, attrs].to_numpy(float)
    return float(np.nanmean(np.abs(P - H)))


# ---------------------------------------------------------------- ShAD rules, updated every step (--shadrules)
from evaluate import anchored_fit, sp  # noqa: E402
ADJ = np.zeros((len(ds.A), 5)); TEXT = ""
_wc = np.array([len(r["user"].split()) for r in ds.rows["fit"] + ds.rows["sel"]])
CUTS = np.quantile(_wc, [.2, .4, .6, .8]).round().astype(int)
BLAB = [f"up to {CUTS[0]} words"] + [f"{CUTS[i] + 1}-{CUTS[i + 1]} words" for i in range(3)] + [f"more than {CUTS[3]} words"]


def rules_text(adj, text):
    A = ds.A
    num = [f"- {A[k]}: " + ", ".join(f"{l}: {'+' if v > 0 else ''}{v:.1f}" for l, v in zip(BLAB, adj[k]))
           for k in range(len(A)) if np.any(np.abs(adj[k]) >= .05)]
    out = ""
    if text:
        out += "## Rating guidance (learned from expert ratings)\n" + text
    if num:
        out += ("\n\n" if out else "") + ("## Length adjustment (learned from expert ratings)\nAfter rating each attribute "
                                           "from the evidence, adjust it for the word count given at the top:\n" + "\n".join(num))
    return out


def shad_correction(instruction):
    """Fit ShAD on the current judge's fit + sel ratings: remaining length correction (points per band) and a description
    of the remaining cross-attribute corrections for the LLM."""
    A, K = ds.A, len(ds.A)
    tr = ds.worst_case(pd.concat([judge(instruction, ds.rows[s], strict=False) for s in ("fit", "sel")], ignore_index=True))
    X = tr[A].to_numpy(float)
    wc = np.array([len(ds.text[i].split()) for i in tr.id])
    z = np.log(wc); zs = (z - z.mean()) / z.std()
    F = np.c_[(X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1), zs]
    Y = ds.gold.loc[tr.id, A].to_numpy(float); sy = Y.std(0)
    W = anchored_fit(F, (Y - Y.mean(0)) / sy)(np.eye(K + 1))
    b = np.searchsorted(CUTS, wc, side="left"); bz = np.array([zs[b == j].mean() for j in range(5)])
    dadj = np.stack([W[K, k] * sy[k] * (bz - bz.mean()) for k in range(K)])
    word = lambda r: ("about as strongly as" if r >= .8 else "more than half as strongly as" if r >= .5 else
                      "about a third as strongly as" if r >= .25 else "slightly, much less than")
    lines = []
    for k in range(K):
        parts = [f"items the judge rates {'higher' if W[j, k] > 0 else 'lower'} on {A[j]} should get higher {A[k]} ratings, "
                 f"{word(abs(W[j, k]) / max(abs(W[k, k]), 1e-6))} the {A[k]} evidence itself counts"
                 for j in np.argsort(-np.abs(W[:K, k])) if j != k and abs(W[j, k]) / max(abs(W[k, k]), 1e-6) >= .1]
        lines.append(f"- {A[k]}: " + ("; ".join(parts) if parts else "no correction needed") + ".")
    return dadj, "\n".join(lines)


def update_rules(instruction, cur_mae, val, step):
    """One ShAD step: refit ShAD on the current judge, add rstep x the remaining length correction, rewrite the text
    rules from the remaining cross-attribute corrections, keep the new rules if validation MAE does not increase."""
    global RULES, ADJ, TEXT
    dadj, corr = shad_correction(instruction)
    msg = (f"An LLM judge rates items with the rubric below. A calibration learned from human expert ratings shows what the "
           f"judge still gets wrong across attributes (length is handled separately; do not mention it).\n\n### Rubric\n"
           f"{instruction}\n\n### Current guidance\n{TEXT or '(none)'}\n\n### Remaining corrections\n{corr}\n\nRevise the "
           f"guidance, in plain words (no numbers, no formulas), so that the judge's own ratings make these corrections; keep "
           f"what still applies; at most 8 bullet points. Return only the guidance between <guidance> and </guidance>.")
    txt = gem.chat([dict(role="user", content=msg)], model=gem.OPTIM, temperature=0.7, max_tokens=2500, cache=False)[0]
    new_text = txt.split("<guidance>", 1)[1].split("</guidance>", 1)[0].strip() if "<guidance>" in txt and "</guidance>" in txt else TEXT
    old = (RULES, ADJ.copy(), TEXT)
    ADJ = ADJ + a.rstep * dadj; TEXT = new_text; RULES = rules_text(ADJ, TEXT)
    new_mae = mae(instruction, val, ds.A)
    kept = new_mae <= cur_mae
    if not kept:
        RULES, ADJ, TEXT = old
    print(f"[all] step {step} ShAD rules: val MAE {new_mae:.3f} {'keep' if kept else 'revert'}", flush=True)
    return (new_mae if kept else cur_mae), kept


def human_str(rid, attrs):
    return json.dumps({x: round(float(v), 2) for x, v in zip(attrs, ds.gold.loc[rid, attrs])})


def optimize(attrs, tag, idx=0):
    """One TextGrad run optimizing the judge prompt for `attrs` (all attributes for ccc, one for single)."""
    done = f"{OUTD}/hist_{tag}.json"
    if os.path.exists(done):  # this attribute's run already finished (resume after an interrupted job)
        return json.load(open(done))[-1]["prompt"]
    rng = random.Random(a.seed * 1000 + idx)
    train, val = list(ds.rows["fit"]), ds.subset("sel", a.nval, seed=a.seed)
    what = "all attributes" if len(attrs) > 1 else f"the attribute '{attrs[0]}' only"
    system_prompt = tg.Variable(ds.v0, requires_grad=True,
                                role_description="system prompt of an LLM judge that rates an item on several attributes")
    model = tg.BlackboxLLM(judge_engine, system_prompt=system_prompt)
    optimizer = tg.TGD(engine=opt_engine, parameters=[system_prompt], constraints=[
        "Keep the attribute names, what each attribute means, and the rating scale.",
        "Do not write an output-format section; the output format is appended separately."])
    if DEC:  # the loss LLM compares the decoded ratings with the human ratings
        eval_fn = tg.loss.MultiFieldEvaluation(
            tg.Variable(f"Below are an item, the ratings a group of human experts gave it, an LLM judge's response, and "
                        f"the judge's final ratings (its scores after a fixed calibration). Considering {what}, compare "
                        f"the judge's final ratings with the human ratings and criticize the judge's reasoning wherever "
                        f"a final rating deviates from the human rating: what did it overlook, overweight or misread? "
                        f"Be concise and specific.",
                        requires_grad=False, role_description="evaluation instruction"),
            role_descriptions=["item to rate", "human expert ratings", "LLM judge response", "judge's final ratings"],
            engine=loss_engine)
    else:
        eval_fn = tg.loss.MultiFieldEvaluation(
            tg.Variable(f"Below are an item, the ratings a group of human experts gave it, and an LLM judge's response "
                        f"with its ratings. Considering {what}, compare the judge's ratings with the human ratings and "
                        f"criticize the judge's reasoning wherever its rating deviates from the human rating: what did it "
                        f"overlook, overweight or misread? Be concise and specific.",
                        requires_grad=False, role_description="evaluation instruction"),
            role_descriptions=["item to rate", "human expert ratings", "LLM judge response"], engine=loss_engine)
    cur_mae = mae(system_prompt.value, val, attrs)
    hist = [dict(step=0, val_mae=cur_mae, kept=True, prompt=system_prompt.value)]
    print(f"[{tag}] step 0 val MAE {cur_mae:.3f}", flush=True)
    for step in range(1, a.steps + 1):
        batch = rng.sample(train, a.batch)
        optimizer.zero_grad()
        losses = []
        for r in batch:
            x = tg.Variable(user_msg(r["user"]), requires_grad=False, role_description="item to rate")
            y = tg.Variable(human_str(r["id"], attrs), requires_grad=False, role_description="human expert ratings")
            response = model(x)
            if DEC:
                sc = gem.parse_scores(response.value)
                P = DEC(np.array([[sc.get(k, np.nan) for k in ds.A]]), [r["id"]])[0]
                f = tg.Variable(json.dumps({k: round(float(v), 2) for k, v in zip(ds.A, P)}), requires_grad=False,
                                role_description="judge's final ratings")
                losses.append(eval_fn([x, y, response, f]))
            else:
                losses.append(eval_fn([x, y, response]))
        total_loss = tg.sum(losses)
        total_loss.backward()
        prev = system_prompt.value
        optimizer.step()
        cand = system_prompt.value  # saved for the per-candidate conflict analysis (scripts/tg_candidate_conflicts.py)
        new_mae = mae(system_prompt.value, val, attrs)
        kept = new_mae <= cur_mae  # val=mae: keep unless validation MAE increases
        if kept:
            cur_mae = new_mae
        else:
            system_prompt.set_value(prev)
        rec = dict(step=step, val_mae=new_mae, kept=kept, prompt=system_prompt.value, cand=cand)
        if a.shadrules:
            cur_mae, rk = update_rules(system_prompt.value, cur_mae, val, step)
            rec.update(rules=RULES, rules_kept=rk, val_mae_after_rules=cur_mae)
        hist.append(rec)
        print(f"[{tag}] step {step} val MAE {new_mae:.3f} {'keep' if kept else 'revert'}  "
              f"spent ${gem.spent(gem.TAG.v)[0]:.2f}", flush=True)
    json.dump(hist, open(f"{OUTD}/hist_{tag}.json", "w"), indent=1)
    return system_prompt.value


if a.shad:
    from shad import Decoder
    DEC = Decoder(ds, judge(ds.v0, ds.rows["fit"]))
if a.mode == "ccc":
    final = optimize(ds.A, "all")
    open(f"{OUTD}/final_prompt.txt", "w").write(final)
    if a.shadrules:
        open(f"{OUTD}/final_rules.txt", "w").write(RULES)
    for s in ("fit", "sel", "test"):
        gem.save_scores(ds, METHOD, a.seed, s, judge(final, ds.rows[s]))
else:
    with ThreadPoolExecutor(len(ds.A)) as ex:  # the per-attribute runs are independent: run them side by side
        finals = dict(zip(ds.A, ex.map(lambda kx: optimize([kx[1]], kx[1], kx[0]), enumerate(ds.A))))
    json.dump(finals, open(f"{OUTD}/final_prompts.json", "w"), indent=1)
    for s in ("fit", "sel", "test"):  # one call per attribute: attribute k is read from prompt k
        cols = {x: judge(finals[x], ds.rows[s]).set_index("id")[x] for x in ds.A}
        gem.save_scores(ds, METHOD, a.seed, s, pd.DataFrame(cols).reset_index().rename(columns={"index": "id"}))
print(a.ds, METHOD, a.seed, "done, spent $%.3f" % gem.spent(gem.TAG.v)[0])

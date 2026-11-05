"""TextGrad (Yuksekgonul et al., 2024; official `textgrad` package) in the two multi-objective modes of Darshan &
Divekar (2026), whose code is not released; we follow their description on top of the official library:

  ccc     one judge prompt for all attributes; the loss LLM critiques all attributes jointly, the gradient LLM and the
          optimizer see all of them (Combined-Combined-Combined).
  single  one prompt per attribute, each optimized in its own run with a loss on that attribute only (Single-Task);
          at inference the judge is called once per attribute.

As in their setup: separate loss / gradient / optimizer LLMs (temperatures 0.3 / 0.3 / 0.7, all the judge model), 12
steps, the output format frozen (only the instruction text is optimized), and val=mae: a step is kept only if the
mean absolute error on 100 validation items does not increase, otherwise the prompt is reverted. The training loop
is the official TextGrad prompt-optimization loop (batch of 3, losses summed with tg.sum, backward, TGD step,
validation revert). Train items: fit; validation: 100 items of sel.
  python baselines/run_textgrad.py <ds> ccc|single [--seed 0] [--steps 12]
"""
import argparse, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import textgrad as tg  # noqa: E402
from textgrad.engine import EngineLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("mode", choices=["ccc", "single"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=12); ap.add_argument("--batch", type=int, default=3)
ap.add_argument("--nval", type=int, default=100); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = common.DS(a.ds)
METHOD = f"textgrad_{a.mode}"
OUTD = ds.out(METHOD, a.seed)


class LLMEngine(EngineLM):
    """textgrad engine backed by the llm package."""

    def __init__(self, temperature):
        self.model_string, self.temperature = "llm", temperature

    def generate(self, prompt, system_prompt=None, **kwargs):
        msgs = ([dict(role="system", content=system_prompt)] if system_prompt else []) + [dict(role="user", content=prompt)]
        return common.chat(msgs, temperature=self.temperature)

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


judge_engine = LLMEngine(0.0)
loss_engine = LLMEngine(0.3)
grad_engine = LLMEngine(0.3)
opt_engine = LLMEngine(0.7)
tg.set_backward_engine(grad_engine, override=True)


def user_msg(text):
    return text.strip() + "\n\n" + ds.FORMAT  # the frozen output format travels with the item


def judge(instruction, rows, strict=True):
    return ds.judge_msgs(lambda r: [dict(role="system", content=instruction), dict(role="user", content=user_msg(r["user"]))],
                         rows, a.workers, strict)


def mae(instruction, rows, attrs):
    df = ds.worst_case(judge(instruction, rows, strict=False))  # a step that breaks the format is reverted
    P = df[attrs].to_numpy(float)
    H = ds.gold.loc[df.id, attrs].to_numpy(float)
    return float(np.nanmean(np.abs(P - H)))


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
            losses.append(eval_fn([x, y, response]))
        total_loss = tg.sum(losses)
        total_loss.backward()
        prev = system_prompt.value
        optimizer.step()
        new_mae = mae(system_prompt.value, val, attrs)
        kept = new_mae <= cur_mae  # val=mae: keep unless validation MAE increases
        if kept:
            cur_mae = new_mae
        else:
            system_prompt.set_value(prev)
        hist.append(dict(step=step, val_mae=new_mae, kept=kept, prompt=system_prompt.value))
        print(f"[{tag}] step {step} val MAE {new_mae:.3f} {'keep' if kept else 'revert'}", flush=True)
    json.dump(hist, open(f"{OUTD}/hist_{tag}.json", "w"), indent=1)
    return system_prompt.value


if a.mode == "ccc":
    final = optimize(ds.A, "all")
    open(f"{OUTD}/final_prompt.txt", "w").write(final)
    for s in ("fit", "sel", "test"):
        common.save_scores(ds, METHOD, a.seed, s, judge(final, ds.rows[s]))
else:
    with ThreadPoolExecutor(len(ds.A)) as ex:  # the per-attribute runs are independent: run them side by side
        finals = dict(zip(ds.A, ex.map(lambda kx: optimize([kx[1]], kx[1], kx[0]), enumerate(ds.A))))
    json.dump(finals, open(f"{OUTD}/final_prompts.json", "w"), indent=1)
    for s in ("fit", "sel", "test"):  # one call per attribute: attribute k is read from prompt k
        cols = {x: judge(finals[x], ds.rows[s]).set_index("id")[x] for x in ds.A}
        common.save_scores(ds, METHOD, a.seed, s, pd.DataFrame(cols).reset_index().rename(columns={"index": "id"}))
common.save_usage(OUTD)
print(a.ds, METHOD, a.seed, "done")

"""DSPy (official `dspy` package) optimizers: MIPROv2 (Opsahl-Ong et al., 2024) and GEPA (Agrawal et al., 2025).

The judge is a dspy.ChainOfThought program whose signature has the item as input and one integer output per attribute,
with the rubric v0 as the signature's instructions (DSPy's adapter supplies the output format). Task LM = the llm package's model
(temperature 0); prompt / reflection LM = the same model at temperature 1. Both optimizers run with auto="light" and their defaults.
DSPy optimizers need a per-example metric: agreement 1 - mean |pred - human| / range over the attributes (when
MIPROv2 bootstraps demos, i.e. trace is not None, a demo counts as correct if the agreement is >= 0.75: mean error at
most one point on a 1-5 scale); GEPA additionally gets textual feedback listing the predicted and human ratings.
Train: fit; validation: 200 items of sel.
  python baselines/run_dspy.py <ds> mipro|gepa|zeroshot [--seed 0]
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import dspy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("opt", choices=["mipro", "gepa", "zeroshot"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nval", type=int, default=200); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = common.DS(a.ds)
METHOD = f"dspy_{a.opt}"
OUTD = ds.out(METHOD, a.seed)

class LLM_LM(dspy.BaseLM):
    """DSPy LM routed through the llm package (DSPy never sees a provider)."""

    def forward(self, prompt=None, messages=None, **kwargs):
        from types import SimpleNamespace as NS
        kw = {**self.kwargs, **kwargs}
        msgs = messages or [dict(role="user", content=prompt)]
        txt = common.chat(msgs, temperature=kw.get("temperature", 0.0))
        return NS(choices=[NS(message=NS(content=txt), finish_reason="stop")], model=self.model,
                  usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})  # usage: see the llm package


lm_judge = LLM_LM("llm", temperature=0.0, cache=False)
lm_opt = LLM_LM("llm", temperature=1.0, cache=False)
dspy.configure(lm=lm_judge)

fields = {"item": (str, dspy.InputField(desc="the item to rate"))}
for x in ds.A:
    fields[x] = (int, dspy.OutputField(desc=f"integer from {ds.LO} to {ds.HI}"))
Judge = dspy.make_signature(fields, instructions=ds.v0)
program = dspy.ChainOfThought(Judge)


def example(r):
    return dspy.Example(item=r["user"], rid=r["id"], **{x: float(v) for x, v in zip(ds.A, ds.human([r["id"]])[0])}).with_inputs("item")


def agreement(gold, pred):
    try:
        P = np.array([float(getattr(pred, x)) for x in ds.A])
    except Exception:  # noqa: BLE001
        return 0.0
    H = np.array([gold[x] for x in ds.A])
    return float(1 - np.mean(np.abs(P - H)) / (ds.HI - ds.LO))


def metric(gold, pred, trace=None):
    s = agreement(gold, pred)
    return s >= 0.75 if trace is not None else s


def metric_fb(gold, pred, trace=None, pred_name=None, pred_trace=None):
    s = agreement(gold, pred)
    try:
        got = {x: getattr(pred, x) for x in ds.A}
    except Exception:  # noqa: BLE001
        got = "unparsable output"
    fb = f"Predicted ratings: {got}. Human expert ratings: { {x: round(gold[x], 2) for x in ds.A} }."
    return dspy.Prediction(score=s, feedback=fb)


train = [example(r) for r in ds.rows["fit"]]
val = [example(r) for r in ds.subset("sel", a.nval, seed=a.seed)]
if a.opt == "mipro":
    tp = dspy.MIPROv2(metric=metric, prompt_model=lm_opt, task_model=lm_judge, auto="light", num_threads=a.workers,
                      seed=a.seed)
    compiled = tp.compile(program, trainset=train, valset=val, requires_permission_to_run=False)
elif a.opt == "gepa":
    tp = dspy.GEPA(metric=metric_fb, auto="light", reflection_lm=lm_opt, num_threads=a.workers, seed=a.seed,
                   log_dir=f"{OUTD}/gepa_log")
    compiled = tp.compile(program, trainset=train, valset=val)
else:
    compiled = program
compiled.save(f"{OUTD}/program.json")


def run(r):
    try:
        p = compiled(item=r["user"])
        return dict(id=r["id"], **{x: float(getattr(p, x)) for x in ds.A})
    except Exception as e:  # noqa: BLE001
        return dict(id=r["id"], error=str(e)[:200])


for s in ("fit", "sel", "test"):
    print(f"final evaluation: {s} ({len(ds.rows[s])} items)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(list(ex.map(run, ds.rows[s])))
    for x in ds.A:
        if x not in df:
            df[x] = np.nan
    bad = df[ds.A].isna().any(axis=1).mean()
    if bad > 0.2:
        raise common.Abort(f"{METHOD} {s}: {bad:.0%} unparsed")
    common.save_scores(ds, METHOD, a.seed, s, df)
common.save_usage(OUTD)
print(a.ds, METHOD, a.seed, "done")

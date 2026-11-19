"""DSPy (official `dspy` package) optimizers: MIPROv2 (Opsahl-Ong et al., 2024) and GEPA (Agrawal et al., 2025).

The judge is a dspy.ChainOfThought program whose signature has the item as input and one integer output per attribute,
with the rubric v0 as the signature's instructions (DSPy's adapter supplies the output format). Task LM = Gemma 4 26B
(temperature 0); prompt / reflection LM = the same model at temperature 1. Both optimizers run with auto="light" and their defaults.
DSPy optimizers need a per-example metric: agreement 1 - mean |pred - human| / range over the attributes (when
MIPROv2 bootstraps demos, i.e. trace is not None, a demo counts as correct if the agreement is >= 0.75: mean error at
most one point on a 1-5 scale); GEPA additionally gets textual feedback listing the predicted and human ratings.
Train: fit; validation: 200 items of sel.
--shad: ShAD in the loop. The metric and GEPA's feedback use the decoded ratings D0(s, z) instead of the raw ones,
where D0 (shad.py) is fitted on the unoptimized program's fit scores (dspy_zeroshot). Method name gets "_shad".
  .venv/bin/python baselines/run_dspy.py <ds> mipro|gepa|zeroshot [--seed 0] [--shad]
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gem  # noqa: E402
import dspy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("opt", choices=["mipro", "gepa", "zeroshot"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nval", type=int, default=200); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--shad", action="store_true")
ap.add_argument("--weights", default=None)  # boosting: per-item weights (scripts/boost_weights.py) on the metric
ap.add_argument("--name", default=None)  # method name override
ap.add_argument("--rerun", default=None)  # judge again with the saved program of this method (noise control)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = a.name or (f"dspy_{a.opt}" + ("_shad" if a.shad else ""))
W = pd.read_csv(a.weights).set_index("id").w.to_dict() if a.weights else {}
DEC = None
if a.shad:
    from shad import Decoder
    DEC = Decoder(ds, pd.read_csv(f"{gem.OUT}/{a.ds}/dspy_zeroshot/seed0/scores_fit.csv"))
gem.TAG.v = f"{a.ds}/{METHOD}/seed{a.seed}"
OUTD = ds.out(METHOD, a.seed)
PRICE = {gem.JUDGE: (0.073e-6, 0.22e-6), "google/gemma-4-31b-it": (0.2e-6, 0.4e-6)}  # $ per input / output token (OpenRouter)

class GemLM(dspy.BaseLM):
    """DSPy LM routed through gem.chat (used for juapi, whose non-streaming GPT calls are capped at 10/min)."""

    def forward(self, prompt=None, messages=None, **kwargs):
        from types import SimpleNamespace as NS
        kw = {**self.kwargs, **kwargs}
        msgs = messages or [dict(role="user", content=prompt)]
        txt, it, ot, _ = gem.chat(msgs, model=self.model, temperature=kw.get("temperature", 0.0),
                                  max_tokens=kw.get("max_tokens", 2000), cache=False)
        return NS(choices=[NS(message=NS(content=txt), finish_reason="stop")], model=self.model,
                  usage={"prompt_tokens": it, "completion_tokens": ot, "total_tokens": it + ot})


if gem.PROVIDER in ("juapi", "llm"):  # route DSPy through gem.chat (and so through the llm package)
    lm_judge = GemLM(gem.JUDGE, temperature=0.0, max_tokens=2000, cache=False)
    lm_opt = GemLM(gem.OPTIM, temperature=1.0, max_tokens=8000, cache=False)
elif gem.PROVIDER == "local":  # OpenAI-compatible local server (vLLM)
    _kw = dict(api_base=gem.BASE_URL, api_key=gem._key(), cache=False, **json.loads(os.environ.get("GEM_EXTRA_BODY", "{}")))
    lm_judge = dspy.LM("openai/" + gem.JUDGE, temperature=0.0, max_tokens=2000, **_kw)
    lm_opt = dspy.LM("openai/" + gem.OPTIM, temperature=1.0, max_tokens=8000, **_kw)
else:
    key = gem._key()
    lm_judge = dspy.LM("openrouter/" + gem.JUDGE, api_key=key, temperature=0.0, max_tokens=2000, cache=False)
    lm_opt = dspy.LM("openrouter/" + gem.OPTIM, api_key=key, temperature=1.0, max_tokens=8000, cache=False)
dspy.configure(lm=lm_judge)

fields = {"item": (str, dspy.InputField(desc="the item to rate"))}
for x in ds.A:
    fields[x] = (int, dspy.OutputField(desc=f"integer from {ds.LO} to {ds.HI}"))
Judge = dspy.make_signature(fields, instructions=ds.v0)
program = dspy.ChainOfThought(Judge)


def example(r):
    return dspy.Example(item=r["user"], rid=r["id"], **{x: float(v) for x, v in zip(ds.A, ds.human([r["id"]])[0])}).with_inputs("item")


def ratings(gold, pred):
    """The ratings the metric compares with humans: raw, or decoded by D0 with --shad."""
    P = np.array([float(getattr(pred, x)) for x in ds.A])
    return DEC(P[None], [gold.rid])[0] if DEC else P


def agreement(gold, pred):
    try:
        P = ratings(gold, pred)
    except Exception:  # noqa: BLE001
        return 0.0
    H = np.array([gold[x] for x in ds.A])
    return float(1 - np.mean(np.abs(P - H)) / (ds.HI - ds.LO))


def metric(gold, pred, trace=None):
    s = agreement(gold, pred)  # demos are bootstrapped on plain agreement; candidates are scored on the weighted one
    return s >= 0.75 if trace is not None else s * W.get(gold.rid, 1.0)


def metric_fb(gold, pred, trace=None, pred_name=None, pred_trace=None):
    s = agreement(gold, pred) * W.get(gold.rid, 1.0)
    try:
        got = {x: round(float(v), 2) for x, v in zip(ds.A, ratings(gold, pred))}
    except Exception:  # noqa: BLE001
        got = "unparsable output"
    fb = f"Predicted ratings: {got}. Human expert ratings: { {x: round(gold[x], 2) for x in ds.A} }."
    return dspy.Prediction(score=s, feedback=fb)


def record_cost(lm, model):
    if gem.PROVIDER in ("juapi", "llm"):  # gem.chat already logged every call
        lm.history.clear(); return 0.0
    it = sum((h.get("usage") or {}).get("prompt_tokens", 0) or 0 for h in lm.history)
    ot = sum((h.get("usage") or {}).get("completion_tokens", 0) or 0 for h in lm.history)
    c = it * PRICE[model][0] + ot * PRICE[model][1] if model in PRICE else 0.0
    with gem._LOCK:
        db = gem._db()
        db.execute("insert into spend values (strftime('%s','now'),?,?,?,?,?)", (gem.TAG.v, model, it, ot, c))
        db.commit()
    lm.history.clear()
    return c


train = [example(r) for r in ds.rows["fit"]]
val = [example(r) for r in ds.subset("sel", a.nval, seed=a.seed)]
if a.rerun:
    compiled = program.deepcopy()
    compiled.load(f"{gem.OUT}/{a.ds}/{a.rerun}/seed{a.seed}/program.json")
elif a.opt == "mipro":
    tp = dspy.MIPROv2(metric=metric, prompt_model=lm_opt, task_model=lm_judge, auto="light", num_threads=a.workers,
                      seed=a.seed)
    compiled = tp.compile(program, trainset=train, valset=val, requires_permission_to_run=False)
elif a.opt == "gepa":
    tp = dspy.GEPA(metric=metric_fb, auto="light", reflection_lm=lm_opt, num_threads=a.workers, seed=a.seed,
                   log_dir=f"{OUTD}/gepa_log")
    compiled = tp.compile(program, trainset=train, valset=val)
else:
    compiled = program
c_opt = record_cost(lm_judge, gem.JUDGE) + record_cost(lm_opt, gem.OPTIM)
compiled.save(f"{OUTD}/program.json")
print(f"optimization spent ${c_opt:.3f}", flush=True)


def run(r):
    try:
        p = compiled(item=r["user"])
        return dict(id=r["id"], **{x: float(getattr(p, x)) for x in ds.A})
    except Exception as e:  # noqa: BLE001
        return dict(id=r["id"], error=str(e)[:200])


for s in ("fit", "sel", "test"):
    with ThreadPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(list(ex.map(run, ds.rows[s])))
    for x in ds.A:
        if x not in df:
            df[x] = np.nan
    bad = df[ds.A].isna().any(axis=1).mean()
    if bad > 0.2:
        raise gem.Abort(f"{METHOD} {s}: {bad:.0%} unparsed")
    gem.save_scores(ds, METHOD, a.seed, s, df)
c_eval = record_cost(lm_judge, gem.JUDGE)
print(a.ds, METHOD, a.seed, f"done, spent ${gem.spent(gem.TAG.v)[0]:.3f} (eval ${c_eval:.3f})")

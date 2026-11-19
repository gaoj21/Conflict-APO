"""Can a conflicting instruction edit be REPAIRED (keep its gain, remove its loss)? RewardBench 2.

For each conflicting edit (an instruction variant that raises one kind's accuracy and lowers another's):
1. Train split: score the base and the edited instruction; per kind accuracy change; flip sets
   fixed  = pairs the edit turned wrong -> right, broken = right -> wrong.
2. Repairs written by the optimizer LLM:
   flip   = sees the per-objective effects AND fixed / broken examples side by side (our method),
   agg    = sees only the per-objective effects (what a scalar / aggregate-feedback self-improver can see).
3. Test split: score base, edit and every repair; report the per-objective change vs base.

Usage: python scripts/conflict_repair.py <llm_cfg_a.json> <llm_cfg_b.json> [samples_per_repair]
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import litellm  # noqa: E402
from concurry.config import global_config as _concurry_config  # noqa: E402
from concurry.core.constants import RateLimitAlgorithm as _RLA  # noqa: E402

# As in run_local.py: the default SlidingWindow limiter rebuilds a per-token timestamp list on every acquire and pins
# the client at 100% CPU with long prompts (requests then queue client-side and time out); GCRA is O(1).
litellm.disable_aiohttp_transport = True
_concurry_config.defaults.rate_limit_algorithm = _RLA.GCRA

from when_gradients_collide import objectives as obj  # noqa: E402
from when_gradients_collide.algorithm import TextGrad  # noqa: E402
from when_gradients_collide.algorithm.mrb import _call, _parse_json, _with_instructions  # noqa: E402
from when_gradients_collide.config import wgc_config  # noqa: E402
from when_gradients_collide.experiment_config import LLMConfig  # noqa: E402
from when_gradients_collide.expt.dataset import RewardBench2  # noqa: E402
from when_gradients_collide.expt.runner import create_task_llm, get_initial_prompt  # noqa: E402
from when_gradients_collide.task_predictor import TaskPredictor  # noqa: E402

BASE = RewardBench2.tasks[0].task_instruction
EDITS = {
    "literal_if": BASE + " Above all, prefer the response that satisfies every explicit constraint in the prompt exactly (format, length, words to use or avoid), even if another response is more informative.",
    "factual": BASE + " Check every factual claim; a response with any false or fabricated claim is worse than one without errors.",
    "helpful": BASE + " Prefer the response that most fully and directly helps the user; refusals, hedging and caveats make a response worse.",
}
N_EX = 4  # fixed and broken examples each

REPAIR_PROMPT = """You are improving the instruction of an LLM judge that picks the best of four responses to a user prompt. The judge does NOT know what kind of prompt it is looking at (the kinds below are only used for evaluation).

Current instruction:
<INSTRUCTION>{base}</INSTRUCTION>

A proposed edit:
<EDITED>{edit}</EDITED>

Measured effect of the edit on held-out data (accuracy change per kind of prompt, in percentage points; token cost change in %):
{effects}
{examples}
Rewrite the edited instruction so that it KEEPS the gains and AVOIDS the losses. The judge cannot see the kind of prompt, so any condition must be decidable from the prompt and responses themselves. Keep it concise.

Respond with JSON only: {{"instruction": "<the full new instruction>", "why": "<one sentence>"}}"""

EX_BLOCK = """
Examples where the edit FIXED the verdict (wrong before, right after):
{fixed}

Examples where the edit BROKE the verdict (right before, wrong after):
{broken}
"""


def clip(x, n):
    x = " ".join(str(x).split())
    return x if len(x) <= n else x[:n] + " ..."


def example(s, p0, p1):
    resp = "\n".join(f"  Response {i}: {clip(s.inputs[f'response_{i}'], 350)}" for i in range(1, 5))
    why = lambda p: clip(str(p.raw_response or "")[:str(p.raw_response or "").rfind("{")], 300)  # noqa: E731
    return (f"- Prompt: {clip(s.inputs['prompt'], 300)}\n{resp}\n  Correct: {s.ground_truths['best']}; "
            f"before: {p0.task_outputs.get('best')} ({why(p0)}); after: {p1.task_outputs.get('best')} ({why(p1)})")


def main(cfg_a, cfg_b, n_rep=2):
    wgc_config.defaults.task_llm_timeout = 3600.0
    wgc_config.defaults.task_llm_max_tokens = 4096
    ds = RewardBench2(data_dir=str(REPO / "data"))
    obj.ACCURACY_TASKS = set(ds.accuracy_tasks)
    names = ["best"]
    objs = obj.build_objectives(names, None, ds.objective_split)
    keys = [o.name for o in objs if o.name not in ("instability", "length_bias", "calibration_emd")]
    prompt0 = get_initial_prompt(dataset=ds, tasks=ds.tasks)
    # Each client pool keeps at most 100 connections (httpx default): 3 pools per server -> ~300 in flight of 384 slots.
    llms = [create_task_llm(llm_config=LLMConfig(**json.load(open(c))), temperature=0.1) for c in (cfg_a, cfg_b) * 3]
    opt = create_task_llm(llm_config=LLMConfig(**json.load(open(cfg_a))), temperature=0.7)
    algo = TextGrad(task_llm=llms[0], gradient_llm=llms[0], optimizer_llm=llms[0], loss_llm=llms[0], tasks=ds.tasks,
                    steps=1, batch_size=3, eval_every=1, validation_metric="mae", name="repair")
    ctx = {k: v for k, v in algo._get_algorithm_context(step=0).items() if k != "batch"}
    batches = {sp: algo._sample_batch(data=getattr(ds, sp)(), dataset=ds, step=-2, full=True) for sp in ("train", "test")}

    def run(jobs):  # jobs: [(split, instruction)] -> preds aligned with the split's samples; two servers in parallel
        def one(j):
            i, (sp, instr) = j
            b = batches[sp]
            p = TaskPredictor.of("standard").predict(b, _with_instructions(prompt0, {"best": instr}), llms[i % len(llms)],
                                                     verbosity=0, failure_tolerance=float("inf"), **ctx)
            by = {x.sample_id: x for x in p}
            return [by.get(s.sample_id) for s in b.samples]
        out = []
        k = len(llms)  # one job per client pool at a time (3 per server): no server-side queue, no timeouts
        for w in range(0, len(jobs), k):
            with ThreadPoolExecutor(max_workers=k) as ex:
                out += list(ex.map(one, list(enumerate(jobs))[w: w + k]))
        return out

    def vals(sp, preds):
        S = batches[sp].samples
        ok = [i for i, p in enumerate(preds) if p is not None]
        return obj.compute_values([preds[i] for i in ok], [S[i] for i in ok], names, None, split=ds.objective_split)

    def effects(v, v0):
        d = []
        for k in keys:
            if k.startswith("rho_"):
                d.append(f"  {k.split('@')[1]}: {100 * (v[k] - v0[k]):+.1f}")
            else:
                d.append(f"  {k}: {100 * (v[k] / max(v0[k], 1e-9) - 1):+.0f}%")
        return "\n".join(d)

    def evaluate(repairs):
        jobs = [("test", BASE)] + [("test", e) for e in EDITS.values()] + [("test", v) for v in repairs.values()]
        t0 = time.time()
        test = run(jobs)
        print(f"test scored in {time.time() - t0:.0f}s\n", flush=True)
        labels = ["base", *EDITS, *[f"{a}/{b}{c}" for (a, b, c) in repairs]]
        T = dict(zip(labels, test))
        tv0 = vals("test", T["base"])
        n = len(batches["test"].samples)
        idx = np.random.default_rng(1).integers(0, n, (500, n))
        St = batches["test"].samples
        print("TEST: change vs base (accuracy points per kind; tokens %); * = 95% paired bootstrap CI excludes 0")
        print(f"{'':22s}" + "".join(f"{k.split('@')[-1][:12]:>13s}" for k in keys) + "       scalar")
        for lab in labels[1:]:
            v = vals("test", T[lab])
            boots = []
            for b in idx[:200]:
                okb = [i for i in b if T[lab][i] is not None and T["base"][i] is not None]
                sb = [St[i] for i in okb]
                x = obj.compute_values([T[lab][i] for i in okb], sb, names, None, split=ds.objective_split)
                y = obj.compute_values([T["base"][i] for i in okb], sb, names, None, split=ds.objective_split)
                boots.append({k: (x[k] - y[k] if k.startswith("rho_") else x[k] / max(y[k], 1e-9) - 1) for k in keys})
            cells = []
            for k in keys:
                lo, hi = np.percentile([bb[k] for bb in boots], [2.5, 97.5])
                d = 100 * (v[k] - tv0[k]) if k.startswith("rho_") else 100 * (v[k] / max(tv0[k], 1e-9) - 1)
                cells.append(f"{d:+12.1f}{'*' if lo > 0 or hi < 0 else ' '}")
            print(f"{lab:22s}" + "".join(cells) + f"   {obj.scalarize(v, tv0, objs):+.2f}")
        for k, v in repairs.items():
            print(f"\n[{k[0]}/{k[1]}{k[2]}] {v}")
        for llm in llms + [opt]:
            llm.stop()

    saved = REPO / "logs" / "rb2_repairs.json"
    if len(sys.argv) > 4 and sys.argv[4] == "--reuse" and saved.exists():  # skip train scoring + repair generation
        repairs = {tuple(k.split("|")[:2]) + (int(k.split("|")[2]),): v for k, v in json.load(open(saved)).items()}
        return evaluate(repairs)
    t0 = time.time()
    train = dict(zip(["base", *EDITS], run([("train", BASE)] + [("train", e) for e in EDITS.values()])))
    print(f"train scored in {time.time() - t0:.0f}s", flush=True)
    S = batches["train"].samples
    v0 = vals("train", train["base"])
    repairs = {}
    prompts, tags = [], []
    for name, edit in EDITS.items():
        v1 = vals("train", train[name])
        gain = [k for k in keys if k.startswith("rho_") and v1[k] - v0[k] > 0.02]
        loss = [k for k in keys if k.startswith("rho_") and v1[k] - v0[k] < -0.02]
        kind = lambda k: k.split("@")[1]  # noqa: E731
        right = lambda p, s: p is not None and p.task_outputs.get("best") == s.ground_truths["best"]  # noqa: E731
        rng = np.random.default_rng(0)
        fixed = [i for i, s in enumerate(S) if s.inputs["kind"] in map(kind, gain)
                 and not right(train["base"][i], s) and right(train[name][i], s)]
        broken = [i for i, s in enumerate(S) if s.inputs["kind"] in map(kind, loss)
                  and right(train["base"][i], s) and not right(train[name][i], s)]
        fixed = [int(i) for i in rng.permutation(fixed)[:N_EX]]
        broken = [int(i) for i in rng.permutation(broken)[:N_EX]]
        print(f"{name}: train gains {[kind(k) for k in gain]} losses {[kind(k) for k in loss]}; "
              f"{len(fixed)} fixed / {len(broken)} broken examples shown", flush=True)
        eff = effects(v1, v0)
        ex = EX_BLOCK.format(fixed="\n".join(example(S[i], train["base"][i], train[name][i]) for i in fixed) or "(none)",
                             broken="\n".join(example(S[i], train["base"][i], train[name][i]) for i in broken) or "(none)")
        for mode, block in (("flip", ex), ("agg", "")):
            for r in range(n_rep):
                prompts.append(REPAIR_PROMPT.format(base=BASE, edit=edit, effects=eff, examples=block))
                tags.append((name, mode, r))
    outs = _call(opt, prompts)
    for (name, mode, r), o in zip(tags, outs):
        p = _parse_json(o)
        if isinstance(p, dict) and isinstance(p.get("instruction"), str) and p["instruction"].strip():
            repairs[(name, mode, r)] = p["instruction"].strip()
    json.dump({f"{a}|{b}|{c}": v for (a, b, c), v in repairs.items()}, open(saved, "w"), indent=1)
    return evaluate(repairs)



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)

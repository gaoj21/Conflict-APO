"""Generate candidate instruction edits for the RewardBench 2 judge the way a self-improver would: from the base
judge's error statistics and error examples on the TRAIN split.

Usage: python scripts/rb2_candidates.py <llm_config.json> <base_train_preds.json> <out_variants.json> [n_calls] [per_call]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import litellm  # noqa: E402
from concurry.config import global_config as _concurry_config  # noqa: E402
from concurry.core.constants import RateLimitAlgorithm as _RLA  # noqa: E402

litellm.disable_aiohttp_transport = True
_concurry_config.defaults.rate_limit_algorithm = _RLA.GCRA
import anyio._backends._asyncio  # noqa: E402,F401
import httpx  # noqa: E402,F401

from when_gradients_collide.algorithm import TextGrad  # noqa: E402
from when_gradients_collide.algorithm.mrb import _call, _parse_json  # noqa: E402
from when_gradients_collide.config import wgc_config  # noqa: E402
from when_gradients_collide.experiment_config import LLMConfig  # noqa: E402
from when_gradients_collide.expt.dataset import RewardBench2  # noqa: E402
from when_gradients_collide.expt.runner import create_task_llm  # noqa: E402

BASE = RewardBench2.tasks[0].task_instruction
KINDS = ["focus", "factuality", "safety", "math", "precise_if"]

PROMPT = """You are improving the instruction of an LLM judge. The judge sees a user prompt and four responses and outputs the number (1-4) of the best response. The judge does NOT know which kind of prompt it is looking at.

Current instruction:
<INSTRUCTION>{base}</INSTRUCTION>

How the judge does on held-out training prompts (accuracy = picked the response humans marked best):
{stats}

Examples of wrong verdicts:
{examples}

Propose {k} DIFFERENT edits to the instruction, each one a short addition (one or two sentences) appended to the current instruction that would fix a kind of error you see. Make them diverse: different error patterns, different strategies (criteria, priorities, how to reason, how much to reason, what to ignore).

Respond with JSON only: {{"edits": ["<sentence(s) to append>", ...]}}"""


def clip(x, n):
    x = " ".join(str(x).split())
    return x if len(x) <= n else x[:n] + " ..."


def main(cfg, base_preds, out_path, n_calls=3, per_call=10):
    wgc_config.defaults.task_llm_timeout = 3600.0
    wgc_config.defaults.task_llm_max_tokens = 8192
    pred = json.load(open(base_preds))
    ds = RewardBench2(data_dir=str(REPO / "data"))
    llm = create_task_llm(llm_config=LLMConfig(**json.load(open(cfg))), temperature=0.9)
    algo = TextGrad(task_llm=llm, gradient_llm=llm, optimizer_llm=llm, loss_llm=llm, tasks=ds.tasks,
                    steps=1, batch_size=3, eval_every=1, validation_metric="mae", name="cand")
    by_id = {s.sample_id: s.inputs for s in algo._sample_batch(data=ds.train(), dataset=ds, step=-2, full=True).samples}
    lines = []
    for k in KINDS:
        rows = [r for r in pred.values() if r["kind"] == k]
        acc = np.mean([r["pred"] == r["gt"] for r in rows])
        picks = Counter(r["pred"] for r in rows)
        lines.append(f"- {k} prompts: accuracy {acc:.0%} (n={len(rows)}); picked positions {dict(sorted(picks.items(), key=str))}")
    stats = "\n".join(lines)
    wrong = [sid for sid, r in pred.items() if r["pred"] != r["gt"]]
    rng = np.random.default_rng(0)
    prompts = []
    for c in range(n_calls):
        ex = []
        for sid in rng.permutation(wrong)[:6]:
            r = by_id[sid]
            resp = "\n".join(f"  Response {j}: {clip(r[f'response_{j}'], 250)}" for j in range(1, 5))
            ex.append(f"- Prompt: {clip(r['prompt'], 250)}\n{resp}\n  Best: {pred[sid]['gt']}; judge picked: {pred[sid]['pred']}")
        prompts.append(PROMPT.format(base=BASE, stats=stats, examples="\n".join(ex) or "(none)", k=per_call))
    edits = []
    for o in _call(llm, prompts):
        p = _parse_json(o)
        for e in (p.get("edits") if isinstance(p, dict) else None) or []:
            if isinstance(e, str) and e.strip() and e.strip() not in edits:
                edits.append(e.strip())
    variants = {"base": BASE, **{f"c{i:02d}": f"{BASE} {e}" for i, e in enumerate(edits)}}
    json.dump(variants, open(out_path, "w"), indent=1)
    print(f"{len(edits)} candidate edits -> {out_path}")
    for i, e in enumerate(edits):
        print(f"c{i:02d}: {e[:200]}")
    llm.stop()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:6]))

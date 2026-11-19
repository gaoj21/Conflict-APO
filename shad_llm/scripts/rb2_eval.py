"""Score instruction variants on a RewardBench 2 split, one process per worker, one file per variant (resumable).

Usage: [RB_DATASET=RewardBench2Mini] python scripts/rb2_eval.py <llm_config.json> <split> <variants.json> <out_dir> <worker> <n_workers>
variants.json: {label: instruction}. Worker w scores labels w, w + n, ... and skips labels already saved.
Saved per label: {sample_id: {"kind", "gt", "pred", "err", "in", "out"}}.
"""

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import litellm  # noqa: E402
from concurry.config import global_config as _concurry_config  # noqa: E402
from concurry.core.constants import RateLimitAlgorithm as _RLA  # noqa: E402

# As in run_local.py: O(1) rate limiter, bounded httpx pool, async backends imported before worker threads start
# (concurrent event-loop threads otherwise deadlock on the import lock during retries).
litellm.disable_aiohttp_transport = True
_concurry_config.defaults.rate_limit_algorithm = _RLA.GCRA
import anyio._backends._asyncio  # noqa: E402,F401
import httpx  # noqa: E402,F401

from when_gradients_collide.algorithm import TextGrad  # noqa: E402
from when_gradients_collide.algorithm.mrb import _with_instructions  # noqa: E402
from when_gradients_collide.config import wgc_config  # noqa: E402
from when_gradients_collide.experiment_config import LLMConfig  # noqa: E402
from when_gradients_collide.expt import dataset as ds_module  # noqa: E402
from when_gradients_collide.expt.runner import create_task_llm, get_initial_prompt  # noqa: E402
from when_gradients_collide.task_predictor import TaskPredictor  # noqa: E402


def main(cfg, split, variants_path, out_dir, worker, n_workers):
    wgc_config.defaults.task_llm_timeout = 3600.0
    wgc_config.defaults.task_llm_max_tokens = 4096
    variants = json.load(open(variants_path))
    mine = [lab for i, lab in enumerate(sorted(variants)) if i % n_workers == worker]
    os.makedirs(out_dir, exist_ok=True)
    todo = [lab for lab in mine if not os.path.exists(os.path.join(out_dir, f"{lab}.json"))]
    if not todo:
        return
    ds = getattr(ds_module, os.environ.get("RB_DATASET", "RewardBench2"))(data_dir=str(REPO / "data"))
    prompt0 = get_initial_prompt(dataset=ds, tasks=ds.tasks)
    llm = create_task_llm(llm_config=LLMConfig(**json.load(open(cfg))), temperature=0.1)
    algo = TextGrad(task_llm=llm, gradient_llm=llm, optimizer_llm=llm, loss_llm=llm, tasks=ds.tasks,
                    steps=1, batch_size=3, eval_every=1, validation_metric="mae", name="rb2eval")
    batch = algo._sample_batch(data=getattr(ds, split)(), dataset=ds, step=-2, full=True)
    ctx = {k: v for k, v in algo._get_algorithm_context(step=0).items() if k != "batch"}
    for lab in todo:
        t0 = time.time()
        preds = TaskPredictor.of("standard").predict(batch, _with_instructions(prompt0, {"best": variants[lab]}), llm,
                                                     verbosity=0, failure_tolerance=float("inf"), **ctx)
        by = {p.sample_id: p for p in preds}
        out = {}
        for s in batch.samples:
            p = by.get(s.sample_id)
            u = (p.usage or {}) if p is not None else {}
            out[s.sample_id] = {"kind": s.inputs["kind"], "gt": s.ground_truths["best"],
                                "pred": None if p is None else p.task_outputs.get("best"),
                                "err": p is None or p.parser_error is not None, "in": u.get("in"), "out": u.get("out")}
        tmp = os.path.join(out_dir, f".{lab}.tmp")
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, os.path.join(out_dir, f"{lab}.json"))
        print(f"[w{worker}] {lab}: {len(preds)} preds in {time.time() - t0:.0f}s", flush=True)
    llm.stop()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))

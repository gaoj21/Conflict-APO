"""Sanity-check a backbone config: score the validation rows with the initial prompt.

Usage: python scripts/sanity_backbone.py <llm_config.json> [temperature]
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from when_gradients_collide.algorithm import TextGrad  # noqa: E402
from when_gradients_collide.config import wgc_config  # noqa: E402
from when_gradients_collide.experiment_config import LLMConfig  # noqa: E402
from when_gradients_collide.expt.dataset import SummEval  # noqa: E402
from when_gradients_collide.expt.runner import create_task_llm, get_initial_prompt  # noqa: E402
from when_gradients_collide.metrics import Metric  # noqa: E402
from when_gradients_collide.task_predictor import TaskPredictor  # noqa: E402


def main(cfg_path: str, temperature: float = 0.1):
    llm_config = LLMConfig(**json.load(open(cfg_path)))
    wgc_config.defaults.task_llm_timeout = 1800.0
    if llm_config.task_model.reasoning_effort:
        wgc_config.defaults.task_llm_max_tokens = llm_config.task_model.max_tokens
    ds = SummEval(data_dir=str(REPO / "data"))
    prompt = get_initial_prompt(dataset=ds, tasks=ds.tasks)
    llm = create_task_llm(llm_config=llm_config, temperature=temperature)
    algo = TextGrad(task_llm=llm, gradient_llm=llm, optimizer_llm=llm, loss_llm=llm, tasks=ds.tasks,
                    steps=1, batch_size=3, eval_every=1, validation_metric="mae", name="sanity")
    algo._setup_validation_batch(dataset=ds)
    t0 = time.time()
    preds = TaskPredictor.of("standard").predict(
        algo._validation_batch, prompt, llm, verbosity=0, failure_tolerance=float("inf"),
        **{k: v for k, v in algo._get_algorithm_context(step=0).items() if k != "batch"},
    )
    dt = time.time() - t0
    fails = sum(1 for p in preds if p.parser_error)
    spear = Metric.get_subclass("SpearmanCorrelation")
    rho = {}
    for t in ds.tasks:
        pairs = [(g.ground_truths[t.task_name], p.task_outputs[t.task_name])
                 for p, g in zip(preds, algo._validation_batch.samples) if t.task_name in p.task_outputs]
        rho[t.task_name] = round(spear.compute(y_true=[a for a, _ in pairs], y_pred=[b for _, b in pairs]), 3)
    print(f"n={len(preds)} parse_fail={fails} wall={dt:.1f}s rho={rho}")
    for p in preds[:2]:
        print("RAW:", repr(str(p.raw_response)[:200]))
    bad = [p for p in preds if p.parser_error][:2]
    for p in bad:
        print("BAD:", repr(str(p.raw_response)[:300]), "|", p.parser_error[:200])
    llm.stop()


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.1)

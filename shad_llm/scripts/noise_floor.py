"""Estimate the noise floor of per-task validation Spearman: evaluate the same prompt R times.

Δρ between two runs of an identical prompt is pure noise (task LLM sampling); the
attribution threshold harm_delta should sit above its spread.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from when_gradients_collide.algorithm import TextGrad  # noqa: E402
from when_gradients_collide.config import wgc_config  # noqa: E402
from when_gradients_collide.experiment_config import LLMConfig  # noqa: E402
from when_gradients_collide.expt.dataset import SummEval  # noqa: E402
from when_gradients_collide.expt.runner import create_task_llm, get_initial_prompt  # noqa: E402
from when_gradients_collide.metrics import Metric  # noqa: E402
from when_gradients_collide.task_predictor import TaskPredictor  # noqa: E402


def main(repeats: int = 5, temperature: float = 0.1):
    wgc_config.defaults.task_llm_timeout = 1800.0
    llm_config = LLMConfig(**json.load(open(REPO / "expt/configs/llm_local_dsv41.json")))
    ds = SummEval(data_dir=str(REPO / "data"))
    prompt = get_initial_prompt(dataset=ds, tasks=ds.tasks)
    task_llm = create_task_llm(llm_config=llm_config, temperature=temperature)
    algo = TextGrad(task_llm=task_llm, gradient_llm=task_llm, optimizer_llm=task_llm, loss_llm=task_llm,
                    tasks=ds.tasks, steps=1, batch_size=3, eval_every=1, validation_metric="mae", name="noise")
    algo._setup_validation_batch(dataset=ds)
    predictor = TaskPredictor.of("standard")
    spear = Metric.get_subclass("SpearmanCorrelation")
    rhos = []
    for _ in range(repeats):
        preds = predictor.predict(algo._validation_batch, prompt, task_llm, verbosity=0,
                                  **{k: v for k, v in algo._get_algorithm_context(step=0).items() if k != "batch"})
        r = {}
        for t in ds.tasks:
            yt = [g.ground_truths[t.task_name] for g in algo._validation_batch.samples]
            yp = [p.task_outputs.get(t.task_name) for p in preds]
            r[t.task_name] = spear.compute(y_true=yt, y_pred=yp)
        rhos.append(r)
        print(r, flush=True)
    for t in ds.tasks:
        deltas = [a[t.task_name] - b[t.task_name] for a, b in combinations(rhos, 2)]
        print(f"{t.task_name:12s} mean rho={np.mean([r[t.task_name] for r in rhos]):.3f} "
              f"|Δρ| mean={np.mean(np.abs(deltas)):.3f} max={np.max(np.abs(deltas)):.3f} sd(Δ)={np.std(deltas):.3f}")
    task_llm.stop()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)

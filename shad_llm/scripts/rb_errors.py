"""Inspect RewardBench parse errors with the initial prompt on a small sample of reasoning / other pairs."""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from when_gradients_collide.algorithm import TextGrad  # noqa: E402
from when_gradients_collide.config import wgc_config  # noqa: E402
from when_gradients_collide.data_structures import Batch  # noqa: E402
from when_gradients_collide.experiment_config import LLMConfig  # noqa: E402
from when_gradients_collide.expt import dataset as ds_module  # noqa: E402
from when_gradients_collide.expt.runner import create_task_llm, get_initial_prompt  # noqa: E402
from when_gradients_collide.task_predictor import TaskPredictor  # noqa: E402

llm_config = LLMConfig(**json.load(open(sys.argv[1])))
wgc_config.defaults.task_llm_timeout = 1800.0
wgc_config.defaults.task_llm_max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
ds = getattr(ds_module, sys.argv[2] if len(sys.argv) > 2 else "RewardBench")(data_dir=str(REPO / "data"))
prompt = get_initial_prompt(dataset=ds, tasks=ds.tasks)
llm = create_task_llm(llm_config=llm_config, temperature=0.1)
algo = TextGrad(task_llm=llm, gradient_llm=llm, optimizer_llm=llm, loss_llm=llm, tasks=ds.tasks,
                steps=1, batch_size=3, eval_every=1, validation_metric="mae", name="err")
batch = algo._sample_batch(data=ds.train(), dataset=ds, step=-2, full=True)
samples = [s for s in batch.samples][:200]
ctx = {k: v for k, v in algo._get_algorithm_context(step=0).items() if k != "batch"}
preds = TaskPredictor.of("standard").predict(Batch(step=-2, samples=samples), prompt, llm, verbosity=0,
                                             failure_tolerance=float("inf"), **ctx)
by = {p.sample_id: p for p in preds}
err = Counter()
tot = Counter()
shown = 0
for s in samples:
    p = by.get(s.sample_id)
    tot[s.inputs["kind"]] += 1
    if p is None or p.parser_error:
        err[s.inputs["kind"]] += 1
        if p is not None:
            err["tok_" + str(min(4096, ((p.usage or {}).get("out") or 0) // 512 * 512))] += 1
        if shown < 3 and p is not None:
            shown += 1
            raw = str(p.raw_response or "")
            print(f"--- kind={s.inputs['kind']} out_tokens={(p.usage or {}).get('out')} len={len(raw)}")
            print("HEAD:", repr(raw[:300]))
            print("TAIL:", repr(raw[-300:]))
            print("ERR:", str(p.parser_error)[:200])
print("errors by kind:", {k: f"{err[k]}/{tot[k]}" for k in tot})
print("PROMPT:\n" + prompt.render_task_prompt(sample=samples[0])[:1500])
llm.stop()

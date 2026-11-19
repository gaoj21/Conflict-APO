"""ProTeGi (Pryzant et al., 2023) with the official code (third_party/LMOps-main/prompt_optimization, unmodified).

Only the task glue is ours: the official optimizer, UCB-bandit evaluator and main-loop logic are used as released; the
LLM call (utils.chatgpt) is routed to the Gemma optimizer, the predictor is the Gemma judge on the rendered prompt, the
task's metric is mean per-attribute Spearman (the official scorer is 0/1 accuracy), and an "error" is an item whose
predicted ratings differ from the rounded human ratings. The official prompt skeleton is used (# Task / # Output
format / # Prediction); ProTeGi edits only the # Task section. Default hyperparameters of the official main.py with
the README's UCB evaluator. Train items = fit + sel (the labelled pool every method gets).
  .venv/bin/python baselines/run_protegi.py <ds> [--seed 0] [--rounds 6]
"""
import argparse, json, os, random, sys
import concurrent.futures.process  # noqa: F401  the official evaluators reference it in an except clause
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "LMOps-main", "prompt_optimization"))
import gem  # noqa: E402
import utils, optimizers, evaluators  # noqa: E402  (official ProTeGi modules)
from liquid import Template  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
ds = gem.DS(a.ds)
METHOD = "protegi"
gem.TAG.v = f"{a.ds}/{METHOD}/seed{a.seed}"
random.seed(a.seed); np.random.seed(a.seed)
OUTD = ds.out(METHOD, a.seed)


def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024, presence_penalty=0, frequency_penalty=0,
            logit_bias={}, timeout=10):  # noqa: B006  same signature as the official utils.chatgpt
    return [gem.chat([dict(role="user", content=prompt)], model=gem.OPTIM, temperature=temperature, max_tokens=max_tokens,
                     cache=False)[0] for _ in range(n)]


utils.chatgpt = chatgpt


def render(prompt, text):
    return Template(prompt).render(text=text)


class Predictor:
    def __init__(self, opt):
        self.opt = opt

    def inference(self, ex, prompt):
        try:
            rendered = render(prompt, ex["text"])
        except Exception:  # noqa: BLE001  a candidate whose text breaks the Liquid template: every rating unparsed
            return tuple(-1 for _ in ds.A)
        txt = gem.chat([dict(role="user", content=rendered)])[0]
        sc = gem.parse_scores(txt)
        return tuple(int(sc[x]) if x in sc else -1 for x in ds.A)


def spearman_of(exs, preds):
    P = np.array(preds, float); P[P < 0] = (ds.LO + ds.HI) / 2  # unparsed -> scale midpoint
    v = ds.spearman(pd.DataFrame(P, columns=ds.A).assign(id=[e["id"] for e in exs]))
    return float(np.nan_to_num(np.nanmean(v))) if np.isfinite(v).any() else 0.0


class Task:
    """Multi-attribute rating task in the interface of the official tasks.ClassificationTask."""

    def __init__(self, max_threads):
        self.max_threads = max_threads

    def get_train_examples(self):
        return [dict(id=r["id"], text=r["user"], label=tuple(int(round(v)) for v in ds.human([r["id"]])[0]))
                for r in ds.rows["fit"] + ds.rows["sel"]]

    def stringify_prediction(self, pred):
        return json.dumps(dict(zip(ds.A, pred)))

    def evaluate(self, predictor, prompt, test_exs, n=100):
        exs = test_exs[:n]
        with ThreadPoolExecutor(self.max_threads) as ex:
            preds = list(ex.map(lambda e: predictor.inference(e, prompt), exs))
        return spearman_of(exs, preds), [e["text"] for e in exs], [e["label"] for e in exs], preds


class SpearmanScorer:
    """Replaces the official Cached01Scorer: mean per-attribute Spearman of each prompt on the sampled items."""

    def __call__(self, predictor, prompts, data, agg="mean", max_threads=1):
        with ThreadPoolExecutor(max_threads) as ex:
            preds = list(ex.map(lambda pe: predictor.inference(pe[1], pe[0]), [(p, e) for p in prompts for e in data]))
        return [spearman_of(data, preds[i * len(data):(i + 1) * len(data)]) for i in range(len(prompts))]


# ---- official main.py defaults (README quickstart uses --evaluator ucb)
config = dict(task=a.ds, max_threads=a.workers, temperature=0.0, optimizer="nl-gradient", rounds=a.rounds, beam_size=4,
              n_test_exs=400, minibatch_size=64, n_gradients=4, errors_per_gradient=4, gradients_per_error=1,
              steps_per_gradient=1, mc_samples_per_step=2, max_expansion_factor=8, engine="chatgpt", evaluator="ucb",
              scorer="01", eval_rounds=8, eval_prompts_per_round=8, samples_per_eval=32, c=1.0, knn_k=2, knn_t=0.993,
              reject_on_errors=False)
config["eval_budget"] = config["samples_per_eval"] * config["eval_rounds"] * config["eval_prompts_per_round"]

task = Task(a.workers)
scorer = SpearmanScorer()
evaluator = evaluators.UCBBanditEvaluator(config)
bf_eval = evaluators.BruteForceEvaluator(config)
gpt4 = Predictor(config)
optimizer = optimizers.ProTeGi(config, evaluator, scorer, a.workers, bf_eval)
train_exs = task.get_train_examples()

prompt0 = f"# Task\n{ds.v0}\n\n# Output format\n{ds.FORMAT}\n\n# Prediction\n{{{{ text }}}}\n"
candidates = [prompt0]
log = open(f"{OUTD}/log.jsonl", "a")
# ---- the official main loop (main.py), without its per-round test evaluation
for rnd in range(config["rounds"] + 1):
    if rnd > 0:
        candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)
    scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
    [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))
    candidates = candidates[:config["beam_size"]]
    scores = scores[:config["beam_size"]]
    log.write(json.dumps(dict(round=rnd, scores=list(scores), candidates=list(candidates),
                              spent=gem.spent(gem.TAG.v)[0])) + "\n"); log.flush()
    print(f"round {rnd}: beam scores {np.round(scores, 3)}  spent ${gem.spent(gem.TAG.v)[0]:.2f}", flush=True)

final = candidates[0]
open(f"{OUTD}/final_prompt.txt", "w").write(final)
for s in ("fit", "sel", "test"):
    gem.save_scores(ds, METHOD, a.seed, s, ds.judge_msgs(lambda r: [dict(role="user", content=render(final, r["user"]))],
                                                          ds.rows[s], a.workers))
print(a.ds, METHOD, a.seed, "done, spent $%.3f" % gem.spent(gem.TAG.v)[0])

"""GRPO on TL;DR summarization with an LLM-judge reward: the raw judge score or the ShAD-decoded score.

  python grpo_tldr.py --reward raw  --out runs/raw
  python grpo_tldr.py --reward shad --out runs/shad

Everything except the reward is identical between the two runs. Every judged completion is logged to
<out>/rewards.jsonl with BOTH rewards and its length, so each run also shows the reward it does not optimize.
Checkpoints every --save steps go to <out>/checkpoint-<step> for eval_gold.py.
"""
import argparse, json, os, re, time

import numpy as np
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

import data
from judge import Judge

ap = argparse.ArgumentParser()
ap.add_argument("--reward", choices=["raw", "shad"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--prompts_per_step", type=int, default=32)
ap.add_argument("--generations", type=int, default=8)
ap.add_argument("--micro_batch", type=int, default=16, help="completions per forward/backward pass")
ap.add_argument("--max_len", type=int, default=96, help="max completion tokens")
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient (TRL default 0)")
ap.add_argument("--save", type=int, default=25)
ap.add_argument("--vllm", action="store_true", help="generate with vLLM colocated on the training GPU (faster)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)

judge = Judge()
K = judge.A.index("overall")
LOG = open(f"{a.out}/rewards.jsonl", "a")
STEP = [0]


def text_of(c):
    t = c[-1]["content"] if isinstance(c, list) else c
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()  # Qwen3 runs with thinking off; strip any left


def reward(prompts, completions, title, post, **kw):
    summ = [text_of(c) for c in completions]
    t0 = time.time()
    S = judge.score(title, post, summ)
    raw = np.where(np.isnan(S["raw"][:, K]), judge.dec.d["fill"][K], S["raw"][:, K])
    shad = S["shad"][:, K]
    STEP[0] += 1
    LOG.write(json.dumps(dict(call=STEP[0], t=time.time(), judge_s=round(time.time() - t0, 1), n=len(summ),
                              parsed=float(S["parsed"].mean()), raw=raw.tolist(), shad=shad.tolist(),
                              words=S["words"].tolist(), raw_all=S["raw"].tolist(), example=summ[0][:300])) + "\n")
    LOG.flush()
    print(f"[reward call {STEP[0]}] raw {raw.mean():.3f}  shad {shad.mean():.3f}  words {S['words'].mean():.1f}  "
          f"parsed {S['parsed'].mean():.0%}  judge {time.time() - t0:.0f}s", flush=True)
    return (raw if a.reward == "raw" else shad).tolist()


train = Dataset.from_list([{k: v for k, v in r.items() if k != "reference"}
                           for r in data.rows("train", a.steps * a.prompts_per_step, seed=a.seed)])
per_step = a.prompts_per_step * a.generations
cuda = torch.cuda.is_available()
cfg = GRPOConfig(
    output_dir=a.out, max_steps=a.steps, learning_rate=a.lr, beta=a.beta, seed=a.seed,
    num_generations=a.generations, max_completion_length=a.max_len, temperature=1.0,
    per_device_train_batch_size=a.micro_batch, gradient_accumulation_steps=per_step // a.micro_batch,
    chat_template_kwargs={"enable_thinking": False},
    bf16=cuda, gradient_checkpointing=cuda, logging_steps=1, save_steps=a.save, save_only_model=True,
    report_to="none", log_completions=False, use_vllm=a.vllm, vllm_mode="colocate",
)
trainer = GRPOTrainer(model=a.model, reward_funcs=reward, args=cfg, train_dataset=train)
trainer.save_model(f"{a.out}/checkpoint-0")  # the starting policy, for the eval curve
trainer.train()
trainer.save_model(f"{a.out}/checkpoint-{a.steps}")

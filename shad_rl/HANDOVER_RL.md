# ShAD-RL 交接文档：TL;DR 上用 judge 做 GRPO reward

代码：<https://github.com/gaoj21/Conflict-APO/tree/shad-rl/shad_rl>（分支 `shad-rl`，目录 `shad_rl/`）

## 1. 要回答的问题

用 LLM judge 的分数当 RL reward 时，policy 会不会学会钻 judge 的空子（judge 分数一直涨，真实质量不涨甚至下降）？
把 judge 分数换成 ShAD 解码后的分数，能不能减轻这种现象？

只做两组，除了 reward 以外完全相同：

| 组 | reward |
|---|---|
| `raw` | judge 给的 overall 分（1–7） |
| `shad` | ShAD 解码后的 overall：judge 的 4 个维度分数 + log 长度 → 人工打分尺度 |

两组都会同时记录两种 reward，所以每组也能看到自己没有优化的那个分数怎么变化。

## 2. 为什么选 TL;DR，以及之前已经知道什么

- TL;DR 是 judge 本身有信息量的数据集：overall 维度上，judge 原始分和人工分的 Spearman 是 0.509，ShAD 解码后是 0.544。
  只用长度只有 0.24，"judge 分 + 长度"仍是 0.509，所以 ShAD 的提升来自维度之间的组合（主要是 accuracy 的权重 +0.136），不是长度。
- 离线检查（3500 条 2020 年 PPO / 监督学习等模型的摘要）：judge **没有**系统性高估 PPO 输出；ShAD 在每一类摘要内部 Spearman 高 0.01–0.07。
  但那些 PPO 模型是针对 OpenAI 自己的 reward model 训练的，没有针对我们这个 judge 钻空子，所以离线看不到 hacking。这就是要跑在线实验的原因。
- 已知风险：ShAD 给 accuracy 正权重，对直接抄原文的抽取式摘要会打高分（离线偏差从 +0.03 变成 +0.24）。policy 有可能学会抄原文，要看生成的样例。

## 3. 文件

| 文件 | 作用 |
|---|---|
| `decoder_tldr.json` | ShAD decoder 参数，用 Gemma v0 在论文 TL;DR fit + sel（约 800 条）的打分拟合；复现论文 test Spearman 0.544 |
| `judge.py` | 调 judge 服务器（OpenAI 兼容），prompt 和论文完全一致（已逐字核对）；返回原始分和解码分；每次调用 180 秒超时、重试 4 次 |
| `data.py` | `trl-lib/tldr`：train 划分做 GRPO 的 prompt，test 划分的固定 300 篇做评估（带人工参考 TL;DR） |
| `grpo_tldr.py` | TRL GRPO 训练，`--reward raw` 或 `--reward shad` |
| `eval_gold.py` | 评估每个 checkpoint：proxy（两种 judge 分）、长度、gold（更强的模型判断和人工参考摘要比谁好，两种顺序各问一次） |
| `run_rl.sh` | 依次训练两组，再评估 |

## 4. 环境

```bash
git clone -b shad-rl git@github.com:gaoj21/Conflict-APO.git
cd Conflict-APO/shad_rl
python -m venv .venv && source .venv/bin/activate
pip install torch            # 装和本机 CUDA 匹配的版本
pip install -r requirements_rl.txt
pip install vllm             # judge 服务器用；也可以装在另一个环境
```

CPU 小测试用的版本是 trl 1.18.2、transformers 5.24.0。换版本的话，`GRPOConfig` 的参数名可能会变。

## 5. 运行

### 5.1 起 judge 服务器（Gemma，和之前 QA 实验一样）

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve <Gemma 4 26B A4B instruct 的 HF 名字> \
  --served-model-name google/gemma-4-26b-a4b-it --max-model-len 8192 --port 8000
export JUDGE_BASE_URL=http://localhost:8000/v1
export JUDGE_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'   # 关掉 thinking，和拟合 decoder 时一致
```

decoder 是在关掉推理的 Gemma 打分上拟合的，judge 必须用同一个模型、同样关掉推理，否则 decoder 不适用。

### 5.2 先确认 judge 打分正常

```bash
python -c "
from judge import Judge; j = Judge()
print(j.score(['My cat'], ['My cat keeps knocking cups off the table every morning. How do I stop her?'],
              ['Cat knocks cups off the table each morning; asking how to stop it.']))"
```

应该打印出 4 个 1–7 之间的整数分和解码分，`parsed` 为 True。

### 5.3 训练 + 评估

```bash
export GOLD_BASE_URL=... GOLD_MODEL=... GOLD_API_KEY=...   # gold 评委：一个比 Gemma 更强、且不同家族的模型
TRAIN_GPU=1 STEPS=200 bash run_rl.sh
```

- 默认设置：policy `Qwen/Qwen3-1.7B`（关掉 thinking），每步 32 个 prompt × 8 个采样，最长 96 tokens，lr 1e-6，KL 系数 0（TRL 默认），每 25 步存一个 checkpoint。
  其他参数通过 `GRPO_ARGS="--lr 2e-6 --vllm"` 之类传进去。
- 不设 `GOLD_MODEL` 也能跑完，只是没有 gold 那一列。
- **中断后可以重跑同一个命令**：训练完成的组会跳过，评估过的 checkpoint 会跳过（训练到一半中断的那组会从头开始）。

### 5.4 GPU 安排

- judge 一张卡，训练一张卡。1.7B policy 加梯度检查点，在 48 GB 或 80 GB 的卡上可以跑；24 GB 的卡要把 `--micro_batch` 调小，比如 4。
- 如果 judge 原来用两张卡做 tensor parallel，就改成一张卡跑 judge（bf16 约 52 GB，卡不够就换 FP8 版本）。

### 5.5 大概要多久

每步 256 次 judge 调用。按之前测到的 4–11 次/秒，judge 这部分约 25–65 秒一步。
200 步每组约 1.5–4 小时，两组加评估一晚上应该能跑完。想更快就先 `STEPS=100`，或者加 `--vllm` 加速生成。

### 5.6 看进度

```bash
tail -f runs/raw.log         # 每次 reward 调用一行：raw / shad 平均分、平均词数、解析成功率、judge 耗时
```

训练开始前会先下载 policy 模型和 `trl-lib/tldr` 数据（约 120 MB）。

## 6. 跑完把这些发给我

1. `runs/eval.log` 最后的表（或 `runs/eval_summary.csv`）：每组每个 checkpoint 的 proxy_raw、proxy_shad、words、gold。
2. `runs/raw/rewards.jsonl` 和 `runs/shad/rewards.jsonl`：训练过程中每次 reward 调用的两种分数和长度。
3. 每组最后一个 checkpoint 的几条样例摘要（在 `runs/<组>/checkpoint-200/eval_300.json` 的 `summaries` 里）。

## 7. 怎么看结果

| 看到的现象 | 说明 |
|---|---|
| `raw` 组 proxy_raw 一直涨，gold 先升后降，或者 words 一直变长 | 出现了 reward hacking，这是我们需要的前提 |
| `shad` 组 gold 的最高点不低于 `raw` 组，且后期下降得更少 | ShAD 减轻了 hacking，故事成立 |
| `raw` 组的 gold 和 proxy 一路同步上升 | 这个设置下没有 hacking，两组的差别会被噪声盖住；可以试试加大 lr 或步数 |
| `shad` 组生成的摘要开始大段抄原文 | 第 2 节说的风险发生了 |

gold 评委本身也有偏好（比如偏爱长摘要），所以 gold 一定要和 words 一起看。

## 8. 已知限制

- 只测试过 CPU 上的 2 步（用很小的模型和假的 judge），真实 GPU 上没有跑过；显存设置和速度都是估计。
- `Qwen/Qwen3-1.7B` 是默认的 policy，没有专门选过，换模型只需要改 `--model`。
- decoder 只在约 800 条人工打分上拟合，而且拟合用的是 2020 年那批模型的摘要；policy 训练后的输出和它们不一样，decoder 可能外推不准。

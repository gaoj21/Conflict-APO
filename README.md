# WGC baselines：基于 `llm` 包的版本（用于跑 GPT 主表）

这是 "Decode the Shared Axes, Prompt the Content" 主表实验代码的一个自包含版本。所有模型调用都经过仓库根目录的
`llm/` 包（`llm-v12` contract）。**`llm/` 由你的机器提供**，不在本仓库里（`.gitignore` 已排除），模型、provider、
参数、key 和价格都在那里配置好。本仓库里的代码不出现任何 provider 名、模型名、endpoint 或 key。

- judge 和 optimizer 用同一个模型，即 `llm` 的默认 provider（和 Gemma 主表的设定一致）。
- 方法：v0、few-shot、DSPy zero-shot、OPRO（官方代码）、TextGrad CCC / Single-Task（官方 `textgrad`）、
  DSPy MIPROv2 / GEPA（官方 `dspy`）。
- 每个方法都把 judge 在 fit / sel / test 上的打分存下来，`evaluate.py` 用同一套流程报告 raw 和
  readout（decoder）的 test Spearman，外加 isotonic 和 length-debias 两个 post-hoc baseline。

和旧版（OpenRouter / juapi）相比，只有 `baselines/common.py` 里的调用层换了，各方法的 runner、超参数和评测逻辑都没变。

## 目录结构

```
Conflict-APO/
├── llm/                  ← 你的 llm 包（自己放进来，不提交）
├── .env                  ← llm 需要的 key（llm 自己读取，不提交）
├── data/                 4 个数据集，已经切分好
│   └── hs2/ hanna/ ell/ merlin/
│       ├── attrs.txt     属性名（空格分隔）
│       ├── format.txt    固定的输出格式（JSON，每个属性一个整数）
│       ├── v0.txt        初始 rubric prompt
│       ├── gold.csv      人工分数：id, 各属性, z（= log 长度，readout 用）
│       └── fit.jsonl  sel.jsonl  test.jsonl   每行 {"id", "user"}，user 是 judge 要打分的文本
├── baselines/
│   ├── common.py           公共 harness：调用 llm、数据集类、打分解析
│   ├── smoke.py          整条链路的冒烟测试（几次真实调用）
│   ├── run_simple.py     v0, fewshot
│   ├── run_opro.py       OPRO（third_party 里的官方代码）
│   ├── run_textgrad.py   TextGrad ccc | single
│   ├── run_dspy.py       DSPy zeroshot | mipro | gepa
│   ├── run_all.py        调度器：跑完整张表，可断点续跑
│   └── evaluate.py       汇总：raw / isotonic / own_len / readout
├── third_party/opro-main/   OPRO 官方代码（未改动）
└── results/gpt/           输出（自动创建）
```

| 名称 | 数据集 | 属性 | 分数范围 | fit / sel / test |
|---|---|---|---|---|
| `hs2` | HelpSteer2 | helpfulness correctness coherence complexity verbosity | 0–4 | 580 / 579 / 480 |
| `hanna` | HANNA（故事生成） | relevance coherence empathy surprise engagement complexity | 1–5 | 352 / 352 / 352 |
| `ell` | ELLIPSE（英语学习者作文） | cohesion syntax vocabulary phraseology grammar conventions | 1–5 | 400 / 400 / 480 |
| `merlin` | MERLIN（德语学习者作文） | grammar orthography vocab_range vocab_control coherence sociolinguistic | 1–6 | 342 / 342 / 342 |

## 1. 安装

```bash
git clone git@github.com:gaoj21/Conflict-APO.git && cd Conflict-APO
python -m venv .venv && . .venv/bin/activate      # Python 3.12（3.10+ 应该也可以）
pip install -r requirements.txt                   # 另外装你的 llm 包需要的依赖
cp -r /path/to/your/llm ./llm                     # 或 ln -s；再把 llm 需要的 .env 放到仓库根目录
```

## 2. 和 `llm` 的接口

只用 `llm` 的公开 API（`from llm import chat_result, batch_result, UsageTracker, ProviderError`），全部在 `baselines/common.py`：

| 用途 | 调用 |
|---|---|
| judge 给一批条目打分（每次评测一个 prompt 在 fit / sel / test 或验证集上的分数） | `batch_result(None, [messages, ...], temperature=0)`。顺序有保证；并发或原生 batch 由 `llm` 决定 |
| 优化器的单次调用（OPRO 生成指令、TextGrad loss / gradient / optimizer、DSPy） | `chat_result(None, messages, temperature=t)` |
| token 和花费 | 每个 `LLMResult` 都交给同一个 `UsageTracker`，任务结束时写入 `usage.json` |

- **参数**：只传 `temperature`。judge 用 0；优化器用官方温度：TextGrad 的 loss / gradient / optimizer 分别是 0.3 / 0.3 / 0.7，
  OPRO 和 DSPy 的 optimizer 是 1.0。模型、provider、其他参数、重试和价格全部按你的 `llm` 配置。
- **结果目录**：写到 `results/gpt/`。换模型时设 `WGC_RUN=<名字>`，写到别的目录。
- **出错时**：
  - `batch_result` 抛出 `ProviderError` 时，最终评测会直接报错停止，也就是没有结果文件，重跑就会续上；
  - 优化器在给候选 prompt 打分时遇到这个错误，这个候选按最差分计。

## 3. 冒烟测试（必做，约 4 次真实调用）

```bash
set -a; . ./.env; set +a
python llm/examples_terminal.py      # llm 自己的冒烟测试
python baselines/smoke.py hanna      # 整条链路：llm → judge → 分数解析；最后一行应为 OK
```

如果出现 `CHECK`，说明 judge 的输出没有以 JSON 结尾（例如 max_tokens 太小、输出被截断），先检查 llm 的配置。
如果某个任务里超过 20% 的 judge 输出解析失败（API 出错、被拒答等），这个任务会直接停止，不会在坏数据上继续优化；修好后重跑即可续上。

## 4. 先跑小规模 pilot（建议）

```bash
python baselines/run_all.py --ds hanna --methods v0,dspy_zeroshot,fewshot --seeds 0 --par 3 --workers 16
python baselines/evaluate.py hanna
```

## 5. 跑完整主表（4 数据集 × 7 方法 × 3 seeds）

```bash
nohup python baselines/run_all.py --ds hs2,hanna,ell,merlin --seeds 0,1,2 --par 10 --workers 16 > run_all.log 2>&1 &
```

- 方法：v0、dspy_zeroshot（不依赖 seed，只跑 seed 0）；fewshot、opro、textgrad_ccc、textgrad_single、dspy_mipro、
  dspy_gepa（3 个 seed）。每个数据集 2 + 6×3 = 20 个任务，共 80 个。
- `--par` 是同时运行的任务数。每个任务内部 judge 的并发由 `llm` 控制（`client.MAX_PARALLEL`）。`--workers` 只用于
  DSPy 的线程数。被限流时调小 `--par`。
- **进度**：调度器启动、完成或失败一个任务时，各打印一行。另外每隔 `--every` 秒（默认 60）打印一次总进度
  （完成 / 运行 / 排队 / 失败）以及每个运行中任务的已运行时间和日志最后一行，例如 OPRO 的
  `[step ~3/12, instruction #22] score ...`、TextGrad 的 `step 5/12 val MAE ...`、最终评测的
  `judged 352 items in 40s, 100% parsed`。后台运行时用 `tail -f run_all.log` 查看。
- 每个任务的完整日志在 `results/gpt/logs/<ds>_<method>_<seed>.log`。
- **断点续跑**：直接重跑同一条命令，已完成的任务（三个 `scores_*.csv` 都在）和正在运行的任务会被跳过。
- **停止**：`touch results/gpt/STOP` 之后不再启动新任务，正在跑的任务会继续跑完。要立即停止，就 kill 对应进程。
- 只跑一部分时，用 `--methods opro,textgrad_ccc` 和 `--ds hs2` 过滤。

也可以单独跑某个任务：

```bash
python baselines/run_simple.py   hanna v0
python baselines/run_simple.py   hanna fewshot --seed 1
python baselines/run_opro.py     hanna --seed 0            # 12 步
python baselines/run_textgrad.py hanna ccc --seed 0        # 12 步, batch 3, val=MAE (100 条 sel)
python baselines/run_textgrad.py hanna single --seed 0     # 每个属性单独优化，推理时每个属性调用一次
python baselines/run_dspy.py     hanna zeroshot
python baselines/run_dspy.py     hanna mipro --seed 0      # auto="light"
python baselines/run_dspy.py     hanna gepa --seed 0       # auto="light"
```

## 6. 汇总结果

```bash
python baselines/evaluate.py                 # 4 个数据集；也可以写 python baselines/evaluate.py hanna hs2
```

输出 `results/gpt/results.csv`（每个 run 一行），并打印每个数据集和方法的 3-seed 均值：

- `raw`：judge 原始分数的 test Spearman（属性平均）；
- `readout`：经过我们的 decoder 之后的 test Spearman；
- `isotonic`、`own_len`：两个 post-hoc baseline；
- `sd_*`：跨 seed 的标准差；`n`：seed 数。

`evaluate.py` 不调用模型，不需要 key。

## 7. 开销参考

Gemma 4 26B 跑完整张表（含重跑，偏上限）：输入约 **8 亿 token**，输出约 **2.3 亿 token**（关闭 reasoning）。大头是 TextGrad
Single-Task（约 40%）和 OPRO（约 20%）；v0、zero-shot 和 few-shot 很便宜。GPT 的实际消耗看各任务的 `usage.json`。建议跑完第 4 步的
pilot 后先看一下。

## 输出文件

```
results/gpt/
├── logs/                 每个任务的日志
├── results.csv           evaluate.py 汇总
└── <ds>/<method>/seed<s>/
    ├── scores_fit.csv  scores_sel.csv  scores_test.csv    judge 分数（id + 各属性）
    ├── usage.json        这个任务的调用数、token（未回报 usage 的调用单独计数）以及 UsageTracker 的统计和花费
    ├── final_prompt.txt / final_prompts.json / program.json   最终 prompt（或 DSPy program）
    └── 各方法的过程记录（opro/、hist_*.json、gepa_log/ 等）
```

把 `results/gpt/` 整个拷回来，就能并入论文的主表。

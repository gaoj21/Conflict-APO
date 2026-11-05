# WGC baselines：基于 `llm` 包的版本（用于跑 GPT 主表）

这是 "Decode the Shared Axes, Prompt the Content" 主表实验代码的一个自包含版本。所有模型调用都经过仓库根目录的
`llm/` 包（`llm-v12`，说明见 `llm/README.md`）。`llm/` 之外的代码不出现任何 provider 名、endpoint 或 key，用哪个模型
完全由 `llm/providers.json` 决定。换到 SafeChain 机器时，只需要改 `llm/providers.json` 和对应的环境变量，并在
`llm/adapters/` 中放入它的 transport，其余文件一个字节都不用改。

- judge 和 optimizer 用同一个模型（和 Gemma 主表一致）。
- 方法：v0、few-shot、DSPy zero-shot、OPRO（官方代码）、TextGrad CCC / Single-Task（官方 `textgrad`）、
  DSPy MIPROv2 / GEPA（官方 `dspy`）。
- 每个方法都把 judge 在 fit / sel / test 上的打分存下来，`evaluate.py` 用同一套流程报告 raw 和
  readout（decoder）的 test Spearman，外加 isotonic 和 length-debias 两个 post-hoc baseline。

和旧版 `baselines/`（OpenRouter / juapi）相比，只换了调用层（新增 `llm/`，`baselines/gem.py` 改为调用它）。各方法的 runner、超参数和
评测逻辑都没变。旧代码备份在 `../backup/code_2026-11-05_before_llm_contract/`。

## 目录结构

```
wgc_gpt/
├── llm/                  模型边界（llm-v12）：所有模型调用都经过这里
│   ├── providers.json    唯一的配置：provider、模型、参数（不含 secret）
│   ├── client.py registry.py types.py usage.py adapters/   实现（外部只用 `from llm import ...`）
│   ├── examples_terminal.py   provider 冒烟测试（真实调用）
│   └── tests/            单元测试（不调用 provider）
├── .env                  ← provider 的 key（llm 包自己读取），不要提交
├── data/                 ← 4 个数据集，已经切分好，直接可用
│   ├── hs2/  hanna/  ell/  merlin/
│   │   ├── attrs.txt     属性名（空格分隔）
│   │   ├── format.txt    固定的输出格式（JSON，每个属性一个整数）
│   │   ├── v0.txt        初始 rubric prompt
│   │   ├── gold.csv      人工分数：id, 各属性, z（= log 长度，readout 用）
│   │   ├── fit.jsonl     每行 {"id", "user"}；user 是 judge 要打分的文本
│   │   ├── sel.jsonl
│   │   └── test.jsonl
├── baselines/
│   ├── gem.py            公共 harness：llm 调用、缓存、花费日志、数据集类、打分解析
│   ├── smoke.py          冒烟测试（几次真实调用）
│   ├── run_simple.py     v0, fewshot
│   ├── run_opro.py       OPRO（third_party 里的官方代码）
│   ├── run_textgrad.py   TextGrad ccc | single
│   ├── run_dspy.py       DSPy zeroshot | mipro | gepa
│   ├── run_all.py        调度器：跑完整张表，可断点续跑
│   └── evaluate.py       汇总：raw / isotonic / own_len / readout
├── third_party/opro-main/   OPRO 官方代码（未改动）
└── results/<WGC_RUN>/       输出（自动创建）
```

数据集规模（fit / sel / test 条数）：

| 名称 | 数据集 | 属性 | 分数范围 | fit / sel / test |
|---|---|---|---|---|
| `hs2` | HelpSteer2 | helpfulness correctness coherence complexity verbosity | 0–4 | 580 / 579 / 480 |
| `hanna` | HANNA（故事生成） | relevance coherence empathy surprise engagement complexity | 1–5 | 352 / 352 / 352 |
| `ell` | ELLIPSE（英语学习者作文） | cohesion syntax vocabulary phraseology grammar conventions | 1–5 | 400 / 400 / 480 |
| `merlin` | MERLIN（德语学习者作文） | grammar orthography vocab_range vocab_control coherence sociolinguistic | 1–6 | 342 / 342 / 342 |

分数范围以各数据集 `format.txt` 中 "integer from X to Y" 为准，代码也是从那里读的。

## 1. 安装

```bash
cd wgc_gpt
python -m venv .venv && . .venv/bin/activate      # Python 3.12（3.10+ 应该也可以）
pip install -r requirements.txt
```

## 2. 配置 `llm/`

`llm/providers.json` 是唯一的配置，已经写好了以下几个 provider：

| provider | transport | 说明 |
|---|---|---|
| `gpt`（default） | `openai_compatible` | `https://api.openai.com/v1`，`gpt-5.6-luna`，key 取自 `OPENAI_API_KEY`，`reasoning_effort: none` |
| `gpt_batch` | `openai_compatible` + `native_batch` | 同上，但 `batch_result` 走 OpenAI Batch API（更便宜，延迟最长 24h） |
| `deepseek` | `litellm` | 示例，`enabled: false` |
| `safechain` | `adapter: safechain` | SafeChain 机器的占位，`enabled: false`；在那台机器上放入 `llm/adapters/safechain.py` 并启用 |

- 换模型或 endpoint：改 `model` 和 `base_url`。其他 OpenAI 形状的 host 也可以直接用 `openai_compatible`。
- 模型参数（max_tokens、reasoning 等）写在 `params` 里。调用时传入的 sampling 选项会覆盖它；本代码只传 `temperature`。
- 切换 provider 只有两种方式：改 `default`，或者设置环境变量 `LLM_PROVIDER`（`EAX_PROVIDER` 优先）。
- key 放在仓库根目录的 `.env`（例如 `OPENAI_API_KEY=...`）或环境变量里，`providers.json` 里不放 secret。

检查和测试：

```bash
python -m pytest llm/tests                   # 单元测试，不调用 provider
set -a; . ./.env; set +a
python llm/examples_terminal.py              # 列出 provider 是否可用，并做 4 次真实调用，打印内容和 usage
```

**模型参数**：本代码只传 `temperature`：judge 用 0，各优化器用各自的官方温度（TextGrad 的 loss / gradient / optimizer
分别是 0.3 / 0.3 / 0.7，OPRO 和 DSPy 的 optimizer 是 1.0）。其余参数全部由 `providers.json` 的 `params` 决定。
temperature 0 的调用会被缓存。

## 3. 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `WGC_RUN` | 运行名，输出到 `results/<WGC_RUN>/`；**一个模型用一个名字**（例如 `gpt-5.6-luna`） | `gpt` |
| `EVO_TOKEN_INPUT_PRICE_PER_1M` / `EVO_TOKEN_OUTPUT_PRICE_PER_1M` / `EVO_TOKEN_CACHED_INPUT_RATIO` | 每百万 token 价格，由 `llm.UsageTracker` 计价，用于花费日志和预算保护 | 0 / 0 / 1 |
| `WGC_BUDGET` | 整个 run 的记录花费超过这个美元数就停止所有任务（需要先设置价格） | 300 |
| `WGC_BATCH` | `1` = 最终评测（fit/sel/test 全量打分）走 `llm.batch_result`。配合 `LLM_PROVIDER=gpt_batch` 时使用 OpenAI Batch API（更便宜、延迟更高）；配合普通 provider 时是并行调用 | 0 |
| `WGC_MAX_FAIL` | 所有线程合计连续失败这么多次后，停止所有任务 | 30 |

provider 的 key 放在 `.env` 或环境变量里，由 `llm` 包自己读取。本代码不读、也不打印 key。

**安全停止**：一旦遇到认证、封禁、额度类错误（401/402/403、`token_disabled`、quota 等）、缺少环境变量，或者连续失败太多次，
`gem.py` 会立刻 `Abort` 当前任务。之后所有调用都直接失败，不会继续重试。被中止的任务没有结果文件，修好之后重跑
`run_all.py` 就会从这里续上。

## 4. 冒烟测试（必做，约 4 次真实调用）

```bash
set -a; . ./.env; set +a
export WGC_RUN=gpt-5.6-luna
export EVO_TOKEN_INPUT_PRICE_PER_1M=<输入价>  EVO_TOKEN_OUTPUT_PRICE_PER_1M=<输出价>
python baselines/smoke.py hanna
```

`smoke.py` 测的是整条链路（llm → judge → 分数解析）。最后一行应为 `OK`，也就是 judge 输出全部解析成功，并且能看到
token usage。如果出现 `CHECK`，说明 judge 的输出没有以 JSON 结尾，先检查 `providers.json` 的 `params`。

## 5. 先跑小规模 pilot（建议）

先只跑 HANNA 上的不优化方法，确认结果和花费正常：

```bash
python baselines/run_all.py --ds hanna --methods v0,dspy_zeroshot,fewshot --seeds 0 --par 3 --workers 16
python baselines/evaluate.py hanna
python -c "import sys; sys.path.insert(0,'baselines'); import gem; print(gem.spent())"   # ($, 调用数, 输入 token, 输出 token)
```

## 6. 跑完整主表（4 数据集 × 7 方法 × 3 seeds）

```bash
nohup python baselines/run_all.py --ds hs2,hanna,ell,merlin --seeds 0,1,2 --par 10 --workers 16 \
      > results/$WGC_RUN/run_all.log 2>&1 &
```

- 方法：v0、dspy_zeroshot（不依赖 seed，只跑 seed 0）；fewshot、opro、textgrad_ccc、textgrad_single、dspy_mipro、
  dspy_gepa（3 个 seed）。每个数据集 2 + 6×3 = 20 个任务，共 80 个。
- `--par` 是同时运行的任务数。`--workers` 是每个任务内部的并发调用数，总并发约为 par × workers。请按 provider 的
  rate limit 调整，被限流时调小。
- 每个任务的日志在 `results/$WGC_RUN/logs/<ds>_<method>_<seed>.log`。
- **断点续跑**：直接重跑同一条命令。已完成的任务（三个 `scores_*.csv` 都在）和正在运行的任务会被跳过。
  temperature 0 的 judge 调用有缓存（`results/$WGC_RUN/cache.sqlite`），重跑不会重复花钱。
- **停止**：`touch results/$WGC_RUN/STOP` 之后不再启动新任务，正在跑的任务会继续跑完。要立即停止，就 kill 对应进程。
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

## 7. 汇总结果

```bash
python baselines/evaluate.py                 # 4 个数据集；也可以写 python baselines/evaluate.py hanna hs2
```

输出 `results/$WGC_RUN/results.csv`（每个 run 一行），并打印每个数据集和方法的 3-seed 均值：

- `raw`：judge 原始分数的 test Spearman（属性平均）；
- `readout`：经过我们的 decoder 之后的 test Spearman；
- `isotonic`、`own_len`：两个 post-hoc baseline；
- `sd_*`：跨 seed 的标准差；`n`：seed 数。

`evaluate.py` 不调用模型，不需要 key。

## 8. 开销参考

Gemma 4 26B 跑完整张表（含重跑，偏上限），总量约为：

- 输入约 **8 亿 token**，输出约 **2.3 亿 token**（关闭 reasoning）；
- 大头是 TextGrad Single-Task（约 40%）和 OPRO（约 20%），其次是 MIPROv2 和 TextGrad CCC；
- v0、zero-shot 和 few-shot 很便宜。

如果 GPT 的输出更长（例如 llm 里开了 reasoning），输出 token 会明显增加。全程设 `WGC_BATCH=1 LLM_PROVIDER=gpt_batch`，最终评测（每个任务 fit/sel/test 各一个 batch）就能用 batch 价格。优化过程中的
单次调用（`chat_result`）在 `gpt_batch` 下照常是普通实时调用，所以整张表可以直接这样跑，只是每个任务结尾要等 batch 完成。
建议先按第 5 步跑 pilot，用 `gem.spent()` 实测之后再估算全量开销。

## 输出文件

```
results/<WGC_RUN>/
├── cache.sqlite          judge 缓存 + 每次调用的 token / 花费日志（spend 表；未回报 usage 的调用记为 NULL）
├── logs/                 每个任务的日志
├── results.csv           evaluate.py 汇总
└── <ds>/<method>/seed<s>/
    ├── scores_fit.csv  scores_sel.csv  scores_test.csv    judge 分数（id + 各属性）
    ├── final_prompt.txt / final_prompts.json / program.json   最终 prompt（或 DSPy program）
    └── 各方法的过程记录（opro/、hist_*.json、gepa_log/ 等）
```

把 `results/<WGC_RUN>/` 整个拷回来，就能并入论文的主表。

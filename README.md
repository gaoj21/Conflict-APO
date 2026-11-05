# WGC baselines：基于 `llm` 包的版本（用于跑 GPT 主表）

这是 "Decode the Shared Axes, Prompt the Content" 主表实验代码的一个自包含版本。所有模型调用都经过仓库根目录的
`llm/` 包（`llm-v12` contract）。**`llm/` 由你的机器提供**，不在本仓库里（`.gitignore` 已排除），模型、provider、
参数、key 和价格都在那里配置好。本仓库里的代码不出现任何 provider 名、模型名、endpoint 或 key。

- judge 和 optimizer 用同一个模型，即 `llm` 的默认 provider（和 Gemma 主表的设定一致）。
- 方法：v0、few-shot、DSPy zero-shot、OPRO（官方代码）、TextGrad CCC / Single-Task（官方 `textgrad`）、
  DSPy MIPROv2 / GEPA（官方 `dspy`）。
- 每个方法都把 judge 在 fit / sel / test 上的打分存下来，`evaluate.py` 用同一套流程报告 raw 和
  readout（decoder）的 test Spearman，外加 isotonic 和 length-debias 两个 post-hoc baseline。

和旧版（OpenRouter / juapi）相比，只有 `baselines/gem.py` 里的调用层换了，各方法的 runner、超参数和评测逻辑都没变。

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
│   ├── gem.py            公共 harness：调用 llm、缓存、token / 花费日志、数据集类、打分解析
│   ├── smoke.py          整条链路的冒烟测试（几次真实调用）
│   ├── run_simple.py     v0, fewshot
│   ├── run_opro.py       OPRO（third_party 里的官方代码）
│   ├── run_textgrad.py   TextGrad ccc | single
│   ├── run_dspy.py       DSPy zeroshot | mipro | gepa
│   ├── run_all.py        调度器：跑完整张表，可断点续跑
│   └── evaluate.py       汇总：raw / isotonic / own_len / readout
├── third_party/opro-main/   OPRO 官方代码（未改动）
└── results/<run>/           输出（自动创建；<run> 默认是 llm 默认 provider 的名字）
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

本代码只用 `llm` 的公开 API：

- 单次调用用 `chat_result(None, messages, temperature=...)`；最终评测在 `WGC_BATCH=1` 时用 `batch_result(None, items, temperature=0)`。
- 读取 `LLMResult.content` 和 `LLMResult.usage`。
- 每次调用的花费由 `UsageTracker` 计算，价格用你在 llm 里配好的。
- 默认的结果目录名取自 `default_provider()`。

第一个参数始终是 `None`，也就是你配置的默认 provider。要换 provider，就改 `providers.json` 的 `default`，或设置 `LLM_PROVIDER`。

只传 `temperature`：judge 用 0，各优化器用官方温度（TextGrad 的 loss / gradient / optimizer 分别是 0.3 / 0.3 / 0.7，
OPRO 和 DSPy 的 optimizer 是 1.0）。其余参数全部按 llm 的配置。temperature 0 的调用会被缓存，重跑不花钱。

可选环境变量（都有默认值，一般不用设）：

| 变量 | 作用 | 默认 |
|---|---|---|
| `WGC_RUN` | 结果目录 `results/<WGC_RUN>/`，换模型时用来区分 | llm 默认 provider 的名字 |
| `WGC_BUDGET` | 整个 run 的花费（按 `UsageTracker` 计）超过这个美元数，就停止所有任务 | 300 |
| `WGC_BATCH` | `1` = 最终评测（fit/sel/test 全量打分）走 `batch_result`，provider 有原生 batch 时更便宜 | 0 |
| `WGC_MAX_FAIL` | 所有线程合计连续失败这么多次，就停止所有任务 | 30 |

**安全停止**：遇到认证、封禁、额度类错误（401/402/403、`token_disabled`、quota 等）、缺少环境变量，或连续失败太多次，
当前任务会立刻 `Abort`，之后不再发出任何调用。被中止的任务没有结果文件，修好后重跑 `run_all.py` 就会续上。

## 3. 冒烟测试（必做，约 4 次真实调用）

```bash
set -a; . ./.env; set +a
python llm/examples_terminal.py      # llm 自己的冒烟测试
python baselines/smoke.py hanna      # 整条链路：llm → judge → 分数解析；最后一行应为 OK
```

如果出现 `CHECK`，说明 judge 的输出没有以 JSON 结尾（例如 max_tokens 太小、输出被截断），先检查 llm 的配置。

## 4. 先跑小规模 pilot（建议）

```bash
python baselines/run_all.py --ds hanna --methods v0,dspy_zeroshot,fewshot --seeds 0 --par 3 --workers 16
python baselines/evaluate.py hanna
python -c "import sys; sys.path.insert(0,'baselines'); import gem; print(gem.spent())"   # ($, 调用数, 输入 token, 输出 token)
```

## 5. 跑完整主表（4 数据集 × 7 方法 × 3 seeds）

```bash
nohup python baselines/run_all.py --ds hs2,hanna,ell,merlin --seeds 0,1,2 --par 10 --workers 16 > run_all.log 2>&1 &
```

- 方法：v0、dspy_zeroshot（不依赖 seed，只跑 seed 0）；fewshot、opro、textgrad_ccc、textgrad_single、dspy_mipro、
  dspy_gepa（3 个 seed）。每个数据集 2 + 6×3 = 20 个任务，共 80 个。
- `--par` 是同时运行的任务数。`--workers` 是每个任务内部的并发调用数，总并发约为 par × workers。请按 provider 的
  rate limit 调整，被限流时调小。
- 每个任务的日志在 `results/<run>/logs/<ds>_<method>_<seed>.log`。
- **断点续跑**：直接重跑同一条命令。已完成的任务（三个 `scores_*.csv` 都在）和正在运行的任务会被跳过。
  temperature 0 的 judge 调用有缓存（`results/<run>/cache.sqlite`），重跑不会重复花钱。
- **停止**：`touch results/<run>/STOP` 之后不再启动新任务，正在跑的任务会继续跑完。要立即停止，就 kill 对应进程。
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

输出 `results/<run>/results.csv`（每个 run 一行），并打印每个数据集和方法的 3-seed 均值：

- `raw`：judge 原始分数的 test Spearman（属性平均）；
- `readout`：经过我们的 decoder 之后的 test Spearman；
- `isotonic`、`own_len`：两个 post-hoc baseline；
- `sd_*`：跨 seed 的标准差；`n`：seed 数。

`evaluate.py` 不调用模型，不需要 key。

## 7. 开销参考

Gemma 4 26B 跑完整张表（含重跑，偏上限），总量约为：

- 输入约 **8 亿 token**，输出约 **2.3 亿 token**（关闭 reasoning）；
- 大头是 TextGrad Single-Task（约 40%）和 OPRO（约 20%），其次是 MIPROv2 和 TextGrad CCC；
- v0、zero-shot 和 few-shot 很便宜。

GPT 的实际消耗取决于 llm 里的模型设置（例如是否开 reasoning）。建议先按第 4 步跑 pilot，用 `gem.spent()` 看实际
token 和花费，再估算全量开销。如果你的 provider 有原生 batch endpoint，加 `WGC_BATCH=1` 后，最终评测会走
`llm.batch_result`。

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

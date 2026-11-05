# Handover：用 Codex（GPT）跑 WGC 主表 baseline

这份文档写给在**本地机器**（Mac 或 Linux）上接手运行的人。读完就能把 GPT 版主表跑起来：模型通过 Codex CLI 调用，也可以换成任何 OpenAI 兼容的 API。

## 0. 这是什么

论文 *Decode the Shared Axes, Prompt the Content* 的主表比较了一个多属性 LLM judge 在 7 种 prompt 方法下的效果，每种方法看两个分数：

- raw：judge 原始分数与人工分数的 test Spearman；
- readout：经过我们的 decoder（per-attribute anchored ridge readout）之后的 test Spearman。

数据集 4 个，seed 3 个。Gemma 4 26B 版已经跑完。这里要跑 **GPT 版**：judge 和 optimizer 都用同一个 GPT 模型。

| 方法 | 代码 | 是否用 seed |
|---|---|---|
| v0（初始 rubric） | `run_simple.py v0` | 只跑 seed 0 |
| DSPy zero-shot | `run_dspy.py zeroshot` | 只跑 seed 0 |
| few-shot（5 个带人工分的例子） | `run_simple.py fewshot` | 0,1,2 |
| OPRO（官方代码，12 步） | `run_opro.py` | 0,1,2 |
| TextGrad CCC（官方 textgrad，12 步） | `run_textgrad.py ccc` | 0,1,2 |
| TextGrad Single-Task（每个属性一个 prompt） | `run_textgrad.py single` | 0,1,2 |
| DSPy MIPROv2 / GEPA（官方 dspy，auto=light） | `run_dspy.py mipro` / `gepa` | 0,1,2 |

数据集：`hs2`（HelpSteer2）、`hanna`（故事生成）、`ell`（ELLIPSE 英语作文）、`merlin`（MERLIN 德语作文）。每个数据集 20 个任务，共 80 个。

## 1. 项目结构

```
Conflict-APO/
├── HANDOVER.md            本文件
├── README.md              详细说明（方法、参数、输出格式）
├── requirements.txt
├── data/<ds>/             数据集，已切分好：attrs.txt format.txt v0.txt gold.csv fit/sel/test.jsonl
├── baselines/
│   ├── common.py          公共代码：调用 llm、读数据、judge 打分、解析分数、token / 花费统计
│   ├── smoke.py           整条链路的冒烟测试
│   ├── run_all.py         调度器：跑全部任务，显示进度、token 和花费，可断点续跑
│   ├── run_simple.py run_opro.py run_textgrad.py run_dspy.py   各方法
│   ├── evaluate.py        汇总结果表（不调用模型）
│   └── usage.py           查看每个任务的 token 和花费
├── llm_codex/             一个完整的 llm 包（llm-v12 接口），默认 provider = codex
│   ├── providers.json     provider 配置
│   ├── adapters/codex.py  通过 `codex exec` 调 GPT
│   ├── adapters/openai_compatible.py  通过 OpenAI 兼容 API 调用（备选）
│   ├── examples_terminal.py   llm 自己的冒烟测试
│   └── tests/             单元测试（不调用模型）
├── third_party/opro-main/ OPRO 官方代码（未改动）
└── results/<WGC_RUN>/     输出（运行后生成）
```

**模型调用只有一个入口**：`baselines/common.py` 通过 `from llm import chat_result, batch_result, UsageTracker` 调用模型。用哪个模型完全由 `llm/providers.json` 决定，实验代码不用改。

## 2. 安装

```bash
git clone git@github.com:gaoj21/Conflict-APO.git && cd Conflict-APO
python3 -m venv .venv && . .venv/bin/activate          # Python 3.10+（测试用的是 3.12）
pip install -r requirements.txt
pip install pytest                                       # 可选，跑 llm 单元测试用

# llm 包：用自带的 codex 版（已经有自己的 llm 包的话，见第 8 节）
cp -r llm_codex llm

# Codex CLI（需要 Node.js）
npm i -g @openai/codex
codex login                                              # ChatGPT 账号登录，或用 API key
```

## 3. 配置 `llm/providers.json`

默认配置如下：

```json
"codex": {
  "adapter": "codex",
  "codex_config": {"model_reasoning_effort": "low"},
  "params": {"timeout": 300}
}
```

- `"model": "gpt-5.6-luna"`：已经设好（与之前的 GPT 实验一致）。不写的话会用 `~/.codex/config.toml` 里的默认模型。
- `codex_config`：每一项都会变成 `codex exec -c key=value`。`model_reasoning_effort` 控制推理强度；之前的 GPT 和 Gemma 实验都关闭了推理，这里设成该模型支持的最低档即可。
- `codex_bin`：`codex` 不在 PATH 里时，填可执行文件的路径（也可以用环境变量 `CODEX_BIN`）。
- 如果改用 OpenAI 官方 API，把 `default` 改成 `"gpt"`，在 `.env` 里写 `OPENAI_API_KEY=...`。这条路按 token 计费，支持 temperature，跑全量更合适。

**codex adapter 的行为：**

- 每次调用启动一个 `codex exec --json` 进程，工作目录是空的临时目录，沙盒是 read-only，加 `--ephemeral`，不保存会话。
- 我们的 system prompt（judge 的 rubric 或优化器的提示词）通过 `model_instructions_file` 替换掉 Codex 默认的 coding-agent 提示。user 消息作为 prompt 传入。
- Codex 自己还会加一小段 sandbox / skills / 环境说明，每次约 700 token，无法去掉。
- 回复取最后一条 `agent_message`，token 用量从 `turn.completed` 读取。
- **temperature 等采样参数传不进 Codex**，会被忽略。所以 judge 不是确定性的 temperature 0，优化器的官方温度也用不上。这一点要写进论文。

## 4. 冒烟测试（必做）

```bash
. .venv/bin/activate
export WGC_RUN=codex                 # 结果写到 results/codex/；换模型时换个名字
export LLM_MAX_PARALLEL=8            # 同时运行的 codex 进程数，按机器性能和账号限额调

python -m pytest -q llm/tests        # llm 单元测试，不调用模型
python llm/examples_terminal.py      # 4 次真实调用，打印回复和 usage
python baselines/smoke.py hanna      # 整条链路，最后一行应为 OK
```

`smoke.py` 最后一行不是 `OK` 时，说明 judge 的输出没能解析成 JSON 分数。先单独跑一次 `codex exec`，看看模型实际回了什么。

## 5. 先跑 pilot

```bash
python baselines/run_all.py --ds hanna --methods v0,dspy_zeroshot,fewshot --seeds 0 --par 3 --every 30
python baselines/usage.py
python baselines/evaluate.py hanna
```

**判断标准**：每个方法的 `readout` 应该高于 `raw`。Gemma 在 hanna 上大约是 raw .37 → readout .47。

**规模参考**：每个 v0 / few-shot / zero-shot 任务要对 fit+sel+test 全部打分，各数据集分别是 hs2 1639 次、hanna 1056 次、ell 1280 次、merlin 1026 次调用。离线测试中，16 个并发 codex 进程跑完 1056 次大约 1 分钟；真实模型取决于响应速度。

## 6. 跑全量

```bash
nohup python baselines/run_all.py --ds hs2,hanna,ell,merlin --seeds 0,1,2 --par 4 > run_all.log 2>&1 &
tail -f run_all.log
```

- **规模**：80 个任务。Gemma 版大约 60 万次调用、8 亿输入 token。大头是 TextGrad Single-Task 和 OPRO，单个任务就要 1–2 万次调用。**用 ChatGPT 订阅登录的 Codex 有用量上限，很可能跑不完全量**，建议按方法分批跑：
  1. `--methods v0,dspy_zeroshot,fewshot`（最便宜）
  2. `--methods textgrad_ccc,dspy_mipro,dspy_gepa`
  3. `--methods opro,textgrad_single`（最贵）
- **并发**：总并发约为 `--par` 个任务 × `LLM_MAX_PARALLEL` 个 codex 进程。每个进程都要占内存，Mac 上建议先从 `--par 4`、`LLM_MAX_PARALLEL=8` 开始。
- **进度**：每 60 秒打印一次（`--every` 可以改），包括完成 / 运行 / 排队 / 失败的任务数、每个运行中任务的 token 和花费，以及整个 run 的累计。
- **花费**：
  - Codex 用订阅登录时没有按 token 的价格，显示 `$0.00`，但 token 数照常统计。
  - 想看等价的 API 花费，设置 `EVO_TOKEN_INPUT_PRICE_PER_1M` 和 `EVO_TOKEN_OUTPUT_PRICE_PER_1M`。
  - 任何时候都可以用 `python baselines/usage.py` 看每个任务的消耗和总计。
- **断点续跑**：直接重跑同一条命令。已经完成的任务（三个 `scores_*.csv` 都在）和正在运行的任务会被跳过。
- **停止**：`touch results/codex/STOP` 之后不再启动新任务；要立即停止就 kill 进程。
- **失败**：调度器打印 `FAILED <任务> (exit …)` 和日志最后一行。完整日志在 `results/codex/logs/<任务>.log`。常见原因：
  - **用量上限**（`usage limit` / `429`）：这类错误不会自动重试。等额度恢复后重跑同一条命令即可续上。
  - **一批 judge 调用里有一个出错**：这一批整批失败，对应任务停止，重跑即可。
  - **超过 20% 的 judge 输出解析失败**：任务主动停止，避免在坏数据上继续优化。

## 7. 交回什么

全部跑完（或者跑完一批）后：

```bash
python baselines/evaluate.py          # 生成 results/codex/results.csv 并打印汇总表
python baselines/usage.py > results/codex/usage_summary.txt
tar czf codex_results.tgz results/codex
```

把 `codex_results.tgz` 发回来。里面包括：

- 每个任务的三份分数文件；
- 最终 prompt；
- `usage.json`；
- 日志；
- 汇总表。

另外请注明用的是哪个模型（`providers.json` 里的 `model`，或 Codex 的默认模型）、`model_reasoning_effort`，以及 Codex 的版本（`codex --version`）。

## 8. 已经有自己的 `llm` 包时

如果本地已经有 llm-v12 的 `llm/` 包，比如另一台机器上的 DeepSeek 或 SafeChain 版，有两种做法：

1. **直接用它**：放到仓库根目录，叫 `llm/`，配置好它的默认 provider，然后从第 4 步开始。实验代码只用公开 API（`chat_result`、`batch_result`、`UsageTracker`、`ProviderError`）。
2. **只加 codex 支持**：
   - 把 `llm_codex/adapters/codex.py` 复制到你的 `llm/adapters/`；
   - 在 `providers.json` 里加 `"codex": {"adapter": "codex", ...}`。

   前提是你的包里 transport 的接口和这里一致：
   - transport 是一个模块，提供 `complete(provider, messages, options) -> LLMResult`；
   - 失败时抛 `ProviderError(msg, status, retryable)`；
   - provider 的额外字段放在 `provider.extra` 里。

   接口不一样的话，照着 `codex.py` 改一下即可，大约 100 行。

## 9. 注意事项

- **不要用 juapi 跑这个实验**。2026-11-04 到 05 期间，juapi 的 key 被停用了 10 次，最后一次是在启动后 20 秒内，就在 TextGrad 优化调用开始的时候。
- **Codex 不是干净的 API 调用**：它带有自己的少量 agent 上下文，也无法设置 temperature。论文里把它作为 "GPT via Codex CLI" 写清楚设置，或者只放在附录作为换一个 judge 的佐证。正式的 GPT 主表最好用官方 API（`gpt` provider）跑。
- **Codex 会附带工具定义**：我们在指令里要求模型不调用工具，沙盒也是 read-only。如果模型仍然调用了工具，那次调用会变成多轮，我们只取最后一条回复。日志里的 token 数会偏高，但不影响分数。
- **HelpSteer2 的内容**：数据里少量 prompt 涉及敏感话题，judge 只做打分，正常情况下不会出问题。如果 hs2 上频繁出现拒答（解析失败），先跑其他 3 个数据集。

## 10. 已验证

在 Linux 上，codex-cli 0.166.0，接本地模拟的 OpenAI Responses 服务器：

- `llm` 单元测试 12 个全部通过；
- 用真实 codex 可执行文件跑通 `smoke.py`、v0（1056 次调用）、TextGrad CCC、DSPy zero-shot、OPRO 和 `evaluate.py`；
- usage 解析正确，rubric 确实作为 developer 指令发出。

**还没有用真实的 GPT 模型测过**。第一次运行请严格按第 4、5 步来。

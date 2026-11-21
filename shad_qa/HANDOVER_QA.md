# ShAD-QA 交接文档（本地 GPU 运行）

代码：<https://github.com/gaoj21/Conflict-APO/tree/shad-qa/shad_qa>（分支 `shad-qa`，目录 `shad_qa/`）
实验细节见同目录的 [README_QA.md](README_QA.md)，这份文档只讲：在做什么、怎么在本地 GPU 上跑、现在结果到哪一步。

## 1. 在做什么

评分任务（HS2 / HANNA / ELL / MERLIN）上，ShAD 规则平均比 TextGrad 高 8 个 Spearman 点。那里优化的是 judge 的分数。
QA 这条线要让 ShAD 改进的是**生成器的答案 A**，不是 judge 分数：

1. 生成器（Gemma 4 26B A4B）按当前 system prompt 回答问题；
2. judge 在若干评价维度上给 A 和参考答案打分；
3. ShAD decoder：参考答案的每个维度 ~ A 的全部维度 + log 长度（ridge），得到 A 偏离参考答案的方向；
4. 用 L1 logistic 回归估计每种偏离让答错概率涨多少，NMF 合成共享轴，按 代价 × 占比 排序；
5. 排名靠前的轴翻译成一句话规则，加进生成器的 prompt；
6. 全部建在真实 APO（TextGrad 官方包）上：先跑 TextGrad，再在它的最终 prompt 上加规则。

评价维度两种来源：
- **固定维度**（`qa/dimshad.py`）：6 个预先给定的维度（verbosity / specificity / granularity / multiplicity / qualification / type_fit）；
- **学出来的维度**（`qa/dimlearn.py`）：LLM 看 300 个池子答案，提出 24 个候选维度，再由数据选出有用的（这样不用人手写维度，也不会过于具体）。

## 2. 数据

全部是公开数据集，已经打包在仓库里（`data/qa/*.jsonl.gz`，36 MB），不需要再下载：

| 数据集 | 说明 |
|---|---|
| DROP | 官方 EM，多 span 答案来自官方 json（已存进 `answers` 字段） |
| MuSiQue | 多跳问答，输入约 2.8k tokens |
| HotpotQA | distractor 设定 |

划分：fit 600 = train 0–599；pool 1000 = train 1100–2099；**val = dev 1000–1499**（`QA_VAL=dev`，默认）；**test = dev 0–999**。

## 3. 本地 GPU 环境

```bash
git clone -b shad-qa git@github.com:gaoj21/Conflict-APO.git
cd Conflict-APO/shad_qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_qa.txt
pip install vllm          # 另开一个环境装也可以，只要服务器能起来
```

### 启动 vLLM

```bash
vllm serve <Gemma 4 26B A4B instruct 在 HF 上的名字> \
  --served-model-name google/gemma-4-26b-a4b-it \
  --max-model-len 16384 --gpu-memory-utilization 0.9 --port 8000
```

- `--served-model-name` 要和代码里的默认模型名 `google/gemma-4-26b-a4b-it` 一致（或者设 `GEM_MODEL` 成你起的名字）。HF 上的确切仓库名请在 HF 上确认。
- 显存：bf16 权重约 52 GB，单卡 80 GB（A100 / H100）可以跑；显存不够用 FP8 版本，或加 `--tensor-parallel-size 2`。
- 之前的实验关掉了推理（thinking），本地也要关：如果该模型的 chat template 有 thinking 开关，设
  `export GEM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'`。

## 4. 运行

```bash
export GEM_PROVIDER=local
export GEM_BASE_URL=http://localhost:8000/v1
export GEM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'   # 见上
WORKERS=64 PY=python bash run_qa_llm.sh musique      # 或 drop / hotpot，不写就三个都跑
```

- 第一步是 preflight：发一个真实请求，打印 `test call OK`，失败会直接退出并报原因。
- 本地没有费用，`GEM_BUDGET` 不用管。
- `WORKERS` 是并发数。vLLM 本地可以开到 64 甚至更多，先看 GPU 占用再调。
- **可以断点续跑**：中断后重跑同一个命令，已经完成的调用都在 sqlite 缓存里，不会重复。

### 一个数据集里的执行顺序

```
TextGrad（12 步）
 ├─ 并行：TextGrad × ShAD edit（每 3 步插一次 edit 规则）
 └─ 串行：ShAD edit（一步）→ error-summary 对照
          → 固定维度：dimshad → 1 条规则 → 多条叠加
          → 学出来的维度：dimlearn → 多条叠加
最后：qa_report.py 汇总成表
```

### 输出在哪里

| 内容 | 路径 |
|---|---|
| 结果 | `results/gemma_local_qa/<数据集>/shadqa_devval/`（DROP 是 `shadqa_official_devval`） |
| 每一步的日志 | `results/gemma_local/logs/qa/<数据集>_<步骤>.log` |
| 最终表 | `results/gemma_local/logs/qa/final_report.txt` |
| 调用缓存 | `results/gemma_local/cache.sqlite` |

脚本只在每一步**跑完**时打印一行，TextGrad 那一步很长，中间看起来像卡住。看进度：

```bash
tail -f results/gemma_local/logs/qa/musique_textgrad.log
```

开头是 `[tg] start val EM ...`（验证集 500 题跑完才出），之后每步一行 `[tg] step N: ...`。

## 5. 现在的结果

### MuSiQue（OpenRouter 上的 Gemma，旧划分：val 来自 train 600–1099）

TextGrad 是绝对值，其余是相对 TextGrad 的差：

| 方法 | val（用来选） | 新的 train 题 | test（dev） |
|---|---|---|---|
| TextGrad | 65.8 | 67.2 | 61.7 |
| + error summary | +2.6 | −1.0 | −1.0 |
| + edit 特征 | +1.4 | −2.6 | −1.1 |
| + 1 条维度规则 | +2.8 | +1.0 | −0.2 |
| + 3 条维度规则叠加 | +4.2 | +1.2 | −1.4 |

诊断：在 val 上有提升，到 test 上消失。原因是在 val 上过拟合，加上 train → dev 的分布偏移。
所以现在把 val 改成从 dev 划（dev 1000–1499，和 test 不重叠），**这是本地要跑的新设计，还没有结果**。

学出来的维度在 MuSiQue 上自己找回了 specificity（具体实体）、temporal precision（时间精度）、direct answer（直接回答）这几个方向，和固定维度一致。

### DROP（官方 EM，旧划分）

| 方法 | test EM |
|---|---|
| TextGrad | 87.8 |
| + edit 规则 | 85.6 |
| + error-summary 规则 | 87.1 |

之前的"提升"是 HF 版本把多 span 答案拍平导致的评分 bug，改成官方答案后不存在。

## 6. 已知问题

- **公司电脑上卡住**：`llm` 包（SafeChain）的调用没有超时，一个请求挂住就会卡死整个步骤。已经修复（`baselines/gem.py`：每次调用最多 180 秒，`GEM_TIMEOUT` 可调；每分钟打印一行 `[llm] N calls ...`），但还没在公司电脑上验证过。本地 vLLM 走的是另一条路径（HTTP，240 秒超时），不受影响。
- **TextGrad 的不确定性**：TextGrad 内部用 set 存 gradient，顺序每次不同；代码里已排序，`tg+edit` 在插入第一条规则前和 `tg` 完全一致。
- **换后端后缓存不共享**：本地 vLLM 的结果和 OpenRouter 的数值会有小差异（精度、采样实现），新结果写到 `results/gemma_local*`，不会和旧结果混在一起。
- `dimlearn` 如果模型一个维度都没提出来会提前退出，看 `<数据集>_shad_learned_dims.log`。

## 7. 下一步

1. 本地先跑 MuSiQue（新划分），看 test 上维度规则是否仍然 ≥ TextGrad；
2. 再跑 DROP、HotpotQA；
3. 如果新划分下 val 的提升仍然传不到 test，考虑在 TextGrad 过程中就用维度规则（`tg+shad` 这一臂已有代码），而不是只加在最终 prompt 上。

# ShAD on QA through your `llm` package (DROP, MuSiQue, HotpotQA)

This folder (branch `shad-qa` of Conflict-APO) runs two ShAD experiments on three QA datasets with your own model:

1. **ShAD alternating with TextGrad** (`tg+edit`): ShAD rules are added round by round, every 3 TextGrad steps.
2. **ShAD added once at the end** (`+ ShAD edit (1 step)`): ShAD rules are added once to TextGrad's final prompt.
   It is compared with an error-summary control (`+ error summary`) built under the same conditions.

All model calls go through the project's `llm` package (llm-v12: `from llm import chat_result`). Which model and
endpoint are used is decided only by `llm/providers.json`; the experiment code never names a provider. The model under
test is also the optimizer (TextGrad's loss, gradient and update calls) and the error-summary writer. No embeddings and
no other external service are used. The data are in `data/qa/` (public datasets, already sampled), so no HuggingFace
access is needed.

---

## 1. Install

```bash
python3 -m venv .venv && . .venv/bin/activate        # Python 3.10+
pip install -r requirements_qa.txt                   # numpy pandas scipy scikit-learn textgrad requests
```

## 2. Put your `llm` package next to this file

Run everything from this folder (`shad_qa/`). Either copy your own package here:

```bash
cp -r /path/to/your/llm ./llm          # or: export GEM_LLM_PATH=<folder that contains llm/>
```

or start from the template in `llm_template/` (the llm-v12 package used before, same as in `shad_llm/`) and edit
`llm/providers.json` (the default provider, or pick one with `GEM_LLM_PROVIDER`):

```bash
cp -r llm_template llm   # then edit llm/providers.json; keys come from the environment variables it names
```

Check one call: `python -c "import llm; print(llm.chat(None, [{'role':'user','content':'Say OK'}]))"`.

## 3. Run

```bash
export GEM_LLM_PROVIDER=<provider name in providers.json>   # optional; default provider otherwise
export WORKERS=16                                          # concurrent requests per process
bash run_qa_llm.sh                                         # drop musique hotpot, in parallel
bash run_qa_llm.sh musique                                 # one dataset
bash run_qa_llm.sh drop hotpot                             # a subset
```

- **Preflight.** The script first imports the `llm` package and makes one test call. If either fails, it prints the
  error and stops before anything else runs.
- **Concurrency.** Up to 2 processes run per dataset at the same time, each with `WORKERS` threads. Three datasets
  with `WORKERS=16` means up to about 96 requests in flight. If your endpoint rate-limits, lower `WORKERS`.
- **Resuming.** Every call at temperature 0 and every TextGrad call is cached in `$GEM_OUT/cache.sqlite`. Every
  finished answer set is saved as a CSV, and TextGrad's state is saved after every step. After an interruption, rerun
  the same command and it continues where it stopped; finished work costs nothing.
- **Progress.** Each script writes its own log under `$GEM_OUT/logs/qa/`:

  | log | what it shows |
  |---|---|
  | `<ds>_textgrad.log` | one line per TextGrad step: candidate val EM, keep / revert, current val EM |
  | `<ds>_textgrad_x_shad.log` | the same for the alternating arm, plus every ShAD candidate with its val EM on steps 3, 6, 9 and 12 |
  | `<ds>_shad_edit.log` | the edit-feature table, the n-gram conditions, every candidate rule with its val EM, the test result |
  | `<ds>_errsum.log` | every error-summary rule with its val EM, the test result, the edit rule vs the error-summary rule |
  | `final_report.txt` | the final table (see section 7) |

  Example: `tail -f results/llm_<label>/logs/qa/musique_textgrad.log`. `python qa/qa_report.py` prints the table at
  any time from what has finished so far.
- **Environment variables.**
  - `GEM_LLM_PROVIDER`: the provider name in `providers.json`.
  - `GEM_LLM_PATH`: the folder that contains `llm/`.
  - `GEM_MODEL`: a label for the logs and the cache (default: the provider name).
  - `GEM_OUT`: the results folder (default `results/llm_<label>`). QA results go to `${GEM_OUT}_qa/`.
  - `WORKERS`, and `PY` (the python executable).

### Order of work per dataset

```
TextGrad (tg, 12 steps)  ──►  test
        │
        ├──►  TextGrad x ShAD edit (tg+edit, 12 steps; ShAD every 3)  ──►  test            [experiment 1]
        │
        └──►  ShAD edit, 1 step on tg's final prompt  ──►  test
                 └──►  error-summary control, 1 step on tg's final prompt  ──►  test       [experiment 2]
                        └──►  ShAD dims: fixed dimensions, 1 rule and stacked  ──►  test     [experiment 3]
                               └──►  ShAD learned dims: stacked  ──►  test                   [experiment 3]
```

`tg+edit` starts after `tg` has finished, so that it can replay `tg`'s cached TextGrad calls (see 5.3).

---

## 4. Common setup (all arms)

### 4.1 Data

| dataset | content | official answer normalization | source |
|---|---|---|---|
| DROP | sports / history passages, numbers, dates, spans | official DROP EM: per token lowercase, numbers as floats, punctuation and articles removed; gold = the original and every validated answer of the official release; a multi-span answer counts only if all its spans are given (any order) | `ucinlp/drop` + official answers |
| MuSiQue | 2–4-hop questions, 20 Wikipedia paragraphs (answerable only) | SQuAD: lowercase, punctuation and articles removed; aliases accepted | `dgslibisey/MuSiQue` |
| HotpotQA | 2-hop questions, distractor setting (2 gold + 8 distractor paragraphs) | SQuAD (as the official script) | `hotpotqa/hotpot_qa` |

Every split is a fixed seed-0 permutation (stored in `data/qa/<ds>_{train,test}.jsonl.gz`). The val set is drawn from the
same split as the test (`QA_VAL=dev`, the default of `run_qa_llm.sh`): with val from the train split, rules chosen on
val gained on fresh train questions but not on the dev split. `QA_VAL=train` gives the earlier design.

| split | size | from | used for |
|---|---|---|---|
| fit | 600 | train, positions 0–599 | TextGrad batches (3 questions per step) |
| val | 500 | official dev / validation set, positions 1000–1499 (disjoint from test) | every keep / revert decision and every choice of rule |
| pool | 1000 | train, positions 1100–2099 | the answers ShAD and the error summary learn from |
| test | 1000 | official dev / validation set, positions 0–999 | the final numbers only; never used for any choice |

**Metric.** Exact match (EM, x100) of the last `Answer:` line after the dataset's normalization. A reply without an
`Answer:` line is wrong. F1 is also saved.

### 4.2 Prompt

- **System message.** The optimized system prompt. It starts as `Think step by step.`, and only TextGrad edits it.
- **User message.**

  ```
  ## Passage            (## Paragraphs for MuSiQue / HotpotQA)
  <context>

  ## Question
  <question>

  Reason as usual. The following rules apply only to how you write the final Answer line:     <- only if rules exist
  - <rule 1>
  - <rule 2> ...

  End with exactly one line: Answer: <the answer>
  ```

  The rule block and the answer format sit in the user message, outside the system prompt, so TextGrad cannot rewrite
  or delete the ShAD rules. The scope line keeps the rules about the answer line only; the reasoning is left alone.
- **Settings.** The model answers at temperature 0 with at most 3000 output tokens.

### 4.3 TextGrad (arm `tg`, the APO method both experiments build on)

This is the official `textgrad` package (0.1.8), with the same loop as the rating-task baselines:

1. Each step samples a batch of 3 fit questions; the same batches are used in every run.
2. The forward pass is the model with the current system prompt.
3. The loss is `tg.loss.MultiFieldEvaluation`, at temperature 0.3. It sees the question, the reference and the
   response, and criticizes where the final answer deviates: what was misread, computed wrongly, or written
   differently.
4. `backward` runs at temperature 0.3, then a `TGD` step at temperature 0.7. The TGD constraint is "do not write an
   output-format section".
5. The new system prompt is evaluated on val and **kept if val EM does not drop**; otherwise it is reverted.
6. There are 12 steps. The final system prompt is evaluated on test.

The state is saved in `textgrad/tg.json`: the system prompt, the rule block, and a log per step.

---

## 5. Experiment 1 — ShAD alternating with TextGrad (`qa/drop_textgrad.py tg+edit`)

**Question.** Does adding ShAD rules round by round during APO beat APO alone?

### 5.1 Loop

For steps 1 to 12:

1. **TextGrad step.** Exactly as in `tg` (4.3), on the current system prompt **with the current rule block** in the
   user message. Kept if val EM does not drop.
2. **ShAD step,** on steps 3, 6, 9 and 12 only:
   - answer the 1000 pool questions with the current system prompt and rule block;
   - turn these answers into the edit-feature model and its rules (5.2) — up to 6 candidates;
   - drop the candidates that are already in the block or were rejected before;
   - test each remaining candidate **alone**: the current prompt plus the block plus this one rule, on val;
   - append the best candidate to the rule block only if its val EM is not below the current val EM. All the others
     go on a "rejected" list and are never proposed again.

At the end, the final system prompt and the rule block are evaluated on test.

### 5.2 ShAD edit model (`qa/editfeat.py`, no LLM anywhere in this step)

ShAD's idea on QA: features measured on the **output side** (the model's answer) against the same features on the
**ground-truth side** (the reference answer). A small model then says which deviations make answers wrong, and where
they occur. Its weights become the rules.

**(a) Features: what the answer adds or drops compared with the reference.** Answer and reference are split into
tokens, and each token gets a class:

| class | tokens |
|---|---|
| `pct` | `%` |
| `dollar` | `$` |
| `num` | a number |
| `commanum` | a number with a thousands comma |
| `unit` | the word right after a number (`yards`, `years`, ...) |
| `cap` | a capitalized word (names, titles) |
| `word` | any other word |
| `paren` | `(` `)` |
| `sep` | list separators: `,` `;` `\|` `and` |

For each class, every answer gets three 0/1 features:
- `added c`: the answer has tokens of class c that the reference does not, and drops none;
- `dropped c`: the reference has tokens of class c that the answer lacks, and the answer adds none;
- `replaced c`: both. This is a different value — **what** the answer says, not **how** it is written — so it is
  excluded from the rules.

**(b) Small model: which edits make answers wrong.** For every edit e, over all pool answers:

- relevance(e) = P(wrong | e) − P(wrong | no e);
- score(e) = P(e | wrong) × max(relevance(e), 0).

The 3 form edits (`added` / `dropped`) with the highest scores are kept. The official normalization already forgives
some edits, such as articles; those get no relevance by themselves.

**(c) Small model: where each edit occurs.** For each of the 3 edits, an L1 logistic regression predicts it from the
question's word 1–2-grams:
- the n-grams must occur in at least 10 questions;
- C is chosen from {0.03, 0.1, 0.3, 1} by 5-fold cross-validation on log loss;
- the two n-grams with the largest positive weights are the edit's condition.

**(d) Translation: weights to rules, by a fixed template.** Each edit gives two candidates:
- global: `- Do not add <X> to the final answer.` (for an `added` edit), or `- Keep <X> in the final answer.` (for a
  `dropped` edit);
- conditional: `- Only for questions containing '<n-gram 1>' or '<n-gram 2>': <same action>. For all other questions,
  write the final answer as before.`

`<X>` is a fixed phrase per class, for example "several items separated by commas, 'and' or '|'" or "the unit word
after the number (e.g. 'yards', 'years')". 3 edits × 2 forms gives up to 6 candidates. Since no LLM rephrases them,
every rule says exactly what the model found.

### 5.3 Fair pairing with `tg`

`tg+edit` uses the same fit batches and the same per-step salt as `tg`. All TextGrad calls are cached, and textgrad's
gradient set is sorted before the update. So **until the first ShAD rule is kept, `tg+edit` replays `tg` exactly**,
and those steps cost nothing. After that point the two arms differ only because of the ShAD rules and their
downstream effect on TextGrad.

### 5.4 Output

The state is in `textgrad/tg+edit.json`:
- `sys`: the final system prompt;
- `block`: the kept ShAD rules;
- `rejected`: every candidate that was tested and not kept;
- `log`: per step, `tg_val`, `tg_kept`, `shad_tested` (each candidate with its val EM), `shad_kept` and `val`.

---

## 6. Experiment 2 — ShAD added once to TextGrad's final prompt

**Question.** Starting from the same APO result, are ShAD's rules better than the rules an LLM writes from the same
errors?

Both arms start from `tg`'s final system prompt, use the same 1000 pool answers (one answer file, shared), give 6
candidate rules, test every candidate alone on the same val, and send only the best on val to the same test. If no
candidate reaches TextGrad's own val EM, the arm keeps TextGrad's prompt, and the table says so.

### 6.1 `+ ShAD edit (1 step)` (`qa/drop_editfeat.py`)

This is the ShAD step of 5.2, applied once to `tg`'s final prompt. It writes:
- `editfeat2/rules.json`: all candidates, with their edit and condition;
- `editfeat2/best.json`: the rule chosen on val, its val EM, and TextGrad's val EM.

The log also prints the full edit-feature table (frequency, relevance, n) and the n-gram conditions.

### 6.2 `+ error summary` (control, `qa/drop_editfeat_ctrl.py`)

The same model reads 30 wrong and 10 right pool answers with their references, and writes up to 6 rules for the final
answer line:

> A model answers <dataset description>. Below are its answers with the reference answers (mostly wrong ones). [pairs]
> Write up to 6 short rules for how to write the final answer line so that it matches the reference answers better.
> Order them from most to least important.

It writes `errsum_ctrl/best.json`, which holds all 6 candidates with their val EM and the chosen rule. On test, the
log also prints the paired difference between the chosen edit rule and the chosen error-summary rule.

The difference between the arms is only **where the rule comes from**: features plus a small model translated by a
template, versus an LLM's reading of the same errors.

---

## 6b. Experiment 3 — ShAD with evaluation dimensions (`qa/dimshad.py`, `qa/dimlearn.py`)

This is the ShAD of the rating tasks moved to the generator. A judge rates the answer A on several **evaluation
dimensions**, ShAD decodes them, and the decoder's weights are translated into rules for the answer prompt.

1. **Judge.** The same model rates the pool answer A and the reference GT, each alone with the question and never
   both together, on K dimensions of how an answer is written (1–5). It does not judge correctness.
   - `dimshad.py` uses 6 fixed dimensions: verbosity, specificity, granularity, multiplicity, qualification and
     type fit. The judge also tags the question type.
   - `dimlearn.py` learns the dimensions instead. The model proposes 24 candidate dimensions from contrasting pairs
     of the first 300 pool answers; the data then select among them.
2. **ShAD decoder.** Per dimension k, a ridge regression fits GT_k from all K dimensions of A plus log length.
   - The weights show which dimensions move together (shared axes).
   - DEV_k = A_k − decoded GT_k is how A should move on dimension k.
3. **Which deviations cost.** P(wrong) is regressed on the deviations above and below the decoded reference
   (logistic).
   - In `dimlearn.py` this is an L1 logistic regression. Its strength is the sparsest one within one SE of the best
     5-fold log loss, and deviations are scaled by their mean absolute size.
   - In `dimlearn.py` the selected deviations are then grouped into shared axes by a non-negative factorization (NMF),
     whose rank is chosen by 5-fold reconstruction.
4. **Translation by template (no LLM).** Every costly deviation (for `dimlearn`, every axis) becomes one global rule.
   Rules are ranked by cost weight × share of answers.
   - **1 rule** (`drop_dimshad_eval.py`): every rule is tested alone on val and the best goes to test.
   - **Stacked** (`drop_dimshad_combine.py [--src dimlearn]`): in rank order, each rule is added to the block if val
     EM does not drop, as in round-by-round APO; the final block goes to test.

## 7. Final table (`qa/qa_report.py`, no model calls)

```
== musique
   arm                     test EM   vs TextGrad        vs error summary
   TextGrad                   xx.x
   + error summary            xx.x    +x.x (SE x.x)
   + ShAD edit (1 step)       xx.x    +x.x (SE x.x) *    +x.x (SE x.x) *
   TextGrad x ShAD edit       xx.x    +x.x (SE x.x)      +x.x (SE x.x)
   rule [...]: the chosen / kept rules
```

- **Differences** are paired over the same 1000 test questions.
- **SE** is a bootstrap over questions (2000 resamples). `*` marks |difference| > 2 SE.
- **Experiment 1:** `TextGrad x ShAD edit` vs `TextGrad`.
- **Experiment 2:** `+ ShAD edit (1 step)` vs `+ error summary` (and vs `TextGrad`).

**Reference (Gemma 4 26B A4B via OpenRouter, DROP, one run):**

| arm | test EM |
|---|---|
| TextGrad | 81.9 |
| + error summary | 81.0 |
| + ShAD edit (1 step) | 83.7 |

The rule chosen on val was "Do not add several items separated by commas, 'and' or '\|' to the final answer". ShAD
edit vs error summary was +2.7 (SE 0.9).

## 8. Size of the run

Approximate number of calls per dataset (the cache removes repeats):

| arm | calls |
|---|---|
| TextGrad (`tg`) | about 7.7k: 12 steps × (about 12 TextGrad calls + 500 val), plus 500 start val and 1000 test |
| TextGrad x ShAD edit | up to about 23k: 4 ShAD steps × (1000 pool + up to 6 × 500 val), the TextGrad steps after divergence, 1000 test |
| ShAD edit (1 step) | about 5.6k: 1000 pool + 6 × 500 val + 1000 test (+ TextGrad's val and test, cached) |
| error-summary control | about 4k: 1 + 6 × 500 val + 1000 test + the edit-rule test (cached) |

That is about 40k calls per dataset. A typical call is about 700 output tokens. Input per call is about 1.1k tokens
on DROP, 1.5k on HotpotQA and 2.8k on MuSiQue. Token counts per call are logged in the `spend` table of
`$GEM_OUT/cache.sqlite`.

## 9. Files

| file | role |
|---|---|
| `run_qa_llm.sh` | the whole pipeline (preflight, arms in order, final table) |
| `qa/drop_textgrad.py` | arms `tg` and `tg+edit` (`tg+shad` is an older ShAD variant with embedding features; it needs OpenRouter embeddings and is not part of this run) |
| `qa/editfeat.py` | ShAD edit model: features, small model, template rules |
| `qa/drop_editfeat.py` | experiment 2, ShAD arm |
| `qa/drop_editfeat_ctrl.py` | experiment 2, error-summary control |
| `qa/qa_report.py` | final table from the saved answers |
| `qa/shadqa.py` | dataset adapters (rows, prompt, scoring), fixed samples |
| `baselines/gem.py` | call layer (`GEM_PROVIDER=llm` -> your `llm` package), cache, token log |
| `data/qa/` | the fixed samples (train positions 0–4099, test 1000) |

The dataset is chosen with `QA_DS=<drop|musique|hotpot>`, which `run_qa_llm.sh` sets. The other files in `qa/` are
earlier QA explorations and are not used by this run.

## 10. Troubleshooting

- **`PREFLIGHT FAILED`.** The `llm` package did not import, or the test call failed. The message above it says which.
- **`llm package: ... 401/403/quota`.** This is an account-level error. Everything stops; fix it and rerun.
- **`... returned an empty response`** (e.g. from SafeChain). Every call that comes back empty or with a transient
  error is retried 4 times (2, 4, 8, 16 s apart); empty replies are never cached. A question still failing after
  that is scored wrong and the run goes on; a TextGrad step still failing is counted as a revert. The preflight call
  allows 1000 output tokens, so a reasoning model has room to answer. If many questions fail, check the provider; failed
  questions have an empty `pred` column in the answers CSV.
- **Other errors on single questions** (for example, content refused). The question is scored wrong and the run goes
  on. If an error occurs inside a TextGrad step, that process stops. Rerun the command; it resumes from the last saved
  step.
- **Many replies without an `Answer:` line in a log.** Usually a TextGrad prompt made the reasoning too long for
  3000 tokens. Val EM catches it and the step is reverted.
- **Checking a single arm by hand:**
  - `QA_DS=musique GEM_PROVIDER=llm python qa/drop_textgrad.py tg --workers 16`
  - `... qa/drop_textgrad.py tg+edit ...`
  - `... qa/drop_editfeat.py ...`
  - `... qa/drop_editfeat_ctrl.py ...` (needs `tg` finished first)

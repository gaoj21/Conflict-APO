# ShAD experiments through your `llm` package (HS2, ELLIPSE, MERLIN)

All model calls go through the project's `llm` package (llm-v12: `from llm import chat_result`). Which model and
endpoint are used is decided only by `llm/providers.json`; the experiment code never names a provider.
HANNA is not included (about 30% of its stories contain violent content).

## 1. Install

```bash
python3 -m venv .venv && . .venv/bin/activate        # Python 3.10+
pip install -r requirements_local.txt
```

## 2. Put your `llm` package next to this file

The scripts import `llm` from this directory (or from `GEM_LLM_PATH`). Either copy your own package here:

```bash
cp -r /path/to/your/llm ./llm
```

or start from the template in `llm_template/` (the llm-v12 package used before) and edit `llm/providers.json`
(the default provider, or pick one with `GEM_LLM_PROVIDER`):

```bash
cp -r llm_template llm   # then edit llm/providers.json; keys come from the environment variables it names
```

Check one call: `python -c "import llm; print(llm.chat(None, [{'role':'user','content':'Say OK'}]))"`.

## 3. Run

```bash
export GEM_LLM_PROVIDER=<provider name in providers.json>   # optional; default provider otherwise
export WORKERS=16                                          # concurrent requests per process (3 datasets run in parallel)
bash run_pipeline_llm.sh                                   # hs2 ell merlin
bash run_pipeline_llm.sh ell                               # one dataset
WITH_SLOW=1 bash run_pipeline_llm.sh                       # also GEPA and OPRO baselines
```

Environment variables: `GEM_PROVIDER=llm` (set by the script), `GEM_LLM_PROVIDER`, `GEM_LLM_PATH` (where `llm/` is,
default this directory), `GEM_MODEL` (a label for logs and the call cache; default the provider name), `GEM_OUT`
(results; default `results/llm_<label>`), `GEM_BUDGET` (off by default for this provider; token counts are logged in
`$GEM_OUT/cache.sqlite`, table `spend`). Judge calls at temperature 0 are cached, so the script is resumable.

## 4. What it runs (per dataset, datasets in parallel)

1. Baselines: v0, few-shot, DSPy zero-shot, DSPy MIPROv2, TextGrad CCC (judge = optimizer = your model);
   with `WITH_SLOW=1` also GEPA and OPRO.
2. Key experiment (`run_distill_cmp.py`): TextGrad's final prompt frozen; teacher = ShAD cross-fitted on fit only;
   rules written by the LLM from (a) ShAD's coefficients, (b) teacher contrastive pairs, (c) human-error contrastive
   pairs (baseline); 3 candidates each, selected on sel by a conservative paired-bootstrap score; test once.
3. ShAD into the final prompt: numeric length rules (`run_postrules.py`), LLM text rules and combined
   (`run_llmrules.py [--combine]`), on v0 and TextGrad's final prompt.
4. Summaries in `$GEM_OUT/logs/`: `main_table.txt` (raw vs + ShAD decoder for every method), `key_experiment.txt`,
   `rules_summary.txt`. Metric: test Spearman, mean over attributes.

Rough size: 40–60k calls per dataset without GEPA/OPRO; judge outputs are ~100–250 tokens.

## 5. Data

`results/claude_{hs2,ell,merlin}/`: attributes, output format, rubric v0 (`prompts/v0.txt`), gold human ratings
(`gold.csv`, with the standardized log length `z`), and fit / sel / test items (`chunks/`).

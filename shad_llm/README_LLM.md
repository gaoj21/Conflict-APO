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
WITH_MIPRO=1 bash run_pipeline_llm.sh                      # also the DSPy MIPROv2 reference
```

The terminal shows only a progress table, refreshed every 30 s (`REFRESH=60` to change, `QUIET=1` to turn it off);
all details go to `$GEM_OUT/logs/*.log`. To check progress from another terminal: `python scripts/progress.py`.
At the end the script prints the final table (also in `$GEM_OUT/logs/final_report.txt`).

Environment variables: `GEM_PROVIDER=llm` (set by the script), `GEM_LLM_PROVIDER`, `GEM_LLM_PATH` (where `llm/` is,
default this directory), `GEM_MODEL` (a label for logs and the call cache; default the provider name), `GEM_OUT`
(results; default `results/llm_<label>`), `GEM_BUDGET` (off by default for this provider; token counts are logged in
`$GEM_OUT/cache.sqlite`, table `spend`). Judge calls at temperature 0 are cached, so the script is resumable.

## 4. Experiment design

**Question.** Multi-attribute judges tuned by APO show conflicts (an edit helps one attribute and hurts another). ShAD
(a per-attribute ridge decoder over all judge scores + log length, anchored to the attribute's own score) shows that
most of this comes from axes shared by all attributes. Can ShAD's corrections be written into the prompt, so the judge
improves **without conflicts and without a decoder at inference**? And is it better to add them once after APO, or
round by round during APO?

**Arms** (judge = optimizer = your model; seed 0; TextGrad CCC as the APO method; same format for every rule arm:
word count shown with each item, half points allowed):

| arm | what it is |
|---|---|
| v0 | the unoptimized rubric (reference) |
| TextGrad | plain TextGrad CCC, 12 steps (reference) |
| TextGrad + format | TextGrad's final prompt with the format change only (**control** for every rule arm) |
| + length rules | ShAD fitted once on the final prompt; its length weights as per-band rating adjustments |
| + text rules | ShAD's cross-attribute weights described to the LLM, which writes number-free rules (3 candidates, best on sel) |
| + combined | text rules (cross-attribute only) + length rules |
| TextGrad x ShAD rounds | after every TextGrad step: refit ShAD on fit+sel, add half of the remaining length correction, rewrite the text rules; keep the update only if val MAE does not increase. The rules sit outside the optimized system prompt, so TextGrad cannot edit them |

**Metrics** (test set, never used for any choice): mean Spearman x100 over attributes of the judge's own ratings
(**raw**, no decoder); for full prompts also **+ ShAD** (the decoder refitted on that prompt's fit+sel scores).
**Conflicts**: per attribute, the change of Spearman vs "TextGrad + format"; an attribute is up / down when the change
exceeds two paired-bootstrap SE; a conflict is at least one up and one down.

**Failed items are dropped.** Step 0 judges every item with v0; any item whose call fails (e.g. the API refuses
its content) or whose ratings cannot be parsed is written to `$GEM_OUT/<ds>/excluded_ids.txt`
(`scripts/screen_items.py`). All later steps skip these items and all evaluations ignore them, so every arm is compared
on the same items. An item that fails later is dropped the same way; account-level errors (key, quota, billing) stop
the run. `$GEM_OUT/logs/screen.txt` reports how many items were dropped per dataset.

**Data roles.** fit + sel: ShAD fitting, rule generation, TextGrad's training and validation; sel: choice among rule
candidates; test: final evaluation only.

**Reading the result.** The method works if the rule arms raise raw Spearman over "TextGrad + format" with no
attribute significantly down; the round-by-round arm is worth its cost only if it beats "+ combined".

**Output.** `$GEM_OUT/logs/final_report.txt` (from `scripts/final_report.py`): one table per dataset with raw, + ShAD,
up / down / conflict and the per-attribute changes. With `WITH_MIPRO=1` also DSPy MIPROv2 (raw vs + ShAD decoder) as a
stronger APO reference.

**Cost.** The round-by-round arm re-judges fit + sel after every step (~12 x 700-1160 calls per dataset); the whole
pipeline is roughly 30-40k calls per dataset.

## 5. Data

`results/claude_{hs2,ell,merlin}/`: attributes, output format, rubric v0 (`prompts/v0.txt`), gold human ratings
(`gold.csv`, with the standardized log length `z`), and fit / sel / test items (`chunks/`).

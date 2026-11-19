#!/bin/bash
# ShAD experiments through the project's `llm` package (your company API), on HS2, ELLIPSE and MERLIN.
#   bash run_pipeline_llm.sh [datasets]        default: hs2 ell merlin
# Resumable: every step is skipped when its output exists. Datasets run in parallel.
set -u
cd "$(dirname "$0")"
export GEM_PROVIDER=llm
PY=${PY:-python}
DS=${*:-"hs2 ell merlin"}
W=${WORKERS:-16}
OUT=${GEM_OUT:-$($PY -c "import sys; sys.path.insert(0,'baselines'); import gem; print(gem.OUT)")}
export GEM_OUT=$OUT
L=$OUT/logs; mkdir -p $L
have() { [ -f $OUT/$1/$2/seed0/scores_test.csv ]; }

one() { d=$1
  # 1. baselines (judge = optimizer = the llm package's model)
  have $d v0            || $PY baselines/run_simple.py   $d v0       --workers $W > $L/${d}_v0.log 2>&1
  have $d fewshot       || $PY baselines/run_simple.py   $d fewshot  --workers $W > $L/${d}_fewshot.log 2>&1
  have $d dspy_zeroshot || $PY baselines/run_dspy.py     $d zeroshot --workers $W > $L/${d}_dspy_zeroshot.log 2>&1
  have $d dspy_mipro    || $PY baselines/run_dspy.py     $d mipro    --workers $W > $L/${d}_dspy_mipro.log 2>&1
  have $d textgrad_ccc  || $PY baselines/run_textgrad.py $d ccc      --workers $W > $L/${d}_textgrad_ccc.log 2>&1
  if [ "${WITH_SLOW:-0}" = 1 ]; then   # slower APO baselines, off by default
    have $d dspy_gepa   || $PY baselines/run_dspy.py     $d gepa     --workers $W > $L/${d}_dspy_gepa.log 2>&1
    have $d opro        || $PY baselines/run_opro.py     $d          --workers $W > $L/${d}_opro.log 2>&1
  fi
  # 2. key experiment: frozen TextGrad prompt; coefficient text vs teacher contrastive vs human-error contrastive rules
  [ -f $OUT/$d/distillcmp_textgrad_ccc/seed0/result.json ] || $PY baselines/run_distill_cmp.py $d --base textgrad_ccc --workers $W > $L/${d}_distillcmp.log 2>&1
  # 3. ShAD rules on final prompts: numeric length rules, LLM text rules, combined
  for b in v0 textgrad_ccc; do
    [ -f $OUT/$d/postrules_$b/seed0/result.json ]     || $PY baselines/run_postrules.py $d $b --workers $W > $L/${d}_postrules_$b.log 2>&1
    [ -f $OUT/$d/llmrules_$b/seed0/result.json ]      || $PY baselines/run_llmrules.py  $d $b --workers $W > $L/${d}_llmrules_$b.log 2>&1
    [ -f $OUT/$d/llmrules_comb_$b/seed0/result.json ] || $PY baselines/run_llmrules.py  $d $b --combine --workers $W > $L/${d}_llmrules_comb_$b.log 2>&1
  done
  echo "$d done $(date)"; }

for d in $DS; do one $d & done; wait
# 4. summaries
$PY baselines/evaluate.py $DS > $L/main_table.txt 2>&1          # every method: raw vs + ShAD decoder (test Spearman)
grep -h "control" $L/*_distillcmp.log      > $L/key_experiment.txt   # rules comparison on the frozen TextGrad prompt
grep -h "original" $L/*_llmrules_comb_*.log > $L/rules_summary.txt
cat $L/main_table.txt $L/key_experiment.txt $L/rules_summary.txt
echo PIPELINE_DONE

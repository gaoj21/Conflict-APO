#!/bin/bash
# Designed experiment (see README_LLM.md, "Experiment design"), through the project's `llm` package.
#   bash run_pipeline_llm.sh [datasets]        default: hs2 ell merlin
# Resumable: every step is skipped when its output exists. Datasets run in parallel; within a dataset the
# round-by-round arm and the control arm run in parallel too.
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

# arm R: TextGrad CCC with ShAD rules (text cross-attribute + numeric length) refitted and updated after every step
rounds() { d=$1
  have $d textgrad_ccc_shadrules || $PY baselines/run_textgrad.py $d ccc --seed 0 --shadrules --name textgrad_ccc_shadrules \
                                     --workers $W > $L/${d}_tg_shadrules.log 2>&1
  echo "$d rounds done $(date)"; }

# arm O (+ references): v0, plain TextGrad CCC, then ShAD rules added once to its final prompt
control() { d=$1
  have $d v0           || $PY baselines/run_simple.py   $d v0  --workers $W > $L/${d}_v0.log 2>&1
  have $d textgrad_ccc || $PY baselines/run_textgrad.py $d ccc --seed 0 --workers $W > $L/${d}_textgrad_ccc.log 2>&1
  [ -f $OUT/$d/postrules_textgrad_ccc/seed0/result.json ]     || $PY baselines/run_postrules.py $d textgrad_ccc --workers $W > $L/${d}_postrules_textgrad_ccc.log 2>&1
  [ -f $OUT/$d/llmrules_textgrad_ccc/seed0/result.json ]      || $PY baselines/run_llmrules.py  $d textgrad_ccc --workers $W > $L/${d}_llmrules_textgrad_ccc.log 2>&1
  [ -f $OUT/$d/llmrules_comb_textgrad_ccc/seed0/result.json ] || $PY baselines/run_llmrules.py  $d textgrad_ccc --combine --workers $W > $L/${d}_llmrules_comb_textgrad_ccc.log 2>&1
  if [ "${WITH_MIPRO:-0}" = 1 ]; then   # optional reference: the strongest APO baseline, raw vs + ShAD decoder
    have $d dspy_mipro || $PY baselines/run_dspy.py $d mipro --workers $W > $L/${d}_dspy_mipro.log 2>&1
  fi
  echo "$d control done $(date)"; }

for d in $DS; do rounds $d & control $d & done; wait
$PY scripts/final_report.py $DS | tee $L/final_report.txt
echo PIPELINE_DONE

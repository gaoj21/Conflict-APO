#!/bin/bash
# QA experiments (see README_QA.md) through the project's `llm` package.
#   bash run_qa_llm.sh [datasets]        default: drop musique hotpot
# Resumable: rerun the same command after an interruption; finished steps are read back from disk.
# Datasets run in parallel. Within a dataset: TextGrad first, then (in parallel) TextGrad x ShAD edit, and the two
# one-step arms on TextGrad's final prompt (ShAD edit rules, then the error-summary control).
set -u
cd "$(dirname "$0")"
export GEM_PROVIDER=llm
PY=${PY:-python}
DS=${*:-"drop musique hotpot"}
W=${WORKERS:-16}
if ! OUT=$(PREFLIGHT_DS="" $PY scripts/preflight.py); then
  echo "PREFLIGHT FAILED: fix the error above, then rerun."; exit 1
fi
export GEM_OUT=$OUT
L=$OUT/logs/qa; mkdir -p $L

one() { d=$1; export QA_DS=$d
  $PY qa/drop_textgrad.py tg --workers $W > $L/${d}_textgrad.log 2>&1 || { echo "[$d] TextGrad failed, see $L/${d}_textgrad.log"; return; }
  echo "[$d] TextGrad done"
  ( $PY qa/drop_textgrad.py tg+edit --workers $W > $L/${d}_textgrad_x_shad.log 2>&1 \
      && echo "[$d] TextGrad x ShAD edit done" || echo "[$d] TextGrad x ShAD edit failed, see $L/${d}_textgrad_x_shad.log" ) &
  ( $PY qa/drop_editfeat.py --workers $W > $L/${d}_shad_edit.log 2>&1 \
      && echo "[$d] ShAD edit (1 step) done" || echo "[$d] ShAD edit failed, see $L/${d}_shad_edit.log"
    $PY qa/drop_editfeat_ctrl.py --workers $W > $L/${d}_errsum.log 2>&1 \
      && echo "[$d] error-summary control done" || echo "[$d] error-summary control failed, see $L/${d}_errsum.log" )
  wait; }

echo "results: $OUT  logs: $L   (progress: tail -f $L/<dataset>_textgrad.log)"
for d in $DS; do one $d & done
wait
$PY qa/qa_report.py $DS | tee $L/final_report.txt
echo "QA_DONE  (table: $L/final_report.txt)"

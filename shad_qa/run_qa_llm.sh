#!/bin/bash
# QA experiments (see README_QA.md) through the project's `llm` package.
#   bash run_qa_llm.sh [datasets]        default: drop musique hotpot
# Resumable: rerun the same command after an interruption; finished steps are read back from disk.
# Datasets run in parallel. Within a dataset: TextGrad first, then (in parallel) TextGrad x ShAD edit, and the two
# one-step arms on TextGrad's final prompt (ShAD edit rules, then the error-summary control).
set -u
cd "$(dirname "$0")"
export GEM_PROVIDER=llm
export QA_VAL=${QA_VAL:-dev}   # val from the dev split, disjoint from test (QA_VAL=train: the earlier design)
PY=${PY:-python}
DS=${*:-"drop musique hotpot"}
W=${WORKERS:-16}
if ! PF=$(PREFLIGHT_DS="" "$PY" scripts/preflight.py); then
  echo "PREFLIGHT FAILED: fix the error above, then rerun."; exit 1
fi
OUT=$(printf '%s\n' "$PF" | tail -n 1)   # the last line only: the llm package may print to stdout too
if [ ! -d "$OUT" ]; then echo "PREFLIGHT: results folder '$OUT' does not exist (preflight printed: $PF)"; exit 1; fi
export GEM_OUT="$OUT"
L="$OUT/logs/qa"; mkdir -p "$L"   # every path is quoted: the folder may contain spaces

# run <log name> <command...>: the command's output goes to $L/<log name>.log; prints done / failed
run() { local log="$L/$1.log"; shift
  if "$@" > "$log" 2>&1; then return 0; else echo "    failed: see $log"; return 1; fi; }

one() { local d=$1; export QA_DS=$d
  run "${d}_textgrad" "$PY" qa/drop_textgrad.py tg --workers "$W" || { echo "[$d] TextGrad failed"; return; }
  echo "[$d] TextGrad done"
  ( run "${d}_textgrad_x_shad" "$PY" qa/drop_textgrad.py tg+edit --workers "$W" && echo "[$d] TextGrad x ShAD edit done" ) &
  ( run "${d}_shad_edit" "$PY" qa/drop_editfeat.py --workers "$W" && echo "[$d] ShAD edit (1 step) done"
    run "${d}_errsum" "$PY" qa/drop_editfeat_ctrl.py --workers "$W" && echo "[$d] error-summary control done"
    # ShAD with evaluation dimensions: fixed dimensions (one rule, and rules stacked one by one on val), then
    # dimensions learned from the data (stacked)
    run "${d}_shad_dims" "$PY" qa/dimshad.py --workers "$W" \
      && run "${d}_shad_dims_1rule" "$PY" qa/drop_dimshad_eval.py --workers "$W" \
      && run "${d}_shad_dims_stacked" "$PY" qa/drop_dimshad_combine.py --workers "$W" && echo "[$d] ShAD dimensions done"
    run "${d}_shad_learned_dims" "$PY" qa/dimlearn.py --workers "$W" \
      && run "${d}_shad_learned_dims_stacked" "$PY" qa/drop_dimshad_combine.py --src dimlearn --workers "$W" \
      && echo "[$d] ShAD learned dimensions done" )
  wait; }

echo "results: $OUT"
echo "logs:    $L   (progress: tail -f \"$L/<dataset>_textgrad.log\")"
for d in $DS; do one "$d" & done   # $DS is a list of dataset names (word splitting intended)
wait
"$PY" qa/qa_report.py $DS | tee "$L/final_report.txt"
echo "QA_DONE  (table: $L/final_report.txt)"

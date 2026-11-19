#!/bin/bash
# Single-Task jobs spend hours in the final evaluation (K prompts x fit/sel/test with 8 threads). Once every attribute
# of a job is optimized (all hist_*.json saved), restart it with 64 threads: the resume path skips optimization.
cd "$(dirname "$0")/.."
declare -A DONE
while true; do
  left=0
  for d in hs2 hanna ell merlin; do
    case $d in ell) R=claude_ell;; *) R=claude_$d;; esac
    n=$(wc -w < results/$R/attrs.txt)
    for s in 0 1 2; do
      dir=results/gemma/$d/textgrad_single/seed$s
      [ -f $dir/scores_test.csv ] && continue
      left=1
      [ -n "${DONE[$d$s]}" ] && continue
      h=$(ls $dir/hist_*.json 2>/dev/null | wc -l)
      if [ "$h" -eq "$n" ]; then
        pid=$(ps -u $USER -o pid=,args= | awk -v d=$d -v s=$s '$2 ~ /python$/ && $3 ~ /run_textgrad\.py$/ && $4==d && $5=="single" && $7==s {print $1}')
        [ -n "$pid" ] && kill $pid && sleep 3
        setsid nohup .venv/bin/python baselines/run_textgrad.py $d single --seed $s --workers 64 > results/gemma/logs/${d}_textgrad_single_${s}.final.log 2>&1 &
        DONE[$d$s]=1; echo "$(date +%H:%M) restarted $d seed $s for final eval"
      fi
    done
  done
  [ $left -eq 0 ] && { echo ALL_SINGLE_DONE; break; }
  sleep 60
done

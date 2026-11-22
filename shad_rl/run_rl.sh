#!/bin/bash
# GRPO on TL;DR with the raw judge reward and with the ShAD reward, then the gold evaluation (see HANDOVER_RL.md).
# Needs the judge server already running (JUDGE_BASE_URL, default http://localhost:8000/v1).
#   TRAIN_GPU=1 STEPS=200 bash run_rl.sh
# Resumable per run: a run whose final checkpoint exists is skipped; eval skips checkpoints already evaluated.
set -eu
cd "$(dirname "$0")"
PY=${PY:-python}
STEPS=${STEPS:-200}
export CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-1}
mkdir -p runs
for r in raw shad; do
  if [ -d "runs/$r/checkpoint-$STEPS" ]; then echo "[$r] done already"; continue; fi
  echo "[$r] training ($STEPS steps), log: runs/$r.log"
  "$PY" grpo_tldr.py --reward "$r" --out "runs/$r" --steps "$STEPS" ${GRPO_ARGS:-} > "runs/$r.log" 2>&1
done
if [ -n "${GOLD_MODEL:-}" ]; then "$PY" eval_gold.py runs/raw runs/shad | tee runs/eval.log
else echo "GOLD_MODEL not set: proxy-only evaluation"; "$PY" eval_gold.py runs/raw runs/shad --gold none | tee runs/eval.log; fi

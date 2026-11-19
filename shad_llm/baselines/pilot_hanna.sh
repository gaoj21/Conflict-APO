#!/bin/bash
# HANNA pilot: every method, seed 0, full settings, in parallel (8 judge threads each)
cd "$(dirname "$0")/.."
P=.venv/bin/python; L=results/gemma/logs; D=hanna; S=0
$P baselines/run_simple.py $D fewshot --seed $S --workers 8 > $L/${D}_fewshot_$S.log 2>&1 &
$P baselines/run_opro.py $D --seed $S --workers 8 > $L/${D}_opro_$S.log 2>&1 &
$P baselines/run_protegi.py $D --seed $S --workers 8 > $L/${D}_protegi_$S.log 2>&1 &
$P baselines/run_textgrad.py $D ccc --seed $S --workers 8 > $L/${D}_textgrad_ccc_$S.log 2>&1 &
$P baselines/run_textgrad.py $D single --seed $S --workers 8 > $L/${D}_textgrad_single_$S.log 2>&1 &
$P baselines/run_dspy.py $D mipro --seed $S --workers 8 > $L/${D}_dspy_mipro_$S.log 2>&1 &
$P baselines/run_dspy.py $D gepa --seed $S --workers 8 > $L/${D}_dspy_gepa_$S.log 2>&1 &
$P baselines/run_dspy.py $D zeroshot --seed $S --workers 8 > $L/${D}_dspy_zeroshot_$S.log 2>&1 &
wait
echo PILOT_DONE

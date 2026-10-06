#!/usr/bin/env bash
# Usage: launch_final.sh VARIANT "N1:42 N1:43 N2:42 ..."   one process per (model, seed); staggered to avoid CUDA/RAM init failures
cd "$(dirname "$0")/.."
VARIANT=$1; JOBS=$2
export NN_THREADS=${NN_THREADS:-1}
for job in $JOBS; do
  model=${job%%:*}; seed=${job##*:}
  sleep ${STAGGER:-30}; .venv/Scripts/python.exe scripts/final_fit_neural.py --model $model --variant $VARIANT --seeds $seed \
      > logs/final_${model}_${VARIANT}_s${seed}.log 2>&1 &
done
wait

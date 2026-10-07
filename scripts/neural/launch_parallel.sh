#!/usr/bin/env bash
# Usage: launch_parallel.sh MODEL VARIANT "fold:seed fold:seed ..."   (one process per (fold, seed))
cd "$(dirname "$0")/.."
MODEL=$1; VARIANT=$2; JOBS=$3
export NN_THREADS=${NN_THREADS:-1}
for job in $JOBS; do
  fold=${job%%:*}; seed=${job##*:}
  sleep ${STAGGER:-0}; .venv/Scripts/python.exe scripts/run_rolling.py --model $MODEL --variant $VARIANT --folds $fold --seeds $seed \
      > logs/${MODEL}_${VARIANT}_${fold}_s${seed}.log 2>&1 &
done
wait

#!/bin/bash
# v3 CTS matrix: 8 seeds x (frozen/naive/egc) x 16 tasks, sequential.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
MODEL=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3
OUT=$REPO/artifacts/v3/cts
mkdir -p $OUT
echo "v3 start $(date -u +%FT%TZ)" > $OUT/v3_status.txt
for seed in 0 1 2 3 4 5 6 7; do
  for variant in frozen naive egc; do
    [ -f $OUT/${variant}_s${seed}_16.log ] && continue
    echo "start ${variant}_s${seed} $(date -u +%FT%TZ)" >> $OUT/v3_status.txt
    setsid nohup nice -n 10 env PYTHONPATH=src $PY -u $REPO/scripts/cts_v2_stream.py \
      --variant $variant --seed $seed --n-tasks 16 --model $MODEL --device cuda:1 \
      > $OUT/${variant}_s${seed}_16.log 2>&1
    echo "done  ${variant}_s${seed} rc=$? $(date -u +%FT%TZ)" >> $OUT/v3_status.txt
  done
done
echo "v3 done $(date -u +%FT%TZ)" >> $OUT/v3_status.txt

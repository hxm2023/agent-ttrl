#!/bin/bash
# v2 CTS learnability matrix: 3 seeds x 2 arms (8-task) + 16-task naive.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
MODEL=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3
OUT=$REPO/artifacts/v2/cts
mkdir -p $OUT
echo "v2 matrix start $(date -u +%FT%TZ)" > $OUT/matrix_status.txt
for seed in 0 1 2; do
  for variant in frozen naive; do
    tag="${variant}_s${seed}"
    echo "start $tag $(date -u +%FT%TZ)" >> $OUT/matrix_status.txt
    setsid nohup nice -n 10 env PYTHONPATH=src $PY -u $REPO/scripts/cts_v2_stream.py \
      --variant $variant --seed $seed --n-tasks 8 --model $MODEL --device cuda:1 \
      > $OUT/${tag}.log 2>&1
    echo "done  $tag rc=$? $(date -u +%FT%TZ)" >> $OUT/matrix_status.txt
  done
done
echo "start naive_s0_16task $(date -u +%FT%TZ)" >> $OUT/matrix_status.txt
setsid nohup nice -n 10 env PYTHONPATH=src $PY -u $REPO/scripts/cts_v2_stream.py \
  --variant naive --seed 0 --n-tasks 16 --model $MODEL --device cuda:1 \
  > $OUT/naive_s0_16task.log 2>&1
echo "done  naive_s0_16task rc=$? $(date -u +%FT%TZ)" >> $OUT/matrix_status.txt
echo "v2 matrix done $(date -u +%FT%TZ)" >> $OUT/matrix_status.txt

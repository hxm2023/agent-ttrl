#!/bin/bash
# ctl5b: clean sequential re-run of ctl5 (tau2 16-task egc x 4 seeds) after
# the parallel-run crash. Same strong config; appends to ctl5_status.txt and
# writes the "ctl5 done" marker the ctl6 launcher waits on.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
MODEL=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3
OUT=$REPO/artifacts/m6
mkdir -p $OUT
PORT=8440
GPORT=55800
echo "ctl5b start $(date -u +%FT%TZ)" >> $OUT/ctl5_status.txt
for seed in 0 1 2 3; do
  tag="ctl5_egc_s${seed}"
  echo "start $tag (ctl5b) $(date -u +%FT%TZ)" >> $OUT/ctl5_status.txt
  setsid nohup nice -n 10 $PY -u $REPO/scripts/tau2_agent_stream.py \
    --model $MODEL --variant egc --seed $seed --n-tasks 16 \
    --lr 5e-5 --n-rollouts 16 --steps 8 \
    --port $PORT --group-port $GPORT > $OUT/${tag}.log 2>&1
  echo "done  $tag (ctl5b) rc=$? $(date -u +%FT%TZ)" >> $OUT/ctl5_status.txt
  PORT=$((PORT+1)); GPORT=$((GPORT+1))
done
echo "ctl5 done $(date -u +%FT%TZ)" >> $OUT/ctl5_status.txt

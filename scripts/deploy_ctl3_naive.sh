#!/bin/bash
# ctl3: naive strong-update control with per-sequence gradient accumulation
# (16 rollouts x 8 steps x lr 5e-5, 16 tasks, 4 seeds) — matches ctl_frozen baselines.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
MODEL=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3
OUT=$REPO/artifacts/m6
mkdir -p $OUT
PORT=8420
GPORT=55600
echo "ctl3 start $(date -u +%FT%TZ)" > $OUT/ctl3_status.txt
for seed in 0 1 2 3; do
  tag="ctl3_naive_s${seed}"
  echo "start $tag $(date -u +%FT%TZ)" >> $OUT/ctl3_status.txt
  setsid nohup nice -n 10 $PY -u $REPO/scripts/tau2_agent_stream.py \
    --model $MODEL --variant naive --seed $seed --n-tasks 16 \
    --lr 5e-5 --n-rollouts 16 --steps 8 \
    --port $PORT --group-port $GPORT > $OUT/${tag}.log 2>&1
  echo "done  $tag rc=$? $(date -u +%FT%TZ)" >> $OUT/ctl3_status.txt
  PORT=$((PORT+1)); GPORT=$((GPORT+1))
done
echo "ctl3 done $(date -u +%FT%TZ)" >> $OUT/ctl3_status.txt

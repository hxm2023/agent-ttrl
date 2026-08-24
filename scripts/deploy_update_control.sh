#!/bin/bash
# Update-effectiveness control: strong updates (lr 5e-5, 16 rollouts, 8 steps)
# naive vs frozen, 16 tasks, 4 seeds.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
MODEL=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3
OUT=$REPO/artifacts/m6
mkdir -p $OUT
PORT=8400
GPORT=55400
echo "control start $(date -u +%FT%TZ)" > $OUT/control_status.txt
for variant in frozen naive; do
  for seed in 0 1 2 3; do
    tag="ctl_${variant}_s${seed}"
    echo "start $tag $(date -u +%FT%TZ)" >> $OUT/control_status.txt
    setsid nohup nice -n 10 $PY -u $REPO/scripts/tau2_agent_stream.py \
      --model $MODEL --variant $variant --seed $seed --n-tasks 16 \
      --lr 5e-5 --n-rollouts 16 --steps 8 \
      --port $PORT --group-port $GPORT > $OUT/${tag}.log 2>&1
    echo "done  $tag rc=$? $(date -u +%FT%TZ)" >> $OUT/control_status.txt
    PORT=$((PORT+1)); GPORT=$((GPORT+1))
  done
done
echo "control done $(date -u +%FT%TZ)" >> $OUT/control_status.txt

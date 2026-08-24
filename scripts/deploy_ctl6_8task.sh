#!/bin/bash
# ctl6: tau2 8-task pool re-run, SAME strong config as the 16-task control
# (16 rollouts x 8 steps x lr 5e-5) — replaces the tainted/unrecoverable
# 8-task cell (pre-audit egc never updated; frozen manifests overwritten).
# 3 variants x 4 seeds = 12 runs.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
MODEL=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3
OUT=$REPO/artifacts/m6
mkdir -p $OUT
PORT=8460
GPORT=56000
echo "ctl6 start $(date -u +%FT%TZ)" > $OUT/ctl6_status.txt
for variant in frozen naive egc; do
  for seed in 0 1 2 3; do
    tag="ctl6_${variant}_s${seed}"
    echo "start $tag $(date -u +%FT%TZ)" >> $OUT/ctl6_status.txt
    setsid nohup nice -n 10 $PY -u $REPO/scripts/tau2_agent_stream.py \
      --model $MODEL --variant $variant --seed $seed --n-tasks 8 \
      --lr 5e-5 --n-rollouts 16 --steps 8 \
      --port $PORT --group-port $GPORT > $OUT/${tag}.log 2>&1
    echo "done  $tag rc=$? $(date -u +%FT%TZ)" >> $OUT/ctl6_status.txt
    PORT=$((PORT+1)); GPORT=$((GPORT+1))
  done
done
echo "ctl6 done $(date -u +%FT%TZ)" >> $OUT/ctl6_status.txt

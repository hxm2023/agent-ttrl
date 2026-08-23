#!/bin/bash
# ctl5c: rerun the two ctl5b seeds lost to the engine crash (s2, s3).
# Same strong config; ports 8442/8443, group-ports 55802/55803.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
MODEL=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3
OUT=$REPO/artifacts/m6
mkdir -p $OUT
echo "ctl5c start $(date -u +%FT%TZ)" >> $OUT/ctl5_status.txt
for seed in 2 3; do
  tag="ctl5_egc_s${seed}"
  echo "start $tag (ctl5c) $(date -u +%FT%TZ)" >> $OUT/ctl5_status.txt
  port=$((8440 + seed)); gport=$((55800 + seed))
  setsid nohup nice -n 10 $PY -u $REPO/scripts/tau2_agent_stream.py \
    --model $MODEL --variant egc --seed $seed --n-tasks 16 \
    --lr 5e-5 --n-rollouts 16 --steps 8 \
    --port $port --group-port $gport > $OUT/${tag}.log 2>&1
  echo "done  $tag (ctl5c) rc=$? $(date -u +%FT%TZ)" >> $OUT/ctl5_status.txt
done
echo "ctl5c done $(date -u +%FT%TZ)" >> $OUT/ctl5_status.txt

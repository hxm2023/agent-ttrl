#!/bin/bash
# M6 factorial: Mistral-7B x tau2 retail x 8 tasks x 3 seeds x {frozen, naive, egc}
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
MODEL=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3
OUT=$REPO/artifacts/m6
mkdir -p $OUT
PORT=8300
GPORT=54300
echo "factorial start $(date -u +%FT%TZ)" > $OUT/factorial_status.txt
for variant in frozen naive egc; do
  for seed in 0 1 2; do
    tag="${variant}_m7_s${seed}"
    echo "start $tag $(date -u +%FT%TZ)" >> $OUT/factorial_status.txt
    setsid nohup nice -n 10 $PY -u $REPO/scripts/tau2_agent_stream.py \
      --model $MODEL --variant $variant --n-tasks 8 --port $PORT --group-port $GPORT \
      > $OUT/${tag}.log 2>&1
    echo "done  $tag rc=$? $(date -u +%FT%TZ)" >> $OUT/factorial_status.txt
    PORT=$((PORT+1)); GPORT=$((GPORT+1))
  done
done
echo "factorial done $(date -u +%FT%TZ)" >> $OUT/factorial_status.txt

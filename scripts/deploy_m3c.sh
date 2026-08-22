#!/bin/bash
# Deploy M3 v2: 3 seeds x 4 update variants (naive/egc/egc_conflict/random_branch),
# 4-step updates, server->GPU1 / trainer->GPU0 layout.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
OUT=$REPO/artifacts/m3
mkdir -p $OUT
PORT=8090
GPORT=51900
echo "deploy6 start $(date -u +%FT%TZ)" > $OUT/batch6_status.txt
for variant in naive egc egc_conflict random_branch; do
  for seed in 0 1 2; do
    tag="${variant}_s${seed}"
    echo "start $tag $(date -u +%FT%TZ)" >> $OUT/batch6_status.txt
    setsid nohup nice -n 10 $PY -u $REPO/scripts/m3_stream_pilot.py \
      --variant $variant --seed $seed --port $PORT --group-port $GPORT \
      --trainer-gpu 0 --server-gpu 1 \
      > $OUT/${tag}.log 2>&1
    echo "done  $tag rc=$? $(date -u +%FT%TZ)" >> $OUT/batch6_status.txt
    PORT=$((PORT+1)); GPORT=$((GPORT+1))
  done
done
echo "deploy6 done $(date -u +%FT%TZ)" >> $OUT/batch6_status.txt

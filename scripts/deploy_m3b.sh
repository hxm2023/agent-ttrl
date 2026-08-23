#!/bin/bash
# Deploy M3 remaining variants (egc/egc_conflict/random_branch x 2 seeds).
# Parallel-with-GRPO-Guard layout: server->GPU0 (Guard probe on GPU0), trainer->GPU1.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
OUT=$REPO/artifacts/m3
mkdir -p $OUT
PORT=8050
GPORT=51500
echo "deploy2 start $(date -u +%FT%TZ)" > $OUT/batch2_status.txt
for variant in egc egc_conflict random_branch; do
  for seed in 0 1; do
    tag="${variant}_s${seed}"
    echo "start $tag $(date -u +%FT%TZ)" >> $OUT/batch2_status.txt
    setsid nohup nice -n 10 $PY -u $REPO/scripts/m3_stream_pilot.py \
      --variant $variant --seed $seed --port $PORT --group-port $GPORT \
      --trainer-gpu 1 --server-gpu 0 \
      > $OUT/${tag}.log 2>&1
    echo "done  $tag rc=$? $(date -u +%FT%TZ)" >> $OUT/batch2_status.txt
    PORT=$((PORT+1)); GPORT=$((GPORT+1))
  done
done
echo "deploy2 done $(date -u +%FT%TZ)" >> $OUT/batch2_status.txt

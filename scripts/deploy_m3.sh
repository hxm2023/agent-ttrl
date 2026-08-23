#!/bin/bash
# Deploy M3 batch: 5 variants x 2 seeds, sequential on autodl2 (setsid + nohup, absolute paths)
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
OUT=$REPO/artifacts/m3
mkdir -p $OUT
PORT=8030
GPORT=51400
echo "deploy start $(date -u +%FT%TZ)" > $OUT/batch_status.txt
for variant in frozen naive egc egc_conflict random_branch; do
  for seed in 0 1; do
    tag="${variant}_s${seed}"
    echo "start $tag $(date -u +%FT%TZ)" >> $OUT/batch_status.txt
    setsid nohup nice -n 10 $PY -u $REPO/scripts/m3_stream_pilot.py \
      --variant $variant --seed $seed --port $PORT --group-port $GPORT \
      > $OUT/${tag}.log 2>&1
    rc=$?
    echo "done  $tag rc=$rc $(date -u +%FT%TZ)" >> $OUT/batch_status.txt
    PORT=$((PORT+1)); GPORT=$((GPORT+1))
  done
done
echo "deploy done $(date -u +%FT%TZ)" >> $OUT/batch_status.txt

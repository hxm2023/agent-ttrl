#!/bin/bash
# Deploy M2 CTS baselines: best_of_n / reflexion / hard_verifier x 2 seeds.
set -u
REPO=/root/autodl-tmp/agent-ttrl
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
OUT=$REPO/artifacts/m2
mkdir -p $OUT
PORT=8150
GPORT=52500
echo "m2 start $(date -u +%FT%TZ)" > $OUT/m2_status.txt
for variant in best_of_n reflexion hard_verifier; do
  for seed in 0 1; do
    tag="${variant}_s${seed}"
    echo "start $tag $(date -u +%FT%TZ)" >> $OUT/m2_status.txt
    setsid nohup nice -n 10 $PY -u $REPO/scripts/m2_baselines.py \
      --variant $variant --seed $seed --port $PORT --group-port $GPORT --server-gpu 1 \
      > $OUT/${tag}.log 2>&1
    echo "done  $tag rc=$? $(date -u +%FT%TZ)" >> $OUT/m2_status.txt
    PORT=$((PORT+1)); GPORT=$((GPORT+1))
  done
done
echo "m2 done $(date -u +%FT%TZ)" >> $OUT/m2_status.txt

#!/bin/bash
# run_tau2_seed.sh <port> <task> <seed-1> <seed-2> ...
PORT=$1; T=$2; shift 2
cd /root/autodl-tmp/agent-ttrl
for S in "$@"; do
  echo "[seed] task $T seed $S start $(date +%H:%M:%S)" >> logs_seed_$PORT.txt
  /root/miniconda3/bin/python scripts/tau2_official_pilot.py --task-idx $T --seed $S --base-url http://localhost:$PORT/v1 >> logs_seed_$PORT.txt 2>&1
  echo "[seed] task $T seed $S done $(date +%H:%M:%S)" >> logs_seed_$PORT.txt
done

#!/bin/bash
# run_tau2_batch.sh <gpu-port> <task-idx-1> <task-idx-2> ...
PORT=$1; shift
cd /root/autodl-tmp/agent-ttrl
for T in "$@"; do
  echo "[batch] task $T on port $PORT start $(date +%H:%M:%S)" >> logs_batch_$PORT.txt
  /root/miniconda3/bin/python scripts/tau2_official_pilot.py --task-idx $T --seed 0 --base-url http://localhost:$PORT/v1 >> logs_batch_$PORT.txt 2>&1
  echo "[batch] task $T done $(date +%H:%M:%S)" >> logs_batch_$PORT.txt
done

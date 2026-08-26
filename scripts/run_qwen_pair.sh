#!/bin/bash
# run_qwen_pair.sh <device> <seed-1> <seed-2> ...
D=$1; shift
cd /root/autodl-tmp/agent-ttrl
for S in "$@"; do
  echo "[batch] v32-pair seed $S start $(date +%H:%M)" >> logs_qwen_pair.txt
  PYTHONPATH=/root/autodl-tmp/agent-ttrl/src /root/miniconda3/bin/python scripts/cts_v2_stream.py --variant naive --seed $S --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --n-tasks 16 --device $D >> logs_qwen_pair.txt 2>&1
  echo "[batch] v32-pair seed $S done $(date +%H:%M)" >> logs_qwen_pair.txt
done

#!/bin/bash
# run_qwen_gate0.sh <device> <seed-1> <seed-2> ...
D=$1; shift
cd /root/autodl-tmp/agent-ttrl
for S in "$@"; do
  echo "[batch] gate0 naive seed $S start $(date +%H:%M)" >> logs_qwen_gate0.txt
  PYTHONPATH=/root/autodl-tmp/agent-ttrl/src /root/miniconda3/bin/python scripts/cts_v30_stream.py --variant naive --seed $S --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --n-tasks 16 --device $D --gate-threshold 0.0 >> logs_qwen_gate0.txt 2>&1
  echo "[batch] gate0 seed $S done $(date +%H:%M)" >> logs_qwen_gate0.txt
done

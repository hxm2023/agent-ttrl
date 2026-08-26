#!/bin/bash
# run_qwen_variant.sh <variant> <device> <seed-1> <seed-2> ...
V=$1; D=$2; shift 2
cd /root/autodl-tmp/agent-ttrl
for S in "$@"; do
  echo "[batch] $V seed $S start $(date +%H:%M)" >> logs_qwen_$V.txt
  PYTHONPATH=/root/autodl-tmp/agent-ttrl/src /root/miniconda3/bin/python scripts/cts_v2_stream.py --variant $V --seed $S --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --n-tasks 16 --device $D >> logs_qwen_$V.txt 2>&1
  echo "[batch] $V seed $S done $(date +%H:%M)" >> logs_qwen_$V.txt
done

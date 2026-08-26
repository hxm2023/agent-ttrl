#!/usr/bin/env bash
if [ -f "$(dirname "$0")/PAUSED" ]; then echo "paused by agent-ttrl (2026-08-24, priority)"; exit 0; fi
# Stage 3 full matrix launcher (runs on autodl2): 2 tasks x 3 estimators x
# 3 seeds = 18 Guard-supervised GRPO runs, sequential (GPU0 trainer only —
# all rollouts use the trainer's own sampler; vLLM is unavailable on this
# server's Blackwell GPU), matched budgets (32 prompts x 8 gens x 3 epochs,
# LoRA rank 16, lr 5e-6). Per run: metrics.json + Guard event/store +
# trajectory records. Each run gets up to 3 attempts (an infra kill must not
# abort the matrix).
#
# Usage (on autodl2): bash stage3/run_matrix.sh [OUT]  (nohup-friendly)
set -uo pipefail
STAGE3=/root/autodl-tmp/agent-ttrl/stage3
OUT=${1:-$STAGE3/out}
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
mkdir -p "$OUT"

# tau2 retail server (needed by the tau2_retail task; single instance)
TAU2_LOG="$OUT/tau2_server.log"
start_tau2() {
  if ! curl -s -m 2 -X POST http://127.0.0.1:8800/reset > /dev/null 2>&1; then
    echo "== starting tau2 server =="
    TAU2_SERVER_PORT=8800 nohup /root/autodl-tmp/appworld-venv/bin/python \
      /root/autodl-tmp/agent-ttrl/scripts/tau2_server.py > "$TAU2_LOG" 2>&1 < /dev/null &
    sleep 8
  fi
  curl -s -m 5 -X POST http://127.0.0.1:8800/reset > /dev/null && echo "tau2 server ok"
}

start_tau2

echo "== stage3 matrix: 2 tasks x 3 estimators x 3 seeds =="
for task in cts_order tau2_retail; do
  for estimator in dense local paired; do
    for seed in 1 2 3; do
      run="$OUT/${task}_${estimator}_s${seed}"
      if [ -f "$run/metrics.json" ]; then
        echo "skip (exists): $run"
        continue
      fi
      for attempt in 1 2 3; do
        echo "== run: $task $estimator seed=$seed (attempt $attempt) =="
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          "$PY" "$STAGE3/train.py" --task "$task" --estimator "$estimator" \
          --seed "$seed" --out "$run" --prompts 32 --gens 8 --epochs 3 \
          > "$OUT/${task}_${estimator}_s${seed}.log" 2>&1
        rc=$?
        if [ -f "$run/metrics.json" ]; then
          echo "== done: $run (rc=$rc) =="
          break
        fi
        echo "== failed (rc=$rc), attempt $attempt of 3 =="
        # clean any GPU residue between attempts
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $pid 2>/dev/null; done
        sleep 5
      done
    done
  done
done
echo "== matrix complete: $OUT =="

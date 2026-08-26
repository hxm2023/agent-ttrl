#!/usr/bin/env bash
if [ -f "$(dirname "$0")/PAUSED" ]; then echo "paused by agent-ttrl (2026-08-24, priority)"; exit 0; fi
# GPU-gated self-healing matrix launcher (runs on autodl2, user-executed or
# watchdog-deployed): waits for the server's other projects (cts_v2_stream /
# agent-ttrl / grpo-guard GPU work) to release the GPUs, then runs the stage3
# matrix; if an external kill interrupts it, the loop restarts it. Runs with
# completed metrics.json are skipped (resume-safe). Ends when 18/18 runs
# have metrics.json.
#
# Deployment: setsid nohup bash stage3/run_gated_matrix.sh \
#   > stage3/gated_matrix.log 2>&1 < /dev/null &
set -uo pipefail
STAGE3=/root/autodl-tmp/agent-ttrl/stage3
OUT=${1:-$STAGE3/out}
GATE_CHECK=300  # seconds between GPU-availability polls

busy() {
  # any foreign GPU compute process (the user's projects use cuda:1 and the
  # trainer uses cuda:0; both show up in nvidia-smi compute-apps)
  local n
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "${n:-0}" -gt 0 ]
}

stable_window() {
  # the GPU must be free for STABLE_POLLS consecutive polls (each GATE_CHECK
  # seconds) before the matrix starts — the user's projects restart
  # intermittently
  local polls=0
  while [ "$polls" -lt 2 ]; do
    if busy; then
      return 1
    fi
    polls=$((polls + 1))
    [ "$polls" -lt 2 ] && sleep "$GATE_CHECK"
  done
  return 0
}

done_count() {
  # count only the 18 real runs (dryruns have metrics.json too but must not
  # count toward completion)
  n=0
  for task in cts_order tau2_retail; do
    for estimator in dense local paired; do
      for seed in 1 2 3; do
        [ -f "$OUT/${task}_${estimator}_s${seed}/metrics.json" ] && n=$((n + 1))
      done
    done
  done
  echo "$n"
}

echo "== gated matrix: waiting for GPU availability =="
while true; do
  n=$(done_count)
  echo "[$(date +%H:%M:%S)] metrics=$n/18"
  if [ "$n" -ge 18 ]; then
    echo "== gated matrix complete: $OUT =="
    break
  fi
  if busy; then
    echo "[$(date +%H:%M:%S)] other project using GPUs; waiting ${GATE_CHECK}s"
    sleep "$GATE_CHECK"
    continue
  fi
  if ! stable_window; then
    echo "[$(date +%H:%M:%S)] GPU not stably free (user project restarts); re-waiting"
    continue
  fi
  echo "[$(date +%H:%M:%S)] GPUs free for a stable window; running matrix (resume-safe)"
  bash "$STAGE3/run_matrix.sh" "$OUT"
  echo "[$(date +%H:%M:%S)] matrix pass finished; re-checking"
  sleep 30
done

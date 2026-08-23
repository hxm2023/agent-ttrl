#!/bin/bash
# Agent-TTRL reproducibility: verify manifests → re-run M4 CPU simulation →
# regenerate figures → compile paper. Exit 0 = everything reproduces.
#
# Note on "checkpoints": this project's artifacts are evidence bundles (run
# manifests under protocols/runs/), not .pt weights — the honest negative
# result means no adapter was committed, so weight persistence is moot.
# Manifest hashing (protocols/GRPO_GUARD_INTEGRATION.md) plays the role of
# checkpoint integrity.
set -euo pipefail
cd "$(dirname "$0")"
FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
pass() { echo -e "${GREEN}[OK]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL=1; }

echo "=== 1/4 run-manifest verification ==="
declare -A MANIFESTS=(
  [m2_baseline]="protocols/runs/m2"
  [m3_cts]="protocols/runs/m3"
  [m4_safecommit]="protocols/runs/M4_stress_simulation.json"
  [m5_appworld]="protocols/runs/m5"
  [m6_tau2]="protocols/runs/m6"
)
for k in "${!MANIFESTS[@]}"; do
  p=${MANIFESTS[$k]}
  if [ -d "$p" ]; then found=$(ls "$p" 2>/dev/null | head -1 | grep -q .; echo $?); else found=$([ -f "$p" ]; echo $?); fi
  if [ "$found" -eq 0 ]; then
    pass "manifests present: $k"
  else
    fail "missing manifests: $k ($p)"
  fi
done

echo "=== 2/4 M4 SafeCommit stress simulation re-run (CPU, deterministic) ==="
python scripts/m4_stress_simulation.py > /tmp/m4_repro.log 2>&1 || true
python - <<'EOF'
import json, subprocess, tempfile, os
try:
    committed = json.load(open("protocols/runs/M4_stress_simulation.json", encoding="utf-8"))
except FileNotFoundError:
    print("[FAIL] M4 manifest missing"); raise SystemExit(1)
gate_rates = {s: committed["streams"][s]["eb_eprocess"]["catastrophic_rate"] for s in committed["streams"]}
# paper claim: zero catastrophic everywhere; benign/mixed commit rates non-degenerate
ok = (all(r == 0.0 for r in gate_rates.values())
      and 0.10 <= committed["streams"]["benign"]["eb_eprocess"]["commit_rate"] <= 0.90
      and 0.10 <= committed["streams"]["mixed"]["eb_eprocess"]["commit_rate"] <= 0.90)
# determinism: re-run must not change the committed manifest
orig = json.dumps(committed, sort_keys=True)
with tempfile.TemporaryDirectory() as td:
    tf = os.path.join(td, "M4_repro.json")
    subprocess.run(["python", "scripts/m4_stress_simulation.py"],
                   cwd=os.getcwd(), capture_output=True, check=True)
    subprocess.run(["cp", "protocols/runs/M4_stress_simulation.json", tf])
    reopened = json.load(open(tf, encoding="utf-8"))
    deterministic = json.dumps(reopened, sort_keys=True) == orig
if not deterministic:
    print("[FAIL] M4 re-run changed the committed manifest (not deterministic)"); raise SystemExit(1)
if ok:
    print("[OK] M4 gate: zero catastrophic-update rate on all 4 streams; commit rates non-degenerate; deterministic re-run")
else:
    print(f"[FAIL] M4 gate rates/commit rates out of spec: {gate_rates}, {commit_rates}"); raise SystemExit(1)
EOF

echo "=== 3/4 figure regeneration ==="
python scripts/make_figures.py > /tmp/figs.log 2>&1
for f in fig1_method fig2_prequential fig3_credit_ablation fig4_safecommit fig5_pareto fig6_heatmap; do
  [ -f "paper/figures/$f.png" ] && pass "figure: $f.png" || fail "missing figure: $f.png"
done

echo "=== 4/4 paper compile ==="
if command -v pdflatex > /dev/null 2>&1; then
  (cd paper && pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1)
  [ -f paper/main.pdf ] && pass "paper/main.pdf compiled" || fail "paper compile failed"
else
  fail "pdflatex not found"
fi

[ $FAIL -eq 0 ] && echo "REPRODUCIBLE: all checks passed" || echo "NOT REPRODUCIBLE: $FAIL check(s) failed"
exit $FAIL

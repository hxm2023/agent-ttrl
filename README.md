> **STATUS (2026-08-24): CURRENT MAIN RESULTS INVALIDATED BY SERVED-POLICY AUDIT** —
> v1 M3/M5/M6 runs never synchronized the LoRA-updated model to the serving
> policy (static served policy + RNG schedule confound). See
> [AUDIT_INVALIDATION.md](AUDIT_INVALIDATION.md) for reason codes; v2
> (policy-consistent EGC-TTRL) is under construction in `src/agent_ttrl/runtime/`.

# Agent-TTRL

Evidence-gated counterfactual test-time RL for stateful tool-using agents under
partial evidence, with strict prequential evaluation. Research scaffold + full
pipeline (M0-M6) with an honest negative result at current scales. Target venue:
ACL 2027 (fallback NeurIPS/ICML 2027). See `Agent-TTRL_顶会导向详细项目设计方案.md`
for the authoritative design input and `CLAUDE.md` for the locked protocol.

## Quick start

```bash
pip install -e .[dev]
pytest            # 127+ tests: schemas, CTS golden pack, stats coverage
bash reproduce.sh # manifest verification + M4 simulation re-run + figures + paper
```

## Structure

```
src/agent_ttrl/      # protocol machinery: schemas, cost ledger, CTS, SafeCommit gate
scripts/             # M2-M6 experiments (deploy scripts for autodl2)
protocols/runs/      # all run manifests (evidence bundles, hashed)
paper/               # ACL 2027 draft, 6 figures, references
portfolio/           # resume material
```

## Protocol red lines (locked)

1. Evidence tiers: E_hard/E_soft enter adaptation; R_hidden (hidden evaluator)
   never enters rollout/branch/gradient/commit-gate.
2. Primary metric: inductive future transfer (prequential AUPC), first-attempt
   scores only; within-task/transductive are supplementary.
3. 3-channel budgets (B_env/B_model/B_update) via one canonical ledger; matched
   caps for baselines.
4. Policy identity binding; updates only at episode boundaries; LoRA only.
5. GRPO-Guard correctness Gate passed 2026-08-22 (contract 24/24; see
   `protocols/GRPO_GUARD_INTEGRATION.md`) — formal results allowed only after it.
6. ≥5 seeds pre-registered; honest negative results are valid results.

## Key numbers (protocols/runs/, 2026-08-23)

- SafeCommit (empirical-Bernstein e-process gate): catastrophic-update rate
  relative reduction 100% vs always-commit on benign/mixed/poisoned/abrupt-shift
  candidate archives, commit rates non-degenerate (0.11/0.10 benign/mixed).
- Deployment-period LoRA-RL: no stable prequential gain across
  CTS/AppWorld/tau2 × Qwen3-4B/Mistral-7B at 8-16 task scale (honest negative).

## Compute

Experiments ran on autodl2 (2×RTX 6000D 84GB), shared with GRPO-Guard per the
shared-card rules in `CLAUDE.md` (low priority `nice -n 10`, staggered phases,
"parallel-with" recorded in manifests).

# RESEARCH_CONTRACT — Agent-TTRL (frozen 2026-08-22, M0)

> This contract binds the project's protocol red lines into a machine-checkable
> form. Any deviation must be a new frozen version with a new protocol hash —
> never an in-place edit after test data is seen.

## 1. Research question (locked, design doc §0/§5.1)

> 部署期连续遇到相关但未见过的状态化工具任务时,Agent 能否只利用当时可获得的、
> 部分可靠的执行证据,进行 session-scoped 参数更新;并在固定交互预算下提升后续任务
> 表现,同时控制错误奖励强化、负迁移和灾难性更新?

## 2. Method (locked after Phase 1; see phase01/CANDIDATE_KILL_LOG.md)

**EGC-TTRL** — two-scale evidence gating:
- **Local gate (credit)**: signed action credit from CRN-coupled paired counterfactual
  branches at selected decision states; gated by t-interval reliability OR
  evidence-conflict (E_hard↔E_soft disagreement) — 2×2 frozen by CTS exact-oracle
  fidelity at decision pilot.
- **Global gate (commit)**: **empirical-Bernstein e-process** at α_total=0.05,
  ε_gain=0.01, ε_harm=0.10, n=512 (FROZEN by coverage simulator 2026-08-22, D6;
  sweep at protocols/sweep_coverage_results.json); atomic commit/rollback with
  policy identity binding.
- Fallback F (C2-only safe deployment-time policy improvement) if the counterfactual
  headline dies under matched controls.

## 3. Evidence tiers (red line 1)

| Tier | May enter adaptation | Examples |
|---|---|---|
| E_hard | YES | schema, API return, permissions, policy rules, state invariants, receipts |
| E_soft | YES (calibrated) | process/result verifier (calibrated on dev) |
| R_hidden | NEVER | benchmark hidden evaluator, test answers |

`R_hidden` never enters rollout selection / branch selection / gradient / commit gate /
hyperparameter selection. Violation ⇒ claim INVALID.

## 4. Performance classes (red line 2)

1. within-task recovery — supplementary
2. transductive adaptation — supplementary
3. **inductive future transfer (prequential)** — PRIMARY: `AUPC_prequential` +
   `sealed_future_holdout_score`

## 5. Budgets (red line 3)

- Three channels: B_env / B_model / B_update; one canonical CostLedger; per-op billing;
  caps pre-registered; no cross-channel exchange; wall-clock/GPU-h = sensitivity only.
- Baselines get the same caps ("best config within cap").

## 6. Policy identity (red line 4)

- rollouts bound to (base_sha256, adapter_sha256, policy_version)
- updates only at episode boundaries; adaptation_scope=domain_session;
  reset_unit=domain_seed; base frozen; LoRA only.

## 7. Dependencies / gates (red lines 5-7)

1. **GRPO-Guard correctness Gate** MUST pass before formal result experiments (M3+).
   Integration contract: protocols/GRPO_GUARD_INTEGRATION.md.
2. **M0 gate** before any GPU: schemas ✓, CTS golden pack ✓, tau2 manifests ✓,
   baseline registry ✓, profile ✓ (M0_DECISION.json: PARTIAL_PASS — remaining items
   are external, not method validity).
3. Statistics: ≥5 seeds (blinded power analysis, frozen once), pre-registered tests
   p<0.01 + effect size/CI, paired hierarchical bootstrap, prequential curves.

## 8. Forbidden project forms (design doc §1.2)

TTRL-only reproduction on math · history-in-prompt as "test-time RL" · Best-of-N as
"online learning" · hidden-evaluator-as-reward · same-task-only reporting ·
asymmetric budget accounting · single-seed conclusions · gain-from-more-tokens
explanations.

## 9. Kill conditions (K1-K5) & falsification

- K1: post-lock work covering the combination cell (deployment LoRA-RL × partial
  evidence × signed counterfactual credit × statistical commit gate × prequential
  transfer) ⇒ pivot.
- K2: gain from more tokens/branches alone (equal-extra-rollout / random-branch controls).
- K3: no causal/statistical interpretation of the credit signal (credit-fidelity
  diagnostic must predict prequential gain).
- K4: fails matched-budget comparison.
- K5: no agent-specific mechanism (works equally on math).
- Full falsification conditions: phase01/PHASE1_DECISION.md.

## 10. Frozen artifacts (hashes)

| Artifact | Location | Status |
|---|---|---|
| design doc | Agent-TTRL_顶会导向详细项目设计方案.md | frozen 2026-08-22 (tracked) |
| claim matrix | novelty/claim_matrix.csv | 2026-08-22 |
| Phase 0-1 logs | phase01/ | 6 files, 2026-08-22 |
| implementation profile | protocols/M0_IMPLEMENTATION_PROFILE.yaml | TEMPLATE→FROZEN (REQUIRED_* resolved for models) |
| schemas | schemas/*.schema.json | 7 schemas, 44 contract tests |
| CTS fixtures | benchmarks/controlled_tool_shift/fixtures.py | golden pack green |
| tau2 manifests | protocols/splits/tau2_*_roles.json | source_commit a2c0247 |
| gate sweep | protocols/sweep_coverage_results.json | D6 freeze evidence |
| Guard contract | protocols/GRPO_GUARD_INTEGRATION.md | pending Guard Gate |

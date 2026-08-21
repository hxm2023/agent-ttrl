# IDEA_REPORT — Agent-TTRL Phase 1 Outcome (2026-08-22)

## Locked Idea

**EGC-TTRL (Evidence-Gated Counterfactual Test-Time RL), revised by Phase 1 debate**:
deployment-period, session-scoped LoRA-RL for stateful tool agents under partial evidence,
with two-scale evidence gating:

1. **Local gate (credit)**: signed action credit from CRN-coupled paired counterfactual
   branches at selected decision states; gated by (M1) t-interval reliability OR
   evidence-conflict (E_hard↔E_soft disagreement) — 2×2 frozen by exact-oracle fidelity.
2. **Global gate (commit)**: statistical commit gate over paired shadow gain + anchor harm
   (M2: fixed-n Hoeffding α_k vs PACE-style e-process, frozen by coverage simulator);
   atomic commit/rollback with policy identity binding.

**Primary metric**: AUPC_prequential (first-attempt hidden outcome on tasks never entering
update/selector/gate) + sealed future holdout. **Budget**: 3-channel hard caps with one
canonical ledger. **Environments**: ControlledToolShift (mechanism) → AppWorld (primary)
→ τ² retail/telecom (second). **Models**: Qwen3-4B-Instruct-2507 (primary),
Mistral-7B-Instruct-v0.3 (second). **Venue**: ACL 2027 primary (user decision), fallback
NeurIPS/ICML 2027.

## Why this idea (evidence chain)

- Phase 0 re-sweep: 30 papers full-text; every design-doc §3 work verified; ≥20 additional
  works mapped (novelty/claim_matrix.csv).
- Individual cells (counterfactual credit: CVT-RL/Tree-RL/APPO/CRAFT/BiPACE/ReBel;
  commit gates: PACE/VaG/Drift2Act/STABLE; deployment updates: aTTT/StarOR/JitRL) are
  contested at training-time/prompt-level/per-instance — but the deployment-time
  partial-evidence closed loop with prequential future-transfer evaluation is clean
  (zero hits at lock; re-sweep again before Phase 2).
- Design doc's own warning honored: novelty headline shrank from "first counterfactual
  credit / first commit gate" to the deployment×partial-evidence combination + strict
  prequential protocol; matched controls pre-committed (B05 CausalFlow, C08 CVT-RL-adapted,
  G02 PACE-adapted).

## Verdicts (CANDIDATE_KILL_LOG.md)
A=27/35 WINNER · B=25/35 component (M1) · F=25/35 fallback · C=23/35 component (M4) ·
E=21/35 component (M3) · D=19/35 killed (saturated cell) · G=19/35 killed (coherence).

## Falsification conditions (binding)
1. Post-lock coverage of the combination → K1 → pivot.
2. Matched counterfactual controls tie C1 → drop headline → F (safe deployment-time
   policy improvement).
3. Credit fidelity ∤ prequential gain → mechanism claim shrinks.
4. SafeCommit ≈ always-rollback → C2 fails.
5. Coverage simulator decides gate variant pre-lock (no test-after choice).

## Entry to Phase 2
Per design doc execution order: M0 (CPU) → GRPO-Guard Gate (external) → CTS → baselines →
EGC credit single-mechanism → SafeCommit single-mechanism → AppWorld + τ² → second model
family + full stats. Red lines: no GPU before M0; no formal results before Guard Gate;
≥5 seeds pre-registered; R001-R003 before public-env runs.

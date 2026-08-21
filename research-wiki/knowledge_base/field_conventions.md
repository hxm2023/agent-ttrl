# Field Conventions: Test-Time RL for Stateful Tool Agents

## Plot Types

- **Prequential learning curves**: task-index (x) vs first-attempt success (y), per seed;
  the field standard for "does the agent improve across the stream" (SEA-Eval, EvoTest,
  EvoPolicyGym all use episode/task-indexed curves). Our primary figure.
- **Grouped bar with error bars (hierarchical bootstrap CI)**: method × environment × model
  main tables; error bars over seeds (outer) and task families (inner).
- **Pareto success–cost surfaces**: success vs tokens/tool-calls/GPU-h, one point per method
  per budget cap (design doc §10.5); used to rebut "gain from more compute".
- **Risk/coverage curves**: selective-risk vs coverage for verifier calibration (design doc
  §6.4); expected calibration error / Brier / AUROC reported alongside.
- **Credit-fidelity scatter/rank plots**: estimated credit vs exact oracle contribution per
  decision state (controlled env); sign accuracy + rank correlation.
- **Commit timelines**: stream position vs commit/rollback events with adapter hashes
  (auditability); catastrophic update markers.
- **Statistical checks**: p-value distributions / bootstrap CI coverage calibration plots
  (from pre-registered tests).

## Notation Conventions

- Policy: π_{θ,φ_k}(a_t|h_t) — frozen backbone θ, session LoRA adapter φ_k; rollout bound to
  (base_sha256, adapter_sha256, policy_version).
- Evidence: e(τ) = [e_schema, e_tool, e_policy, e_state, e_user, e_soft]; E_hard/E_soft enter
  adaptation, R_hidden never. Utility g(e) = wᵀe − λ_c c(τ), λ_c=0 in primary.
- Credit: ĉ_i = Ū_i − mean_j(Ū_j) (group-relative signed credit); reliability gate
  L_i/U_i = ĉ_i ± t_{R-1,0.90}·sqrt(v̂_i/R); α_i = 1{L_i > η ∨ U_i < −η}.
- Prequential outcome: Y_k^pre = R_hidden(τ_k^first); AUPC_prequential = (1/N)Σ Y_k^pre.
- Budget: C = (N_env, N_model_tok, N_update_tok) ⪯ B; one canonical ledger event per op.
- Commit gate: LCB_gain ≥ ε_gain AND UCB_harm ≤ ε_harm AND GuardDecision==ALLOW.
- Branch protocol: G=4 actions × R=4 continuation seeds, common random numbers, snapshot
  restore, inclusion prob q_t (no IPW in v0.1).

## Paper Structure (agent test-time learning, main-conference style)

- Typical sections: Intro → Related Work (dual-domain: TTA/TTT + agent RL) → Problem
  (POMDP stream, evidence tiers) → Method → Protocol (splits, budgets, statistics) →
  Experiments (correctness → main → mechanism → safety) → Limitations → Artifacts.
- Length: 8–10 pages + appendix (protocol schemas, fault matrix, full ledgers).
- Figure count: 4–6 main figures + tables; main tables must be reconstructible from raw
  manifests (paper artifact tests).
- Venue norms: ACL-style prefers language-agent framing + evaluation rigor; ICML/NeurIPS
  prefers algorithm + statistics. ACL 2027 is our primary (user decision 2026-08-22).

## Terminology Pitfalls

- Do NOT call history-in-prompt "test-time RL"; do NOT call Best-of-N "online learning";
  do NOT call hidden-evaluator-as-reward "label-free".
- "Adaptation" ≠ "training" ≠ "RL": parameter/activation change vs interaction-reward-driven
  policy change (design doc §2.1 table).
- Prequential ≠ transductive ≠ within-task recovery: report separately (design doc §2.4).
- Sentinel data is online model-selection data — never doubles as unbiased eval set.
- Always report which evidence entered the gradient vs which was evaluation-only.

## Reproducibility Conventions (project, from design doc §16-18)

- Immutable implementation profile (Python 3.11.11, PyTorch 2.7.1+cu126, Transformers
  4.53.2, PEFT 0.16.0, vLLM 0.10.0, native clipped action-token objective — NOT TRL/verl as
  semantic owner).
- Six JSON Schemas (online_stream / gate_manifest / sealed_audit / evidence_bundle /
  branch_record+update_row / candidate_adapter_decision) validated by positive+negative fixtures.
- Fault manifest F01–F12 with reason codes; property tests (branch determinism, mask scope,
  rollback logits tolerance, ledger conservation).
- Run IDs: ATTRL-{M}-{env}-{model}-{method}-{seed}-{rev}; only COMPLETED+VALID runs enter
  formal summaries; INVALID never mixed into means.

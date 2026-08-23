# Baseline Sources

Provenance of every baseline/method family compared in the paper. Contract per
design doc §17.2: pin an official commit, or record a faithful-reimplementation
hash; a work without usable code gets a signed NOT_COMPARABLE_REPORT, never
silent omission. See `protocols/baseline_registry.yaml` for the machine-readable
registry.

## Method families actually run (M2-M6)

| Method | Where implemented | Provenance | Notes |
|---|---|---|---|
| frozen (no update) | `scripts/m3_stream_pilot.py`, `scripts/tau2_agent_stream.py` (variant=frozen) | n/a (our protocol baseline) | First-attempt scores without any adaptation; matched budget baseline |
| best-of-n | `scripts/m2_baselines.py` (best_of_n) | Standard inference-time scaling; no parameter update (design doc §1.2: BoN is a matched-cost baseline, not "online learning") | Reports first-attempt AUPC under the same B_model token cap |
| reflexion | `scripts/m2_baselines.py` (reflexion) | Shinn et al. 2023 (Reflexion) — verbatim textual self-feedback | Context-level baseline; no parameter update |
| hard verifier | `scripts/m2_baselines.py` (hard_verifier) | Our verifier-only probe (E_hard receipts) | Context-level baseline |
| naive LoRA-RL | `scripts/tau2_agent_stream.py` (variant=naive) | Our update baseline: REINFORCE-style objective (L = -adv · mean completion-token logp) with group-relative utility advantage; LoRA r=16 on all linear modules; ≤4 sequential steps on the same materialized rollouts (no importance ratio / KL — documented simplification of the R002 Guard-validated clipped-GRPO chain) | Parameter update; no evidence gating |
| egc (evidence-gated credit) | `scripts/tau2_agent_stream.py` (variant=egc) | Our mechanism (H1): naive + reliability gate zeroing \|z\| < 0.5 credits. NOTE: a pre-audit bug gated the whole update block on `variant=="naive"`, making egc a frozen-equivalent no-op in the first tau2 factorial; fixed 2026-08-23 and re-run | Parameter update with evidence gating |

## Families registered but NOT run (signed not-comparable)

| Method | Paper | Reason |
|---|---|---|
| ACE | arXiv 2510.04618 | Official repo required; registered REQUIRED_COMMIT, not compared in this study (matched context-level baselines already cover the "no parameter update" cell; see EXPERIMENT_DECISION_LOG.md D12) |
| GTTA | arXiv 2511.04847 | WebArena-oriented; requires its training pipeline; registered not-comparable (D12) |
| OLIVIA / MemoPilot / LEAFE / CausalFlow | arXiv 2605.11169 / 2606.08656 / 2603.16843 / 2605.25338 | No usable official code at study time; registered REQUIRED_COMMIT_OR_NOT_COMPARABLE (D12) |

## Budget fairness

All variants in a comparison run under the same pre-registered caps: same
B_model token cap for rollouts, same number of first attempts, hidden evaluator
(R_hidden) never accessed by any variant. The 3-channel ledger
(B_env/B_model/B_update) is recorded per run in `protocols/runs/` manifests.

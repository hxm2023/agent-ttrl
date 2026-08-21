# PHASE0_DECISION — Re-sweep Verdict (2026-08-22)

- **Decision**: The design-doc §3 map is verified (all 11 works exist and are characterized
  correctly) but **incomplete**: the re-sweep surfaced ≥20 additional 2025-2026 works that
  occupy the individual mechanism cells of EGC-TTRL (C1 counterfactual credit, C2 commit
  gate). The **combination cell** (deployment-stream LoRA-RL × partial/conflicting evidence
  × paired counterfactual signed credit × risk-controlled commit × inductive prequential
  transfer, for stateful tool agents) is **not covered by any single work found**.
- **Evidence**: 30 papers downloaded + full-text indexed (research-wiki/); claim matrix at
  novelty/claim_matrix.csv. Closest per cell:
  - C1 cell: CVT-RL (2606.05263, training-time PCCC counterfactual credit + validity gating),
    Tree-RL/SHEPHERD (2605.10913, training-time fork advantage), APPO (2606.12384), CRAFT
    (2606.29476), BiPACE (2606.25556), CausalFlow (2605.25338, offline repair).
  - C2 cell: PACE (2606.08106, anytime-valid e-process commit gate, prompt-level), VaG
    (2608.05810, verifier pre-commit gating), Drift2Act (risk certificates), STABLE
    (2510.16089, gated LoRA merge).
  - Deployment-parameter cell: aTTT (2607.03441, in-episode LoRA TTT — no RL, no gate),
    StarOR (2606.15197, per-instance test-time LoRA-GRPO in optimization — no stream/prequential,
    no gate), JitRL (2601.18510, training-free).
  - Prequential cell: SEA-Eval (2604.08988), EvoTest/J-TTL (2510.13220), EvoPolicyGym (2607.02440).
  - Same-venue near-miss: SAGE (ACL 2026, skill-library GRPO on AppWorld).
- **Alternative considered**: kill/pivot EGC-TTRL entirely because each pillar is individually
  contested. **Why not**: the design doc's own Go/Pivot table allows GO when no work covers
  both C1 and C2 *and* the components solve one bottleneck; the sweep shows C1/C2 are each
  covered only in restricted forms (training-time / prompt-level / per-instance), never the
  deployment-time partial-evidence combination. BUT the novelty headline must shrink from
  "first counterfactual credit / first commit gate" to "the deployment-time partial-evidence
  closed loop with strict prequential protocol".
- **Compute spent**: 0 GPUh (desk work; ~30 web searches + 30 paper downloads + KB).
- **Falsification**: any single work (post-2026-08-22) that performs deployment-period
  LoRA-RL for stateful agents using partial-evidence-only signed branch credit AND a
  statistical commit gate AND reports inductive future transfer → K1 → pivot. Also: if
  Phase 1 debate shows C1+C2 cannot be one-bottleneck-joined, contract per §4.3 table.
- **Next**: Phase 1 — candidate generation ≥5 (≥3 fresh), debate, scoop check, adversarial
  gate (R1-R7), decision logs, IDEA_REPORT.

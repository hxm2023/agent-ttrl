# CANDIDATE_KILL_LOG — Agent-TTRL Phase 1 (2026-08-22)

Protocol: ≥5 candidates, ≥3 fresh; every candidate gets a scored verdict + K1-K5 check +
decision + evidence + alternative + falsification condition. No prior probability at
generation; verdicts only after evidence. Kill ≠ waste: killed candidates become
baselines/ablation rungs. Scoring: 7 dims × 1-5 = /35 (Novelty, Scientific value,
Mechanism coherence, Compute feasibility, Publishability, Job relevance, Risk-inverse);
Novelty ≥4 required for WINNER contention.

---

## Candidate A — EGC-TTRL (two-scale evidence gating; design-doc H1)

**Mechanism**: matched counterfactual branches (G=4 actions × R=4 CRN continuation seeds,
from a critical-decision selector) → reliability-gated signed credit (ĉ_i = Ū_i − group mean,
t-interval gate η_credit) → action-token LoRA-GRPO (clipped, KL-to-parent) → SafeCommit
(shadow gain + anchor harm, fixed-n Hoeffding with cross-candidate α_k allocation).

**Scores**: Novelty 3 · Scientific value 5 · Coherence 4 · Compute 3 · Publishability 4 ·
Job relevance 5 · Risk-inverse 3 → **27/35**.

**K1-K5**: K1 PASS (no work covers the deployment×partial-evidence×signed-credit×commit×
prequential combination; individual cells contested — see below). K2 PASS-conditional
(equal-extra-rollout + random/unpaired branch controls pre-registered). K3 PASS-conditional
(estimand C_μ on selected decision states; local gate is heuristic only — no coverage claim;
global gate has coverage claim). K4 PASS-conditional (3-channel ledger, same caps for all).
K5 PASS (E_hard state invariants / collateral damage / irreversible actions are agent-specific;
math has none).

**Decision**: **WINNER (provisional, with mandatory modifications M1-M4 below)**
**Evidence**: re-sweep to 2026-08-22 — see novelty/claim_matrix.csv + KB. Closest C1-cell:
CVT-RL 2606.05263 (training-time PCCC, full verifiable reward, no commit gate), Tree-RL
2605.10913 (training-time fork advantage), APPO 2606.12384, CRAFT 2606.29476, BiPACE
2606.25556, ReBel 2605.20061 (belief-consistency credit, training-time), RRPO/SRPO
2605.30227 (reset-based counterfactuals, training-time). Closest C2-cell: PACE 2606.08106
(prompt-level e-process commit gate), VaG 2608.05810 (skill-level pre-commit), Drift2Act
(risk certificates), STABLE 2510.16089 (gated LoRA merge, continual learning). Closest
deployment-parameter cell: aTTT 2607.03441 (in-episode LoRA TTT, no RL/no gate/no prequential),
StarOR 2606.15197 (per-instance LoRA-GRPO, optimization, no stream), JitRL 2601.18510
(training-free). Closest prequential cell: SEA-Eval 2604.08988, EvoTest 2510.13220.
Same-venue near-miss: SAGE (ACL 2026).

**Mandatory modifications (from debate)**:
- **M1 (adopt B as local-gate variant)**: local credit gate compared as 2×2 — t-interval
  reliability gate (design doc) vs evidence-conflict gate (Candidate B). Pre-register both;
  winner determined by CTS exact-oracle credit fidelity, then frozen.
- **M2 (e-process global gate)**: PACE showed anytime-valid e-process gates dominate
  fixed-threshold acceptance; run coverage simulator on (i) fixed-n Hoeffding + α_k
  allocation (design doc) vs (ii) testing-by-betting e-process; whichever passes coverage
  with better power becomes primary, other is variant.
- **M3 (CRN formalization, from E)**: branch protocol formalized as CRN estimator with
  variance-reduction analysis (extends Maliakkal et al., RLJ 2606.04732 to deployment-time
  LLM agent credit); statistics section in paper, no headline change.
- **M4 (interference pre-screen, from C)**: optional cheap pre-screen before shadow eval
  (gradient-interference vs anchors); reported as efficiency variant, never a replacement
  for the shadow gate in primary analysis.

**Alternatives considered**: B (conflict-credit) as headline — rejected because it covers
only the credit-quality bottleneck; it cannot detect negative transfer from *reliable*
evidence (anchor harm), which is C2's job. F (C2-only contraction) as headline — kept as
fallback per §4.3 row 2 (if adversarial gate kills the counterfactual headline, the paper
becomes "safe deployment-time policy improvement").

**Falsification of this decision**: (1) Any new single work (post-2026-08-22) covering
deployment-stream LoRA-RL + partial-evidence signed credit + statistical commit gate +
inductive prequential transfer → K1 → pivot. (2) If matched-cost CVT-RL-style or
Tree-RL-style controls (adapted to deployment) match A on AUPC_prequential AND credit
fidelity does not predict prequential gain → delete counterfactual-credit headline, go F.
(3) If SafeCommit commit rate ≈ 0 or only helps by refusing updates → C2 fails → F.

---

## Candidate B — Evidence-Conflict Credit (E_C: cross-tier disagreement as the credit gate)

**Mechanism (fresh)**: signed credit is emitted only when E_hard (state invariants,
receipts) and E_soft (calibrated verifier) agree; disagreement → abstain from gradient AND
increment a drift counter that can halt further adaptation (domain-level alarm). One
mechanism, one bottleneck (uncertain evidence must not change policy).

**Scores**: Novelty 4 · Scientific value 3 · Coherence 5 · Compute 4 · Publishability 3 ·
Job relevance 3 · Risk-inverse 3 → **25/35**.

**K1-K5**: K1 PASS-conditional (cross-tier conflict as credit gate is open in sweep; DAC
2606.10684 uses single-verifier abstention in multi-agent QA, not cross-evidence-tier
conflict in stateful tool envs; no deployment version found). K2 PASS (no extra rollout
needed). K3 PASS (disagreement is a well-defined signal). K4 PASS. K5 PASS (state
invariants vs verifier conflict is agent-specific).

**Decision**: **KILL-as-core → COMPONENT (absorbed into A as local-gate variant M1)**
**Evidence**: scoop check: "verifier disagreement abstain agent credit" → DAC 2606.10684
(evidence-sufficiency abstention as reward, QA search agents) is the only partial hit; no
work gates credit by E_hard/E_soft conflict. Design-doc fixture CTS-F08 (poisoned success
receipt vs DB invariant) is exactly this conflict.
**Alternative**: keep B standalone — rejected: narrower than A (no anchor-harm control),
weaker job-fit, and the same benchmark environments are needed anyway.
**Falsification**: if in A's 2×2 the conflict gate does not beat the t-interval gate on
CTS credit fidelity (≥10pp sign accuracy), B is dead even as component → pure diagnostic.

---

## Candidate C — Interference-Gated Commit (gradient-diagnostic SafeCommit)

**Mechanism (fresh)**: commit decision via gradient-interference diagnostics between the
candidate LoRA update and anchor-capability gradients (negative cosine → rollback), as a
cheap B_model-only pre-screen before (or replacement for) shadow evaluation.

**Scores**: Novelty 3 · Scientific value 3 · Coherence 3 · Compute 4 · Publishability 3 ·
Job relevance 3 · Risk-inverse 4 → **23/35**.

**K1-K5**: K1 RISK (the mechanism is borrowed from continual learning: Rosetta MAOP
2607.00293, igfa 2607.09202 similarity-gated share-or-orthogonalize, SLICE 2605.12752,
DOC, OrthoSkillVLA 2608.19589 all own gradient-interference gating/projection in CL).
K2 PASS. K3 PASS. K4 PASS. K5 WEAK (interference diagnostics are model-generic).

**Decision**: **KILL-as-core → COMPONENT (M4: cheap pre-screen variant; also informs
anchor-gradient instrumentation)**
**Evidence**: scoop check "gradient interference commit gate adapter" → the CL cell is
saturated; the deployment-time *commit-gate* use is fresh but reviewers would answer
"use orthogonal projection instead of detection" — a losing argument.
**Alternative**: adopt igfa-style projection to *prevent* interference — rejected: that
changes the problem (we are not protecting a lifelong model; we are gating session
adapters), and design doc forbids accumulating extra modules.
**Falsification**: if M4's pre-screen does not save ≥20% shadow-eval budget at equal
catastrophic-update rate, it is dropped from the paper (appendix only).

---

## Candidate D — Plan-Level (Strategy-Token) Test-Time RL

**Mechanism (fresh)**: LoRA updates on plan/strategy token spans (goal decomposition,
planning) rather than action tokens; hypothesis: plan-level updates induce task-family
abstraction → better inductive transfer, less same-task memorization.

**Scores**: Novelty 2 · Scientific value 3 · Coherence 3 · Compute 4 · Publishability 2 ·
Job relevance 3 · Risk-inverse 2 → **19/35**.

**K1-K5**: **K1 FAIL-ADJACENT**: the plan-level credit cell is saturated at training time:
HiPER 2602.16165 (hierarchical plan-execute RL, HAE), Preplan-and-Anchor 2510.13554
(token-level plan/anchor advantages), DARS 2608.20161 (dual-level planner credit), StraTA,
HiMAC, GEAR. A test-time plan-level variant is an incremental re-skin.

**Decision**: **KILL** (K1-adjacent; mechanism cell saturated)
**Evidence**: scoop check "plan-level strategy token credit assignment" → above 6 works.
**Alternative**: none needed.
**Falsification**: if kept, it would fail R5 ("would work equally on math") — plan tokens
exist in CoT math too.

---

## Candidate E — CRN-Paired Evidence-Branch Estimator (variance theory)

**Mechanism (fresh)**: formalize common-random-numbers paired counterfactual branches for
deployment-time LLM tool agents; variance-reduction analysis + finite-sample signed-credit
bounds under partial evidence; deployment-time CRN coupling protocol (incl. user-simulator
coupling in τ²).

**Scores**: Novelty 2 · Scientific value 3 · Coherence 4 · Compute 4 · Publishability 2 ·
Job relevance 3 · Risk-inverse 3 → **21/35**.

**K1-K5**: K1 RISK: Maliakkal et al. "Using Common Random Numbers for Simulation-based
Planning with Rollouts" (RLJ 2605.04732) already proves CRN variance reduction for MDP
planning and names LLM/deep-RL as future work; our version would be a direct extension.
K2-K5 PASS.

**Decision**: **KILL-as-core → COMPONENT (M3: formal statistics section + branch protocol
of A; also BV-Blend 2606.28707 / BASIS / Kernelized-AE 2604.28005 as advantage-variance
related work)**
**Evidence**: scoop check "common random numbers paired rollouts variance reduction" → RLJ
2605.04732 (formal base), BV-Blend (group-normalization collapse fix), BASIS, Kernelized AE.
**Alternative**: standalone theory paper — rejected (thin, incremental, no job fit).
**Falsification**: if CRN coupling cannot be maintained in τ² (non-replayable user
simulator), the CRN claim is restricted to controlled envs + AppWorld, and τ² reports
unpaired sensitivity.

---

## Candidate F — Safe Deployment-Time Policy Improvement (C2-only contraction)

**Mechanism (design-doc contraction)**: drop counterfactual-credit headline; deployment-time
LoRA-RL from partial evidence (naive/T3RL-style rewards) + risk-controlled commit
(shadow gain + anchor harm) as the sole contribution; prequential primary metric.

**Scores**: Novelty 3 · Scientific value 4 · Coherence 4 · Compute 4 · Publishability 3 ·
Job relevance 4 · Risk-inverse 3 → **25/35**.

**K1-K5**: K1 PASS-conditional (parameter-level commit in a deployment RL loop with anchor
retention is open; PACE is prompt-level/offline, VaG is skill-level, Drift2Act is
TTA-risk-level). K2 PASS. K3 PASS. K4 PASS. K5 PARTIAL (anchor retention is agent-relevant
but mechanism-generic).

**Decision**: **CONDITIONAL-KEEP as fallback** (design doc §4.3 row 2: if C1's
counterfactual headline dies under matched controls or new coverage, the paper becomes F).
**Evidence**: PACE/VaG/Drift2Act own the gate concept; F's delta is only the setting
(parameter-level, RL loop, partial evidence, anchor retention) — sufficient for a solid
but not a flagship paper.
**Alternative**: full EGC-TTRL (A) — preferred while its falsification conditions hold.
**Falsification**: F is adopted iff (i) A's C1 fails matched controls, or (ii) a post-lock
work covers A's combination. F itself fails if a post-lock work covers "parameter-level
commit gate + partial evidence + prequential" in stateful agents.

---

## Candidate G — Budget-Constrained Branch Allocation (meta-policy over the stream)

**Mechanism (fresh)**: learn where/when to spend the branch budget (B_env/B_model) as an
online meta-policy maximizing prequential AUPC under the 3-channel caps.

**Scores**: Novelty 3 · Scientific value 3 · Coherence 2 · Compute 3 · Publishability 2 ·
Job relevance 3 · Risk-inverse 3 → **19/35**.

**K1-K5**: K1 PASS-conditional (APPO 2606.12384 selects branch points at training time;
deployment-time budget-constrained allocation not found). K2 PASS. K3 WEAK. K4 PASS.
K5 WEAK. **Coherence FAIL**: a learned meta-controller stacked on the credit+commit loop is
exactly the "system collage" the design doc forbids (§1.3, §3.2 test 1).

**Decision**: **KILL** (coherence violation)
**Evidence**: design doc §1.3/§3.2; APPO covers branch-selection cell at training time.
**Alternative**: keep selector as a dev-frozen heuristic (design doc §7.2 selectors) — the
lightweight version, no meta-RL.
**Falsification**: n/a (killed on coherence; if A's mechanism tests later show selector
quality dominates credit quality, revisit as a dev-frozen selector study only).

---

## Summary Table

| ID | Candidate | Score | Decision | Fate |
|----|-----------|-------|----------|------|
| A | EGC-TTRL (two-scale evidence gating) | 27/35 | WINNER (provisional) | primary; M1-M4 applied |
| B | Evidence-conflict credit gate | 25/35 | KILL-as-core → COMPONENT | local-gate variant (M1) |
| C | Interference-gated commit | 23/35 | KILL-as-core → COMPONENT | pre-screen variant (M4) |
| D | Plan-level test-time RL | 19/35 | KILL | appendix ablation rung (plan-token credit contrast) |
| E | CRN paired-branch estimator | 21/35 | KILL-as-core → COMPONENT | statistics section + protocol (M3) |
| F | Safe deployment-time policy improvement | 25/35 | CONDITIONAL-KEEP | fallback per §4.3 |
| G | Budget-constrained branch allocation | 19/35 | KILL | selector stays dev-frozen heuristic |

**Fresh candidates**: B, C, D, E, G = 5 (≥3 required ✓). Design-doc-derived: A, F.
**Ablation ladder pre-designed from killed candidates**: local-gate 2×2 (t-interval vs
conflict, from B), global-gate e-process variant (from PACE, M2), interference pre-screen
(from C), plan-token credit contrast (from D), unpaired/random branch (from E's protocol
contrast). Re-sweep at candidate lock, after adversarial gate, and before Phase 2 (K1 watch).

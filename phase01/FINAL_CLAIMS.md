# FINAL_CLAIMS — Agent-TTRL (preliminary, Phase 1; NOT yet evidenced)

Status: candidate claims for pre-registration. No number in this file is a result.
Claim wording follows design doc §5.2/5.3 + Phase 1 debate modifications (M1-M4).

---

## EXECUTED REALITY (updated 2026-08-23, after M1-M6; supersedes where in conflict)

C1 (evidence-grounded action credit → prequential gain) is **NULL at the scales
run** — no stable inductive-future-transfer gain from deployment-period LoRA-GRPO
across CTS/AppWorld/tau2 × Qwen3-4B/Mistral-7B (EXPERIMENT_DECISION_LOG.md D12,
D15-D17; paper 05_results.tex). The mechanism machinery (paired-branch credit,
reliability gate, R003 sign conservation) was validated in isolation but its
preqential-gain claim was not escalated. Accordingly, C2's "retaining ≥80% of
C1's prequential gain" is vacuous against a null C1 and is dropped from the
paper claims.

C2's gate winner was **frozen by the coverage simulator as empirical-Bernstein
e-process** (D6; fixed-n Hoeffding passed NO operating point, D6 detail). The
"fixed-n Hoeffding α_k or PACE-style e-process" alternation above is historical.

Final claims (paper, honest synthesis): (a) protocol machinery validated
end-to-end (evidence tiers, prequential primary metric, 3-channel ledger,
policy identity binding, guarded rollout chain, GRPO-Guard Gate 24/24); (b)
SafeCommit EB e-process gate: catastrophic-update rate relative reduction 100%
vs always-commit, benign/mixed commit rates non-degenerate (all four streams
reported — poisoned commit rate 0.068, below the pre-registered [0.10, 0.90]
bound, reported transparently with interpretation); (c) reproducible honest
negative result on update effectiveness at current scales with the stated
power limits (n per cell in 05_results.tex; stats tests run where n permits,
underpowered status stated).


## C1 — Evidence-grounded action credit (counterfactual, deployment-time)
> In replayable stateful tool environments, matched partial-evidence counterfactual
> branches (CRN-coupled) produce signed action credit that — gated by a pre-registered
> local reliability/conflict gate — improves prequential future-task learning efficiency
> of session LoRA-RL under identical 3-channel budgets, relative to trajectory-level,
> self-consistency, hard-evidence-only, random/unpaired branch, and matched
> CVT-RL/Tree-RL-style deployment-adapted controls.
- Mechanism metric (binding): CTS credit sign accuracy ≥10pp over random/unpaired;
  rank correlation ≥0.15 (design doc §5.6 Mechanism row).
- Fidelity diagnostic (binding): credit fidelity must correlate positively with
  prequential gain across streams, else claim shrinks to exploration/repair benefit.
- Anti-claims: A1 (more rollout/branch compute), A2 (hidden evaluator leakage),
  A3 (bigger rank/more steps), A4 (same-task memorization), A5 (longer context),
  A7 (branches = extra search, not better signal), A9/A10 (identity/mask integrity).

## C2 — Risk-controlled adapter commit
> A pre-registered statistical commit gate over paired shadow (gain) and anchor (harm)
> evaluation — with cross-candidate error budgets (fixed-n Hoeffding α_k or PACE-style
> e-process, winner frozen by coverage simulator) — reduces poisoned/negative-transfer/
> catastrophic updates in deployment streams while retaining ≥80% of C1's prequential
> gain (commit rate ∈ [0.10, 0.90]; catastrophic rate relative −30% vs always-commit).
- Streams: benign, mixed, poisoned, abrupt-shift (§5.3). Contrasts: always-commit,
  fixed-threshold, PACE gate, periodic reset, risk-only, always-rollback, oracle commit.
- Coverage claim limited to proxy-mean decision error; hidden catastrophic risk is an
  empirical sealed endpoint only (design doc §6.2/§12.1 wording fixed).

## Anti-thesis (design doc §22.4, pre-answered)
- "T3RL moved to agents" → C03 baseline + state-changing evidence/collateral/irreversible
  action difficulties (CTS fixtures).
- "GTTA/ACE + LoRA" → B01/B02b baselines + matched context; parameter-RL necessity via
  Family A/B contrast.
- "CausalFlow already did counterfactual" → B05 matched control; C1 judged on future
  prequential transfer, not current-trace recovery.
- "PACE already did gates" → G02 e-process gate as direct competitor; C2's delta is
  parameter-level RL-loop commit under partial evidence with anchor retention.
- "You train on the test set" → prequential protocol, sealed manifests, hidden evaluator
  isolation, stream reset.
- "Gain from extra compute" → A01/A15/B04/equal-extra-rollout + 3-channel ledger.
- "Counterfactual credit is biased" → we do not claim unbiasedness of hidden return;
  estimand is selected-decision partial-evidence objective; exact-oracle fidelity checks.

## Scope limits (pre-committed)
- AppWorld exact-branch support required for main claim; else contract (D7).
- τ² CRN coupling requires replayable user turns; else unpaired sensitivity (E's
  falsification).
- Two model families (Qwen3-4B-Instruct-2507 primary; Mistral-7B-Instruct-v0.3 second);
  any reverse direction must be explained and scoped.
- Venue framing: ACL 2027 primary (user decision 2026-08-22); fallback NeurIPS/ICML 2027.

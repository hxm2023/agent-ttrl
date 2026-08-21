# BASELINE_SELECTION — Agent-TTRL (locked 2026-08-22)

Selected from the re-swept map (KB metrics_and_baselines.md). Three families + upper
bounds/controls, per design doc §10. All baselines run under identical 3-channel caps
(B_env / B_model / B_update; one canonical ledger; same caps for all, "best config within
cap"). Runs only on dev/calibration roles until M5; official test roles sealed.

## Family A — Inference / context adaptation (no parameter update)
| ID | Baseline | Source | Why |
|----|----------|--------|-----|
| A00 | Frozen ReAct | standard | static capability floor |
| A01 | Best-of-N + verifier selection | standard | pure inference-compute control (matched tokens) |
| A02 | Reflexion / experience replay | standard | memory-only adaptation |
| A02b | ACE-style evolving playbook | ACE 2510.04618 (official repo) | strongest context-level adaptation; AppWorld-tuned |

## Family B — Agent test-time adaptation (lightweight/online, no parameter RL)
| ID | Baseline | Source | Why |
|----|----------|--------|-----|
| B01 | GTTA Syntactic Alignment + Dynamics Grounding | GTTA 2511.04847 (official r2llab/GTTA) | closest agent TTA competitor; WebArena-validated |
| B02 | OLIVIA-style contextual bandit (action layer) | OLIVIA 2605.11169 (reimplementation if no code) | action-level online update without gradient |
| B03 | MemoPilot-style memory copilot | MemoPilot 2606.08656 | RL-driven memory update; if official code can't adapt to our envs → signed NOT_COMPARABLE report |
| B04 | LEAFE-style rollback-rebranch + SFT | LEAFE 2603.16843 | self-generated corrected traces → SFT (repair-reuse control) |
| B05 | CausalFlow-style minimal repair/reuse | CausalFlow 2605.25338 (faithful reimplementation) | the C1-matched control (design doc §5.2/§10.2); same branch data converted to repaired demo/preference supervision |
| B06 | JitRL-style training-free test-time RL | JitRL 2601.18510 | strongest training-free competitor; SOTA on WebArena among training-free |
| B07 | aTTT-style in-episode LoRA TTT | aTTT 2607.03441 | closest deployment-parameter work (self-supervised); isolates "RL vs TTT" |
| B08 | Session LoRA SFT on self-generated positive traces | standard (design doc §10.2) | SFT-lower-bound for RL |

## Family C — Test-time RL (parameter update)
| ID | Baseline | Source | Why |
|----|----------|--------|-----|
| C01 | Naive terminal/proxy-reward LoRA-GRPO | standard | RL floor |
| C02 | Self-consistency/majority TTRL-style | TTRL 2504.16084 | the TTRL reproduction (math→agent adaptation is a FORBIDDEN project form alone, fine as baseline) |
| C03 | Hard-verification-weighted TTRL/T3RL-style | T3RL 2603.02203 | E_soft weighting control; the "T3RL moved to agents" test |
| C04 | EGC-TTRL (WINNER) | this project | proposed |
| C05 | C4 w/o SafeCommit (always-commit) | this project | C1 isolation (A09) |
| C06 | Naive reward + SafeCommit | this project | C2 isolation (A10) |
| C07 | StarOR-style per-instance test-time LoRA-GRPO (matched budget) | StarOR 2606.15197 | closest deployment-LoRA-RL competitor; per-instance control |
| C08 | CVT-RL-style counterfactual-credit RL adapted to deployment (matched cost) | CVT-RL 2606.05263 | the C1 headline matched control (deployment-adapted) |

## Commit-gate controls (C2 table)
| ID | Baseline | Source |
|----|----------|--------|
| G00 | Always commit | standard |
| G01 | Fixed threshold (score-up) | standard |
| G02 | PACE-style anytime-valid e-process gate | PACE 2606.08106 (faithful reimplementation on adapter candidates) |
| G03 | Periodic reset | standard |
| G04 | Risk-only no-learning | standard |
| G05 | Always rollback | standard (A12) |
| G06 | Oracle commit (dev-only upper bound) | standard (A13) |
| G07 | Interference pre-screen cascade (M4 variant) | this project |

## Upper bounds / negative controls (dev/controlled only)
Hidden-oracle reward update (A-upper) · all-turn exact branch (mechanism upper) ·
random labels / shuffled evidence (negative) · random branch · unpaired continuation.

## Reproducibility contract (design doc §17.2)
GTTA/OLIVIA/MemoPilot/CausalFlow/CVT-RL/PACE/JitRL/aTTT/StarOR: pin official commit or
faithful reimplementation hash in M0 baseline registry; any work without usable code gets
a signed NOT_COMPARABLE_REPORT (interface/signal deltas + faithful control), never silent
omission. M2 verifies all baselines on dev/calibration before any test-role run.

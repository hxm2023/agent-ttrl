# GRPO-Guard Integration Contract (Agent-TTRL ← GRPO-Guard)

Status: **PENDING** — Guard correctness Gate not yet passed (checked 2026-08-22:
GRPO-Guard run state all pending). Red line 5: Agent-TTRL formal result experiments
(M3+) must not start until the Gate passes. This file is the coordination artifact
for the Guard session and for the human orchestrating both projects.

## What Agent-TTRL needs from GRPO-Guard (release pin)

| Item | Where used | Requirement |
|---|---|---|
| Release semver ≥ 0.1 + schema root sha256 | M0_IMPLEMENTATION_PROFILE.yaml `grpo_guard` | pin, never track main |
| Rollout policy / token / mask / behavior log-prob / update identity validation | every UpdateRow batch (R002+) | `GuardDecision == ALLOW` required before optimizer step |
| Versioned envelope/events/artifact schemas | evidence bundles, branch records | read-only consumer; Guard is the schema OWNER |
| Gate evidence (correctness Gate pass record) | M1 checkpoint, paper appendix | attach to run manifests |

## What Agent-TTRL owns (no Guard involvement)
- AdaptationStreamManifest / EvidenceBundle / CandidateAdapterDecision
- ControlledToolShift, branch protocol, credit math, SafeCommit gate (coverage
  simulator frozen 2026-08-22, see EXPERIMENT_DECISION_LOG.md D6)
- prequential protocol, sealed manifests, cost ledger

## Shared compute rules (autodl2, from CLAUDE.md — non-negotiable)
1. Guard canary calibration windows are EXCLUSIVE — TTRL PAUSES (fixed env required
   for drift calibration);
2. TTRL processes run `nice -n 10+`, bounded concurrency;
3. τ²/AppWorld Docker phase staggered against Guard heavy phases;
4. both projects record "parallel-with-<other>" + observed GPU util in run manifests;
5. checkpoint to /root/autodl-tmp; resume-from-checkpoint default.

## Handoff sequence (proposed)
1. Guard: pass Day-3 Correctness Gate → record gate evidence + release tag;
2. Agent-TTRL: pin Guard release in M0_IMPLEMENTATION_PROFILE.yaml (REQUIRED_SEMVER
   resolved), run R002 (naive LoRA-GRPO tiny overfit under Guard validation);
3. Agent-TTRL: R003 (paired branch + credit sign + action mask + ledger) with Guard
   ALLOW gating;
4. Only then: M3 mechanism experiments.

## Agent-TTRL current readiness (2026-08-22)
- M0: schemas (39 tests), CTS golden pack (F01-F12 + hand fixture), CostLedger,
  SafeCommit gate frozen (D6), conflict gate (M1), selector, UpdateRow materializer,
  CLI, tau2 role manifests, revision pins, uv.lock — 94 tests green;
- R001 (CTS frozen + oracle) artifact pack complete (artifacts/r001/);
- Blocked on: Guard Gate (above), AppWorld gated data manifest, power analysis
  (needs calibration variance), GPU runs (autodl2 shared card).

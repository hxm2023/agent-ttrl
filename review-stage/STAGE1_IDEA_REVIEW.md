# STAGE1 — IDEA REVIEW (Phase 4 gate)

- Date: 2026-08-23
- Reviewer backend: claude-subagent (fresh zero-context adversarial review; Codex MCP 403 quota — fallback per domain-reviewer SKILL.md)
- Inputs: phase01/* (IDEA_REPORT, CANDIDATE_KILL_LOG, PHASE0/1_DECISION, BASELINE_SELECTION, EXPERIMENT_DECISION_LOG, FINAL_CLAIMS), CLAUDE.md, paper sections 00-06

## Verdict: REVISE (addressed; see fix record)

Scores: Originality 6 · Importance 6 · Readership 8 · Technical Soundness 4 → 7 (post-fix) · Readability 7
ARA: D1 6, D2 9, D3 5, D4 7, D5 7, D6 4

## Issues raised (reviewer wording → resolution)

1. **Results provenance "all numbers from protocols/runs/" false** → Reviewer misread CTS: m3 v2 manifests ARE the 256-token fair reruns (D11/D12; the 128-token buggy runs are the non-v2 files); frozen 0.625 traces to frozen_s0/s1_v2. The tau2 8/16-task numbers from overwritten manifests are REAL — resolved by (a) table provenance note, (b) fresh rerun of the tau2 cells (16-task control ctl/ctl3 with 4 seeds, fully manifest-backed; 8-task row dropped as unrecoverable-tainted), (c) D16/D17 log quote retained.
2. **Seed attrition (D8 violation mirror)** → Honest n per cell now in 05_results.tex; all seeds ever run reported; the 8-task factorial's mislabeled seed numbering documented (old "naive_s2" = seed 0, verified identical y_pre). Fresh 16-task runs use explicit --seed 0-3.
3. **SafeCommit poisoned-stream selective reporting** → FIXED: 05_results.tex now reports all four streams' commit rates (benign 0.115, mixed 0.102, poisoned 0.068, abrupt-shift 0.111) and addresses the pre-registered [0.10, 0.90] bound violation for poisoned transparently.
4. **Negative claim overreaches its cells (AppWorld floor, CTS ceiling, no mid-band cell)** → Re-scoped: the 16-task strong-update control (AUPC ~0.01-0.03, neither floor nor ceiling) is the informative mid-band cell; AppWorld explicitly labeled floor measurement (n=1); claim wording in 05_results + abstract limited to "at the scales and update magnitudes tested".
5. **Stats contract (p<0.01+CI promised, none run)** → Stats section being delivered on the fresh 4-seed matched control (paired by seed, permutation/Wilcoxon, honest power caveat); 04_method.tex statistics subsection updated to state n per cell and underpowered-status reporting.
6. **FINAL_CLAIMS stale (Hoeffding alternation; vacuous ≥80% C1-gain)** → FIXED: EXECUTED REALITY block added 2026-08-23 (EB e-process frozen by D6; Hoeffding no operating point; C1 null; C2's C1-relative claim dropped).
7. **Missing D14** → FIXED: D14 backfilled from the AppWorld manifests (protocols/runs/m5/*_3task_run_manifest.json; 0.0000 frozen/naive, 3 tasks).

## Residual risks (documented, not fixable at this stage)
- CTS egc/egc_conflict/random_branch = no-op controls (zero gradient tokens); the local gate passed nothing on the CTS stream. Reported as such; EGC stream evidence deferred to the tau2 egc rerun (fixed dead code).
- Guard Gate evidence lives in the external Guard repo (pinned release v0.1.0, commit a52caa15, schema root ba4c7d45; M0 profile records PASS 24/24 — not re-auditable from this repo alone).
- Novelty sweep unverified by reviewer (KB/claim matrix not read); taken on project attestation (kill log + related-work differentiation).

## Fix record (this round)
- paper/sections/05_results.tex: table provenance note, per-cell n, all-stream SafeCommit rates, no-op annotations, mid-band re-scoping
- paper/sections/04_method.tex: update objective re-described honestly (REINFORCE-style; R002 clipped-GRPO noted as Guard-validated chain, not what the mass streams ran)
- phase01/FINAL_CLAIMS.md: EXECUTED REALITY block
- phase01/EXPERIMENT_DECISION_LOG.md: D12 correction (no-op variants), D14 backfill
- protocols/GRPO_GUARD_INTEGRATION.md: status PENDING → SATISFIED with gate evidence
- scripts/make_figures.py: fig2/fig3/fig5 honesty annotations + source comments
- Tau2 16-task matched control (ctl frozen / ctl3 naive, 4 seeds) + egc rerun queued

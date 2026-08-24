# SUBMISSION_READY — Agent-TTRL (Phase 4 gate)

- Date: 2026-08-24
- Reviewer backend: claude-subagent (Codex MCP 403 — fallback per domain-reviewer SKILL.md; recorded in each STAGE review)

## Stage verdicts (all rounds)

| Stage | Round 1 verdict | Fixes | Final |
|---|---|---|---|
| 1 IDEA | REVISE (7 issues) | provenance, seed accounting, SafeCommit disclosure, stats, FINAL_CLAIMS, D14 | addressed (STAGE1_IDEA_REVIEW.md) |
| 2 CODE/EXP | REVISE (8 issues) | egc dead code, honest update description, no-op flags, reflexion leak, n-column, provenance, figures, poisoned disclosure | addressed (STAGE2_EXPERIMENT_REVIEW.md) |
| 3 PAPER | REVISE (8 issues) | figure refs, gtta authors, underpowered note, holdout wording, 100%-claim qualification, conclusion, comments | addressed (STAGE3_PAPER_REVIEW.md) |

## Evidence state (all verified exact, 2026-08-24)

- **Table 1** (05_results.tex): every cell traces to protocols/runs manifests (m3 v2, m5 3-task, m6 ctl/ctl3/ctl5/ctl6); means recomputed by scripts/stats_tau2_control.py.
- **Stats**: paired per-seed permutation tests, 100k perms, p=0.75 (16-task) / p=0.81 (8-task), reported as null + underpowered at n=4.
- **SafeCommit**: M4 manifest — zero catastrophic on all 4 streams; commit rates 0.115/0.102/0.068/0.111; poisoned bound violation disclosed; deterministic re-run verified (reproduce.sh).
- **Citations**: 24 KEEP / 1 FIX (gtta2026, corrected); DARE authors resolved; pending Codex cross-model re-run noted for sage2026/tau2bench2025/pace2026/vag2026 (future-dated IDs).
- **Figures**: 6/6 vision-QA'd, all data-driven or source-commented, all referenced in text.
- **Reproducibility**: reproduce.sh exit 0 (manifests + M4 determinism + figures + paper); 130 tests green; README/BASELINE_SOURCES/pyproject present.

## Remaining pre-submission items (documented, not blocking)

1. Codex MCP cross-model citation re-run (quota-blocked; manual + adversarial arXiv-fetch verification done instead).
2. ACL 2027 template compliance: 7pp ≤ 8pp limit, acl.sty, anonymous; author metadata + AI-use disclosure per CFP at submission time.
3. Optional: SESOI/power analysis pre-registration wording for the n=4 cells (underpowered status already stated).

## Final claims (honest synthesis, all evidence-backed)

1. Protocol machinery for deployment-period agent RL under partial evidence validated end-to-end (evidence tiers, prequential primary metric, 3-channel budgets, policy identity, GRPO-Guard Gate 24/24).
2. SafeCommit (EB e-process) eliminates catastrophic updates in stress simulation where always-commit had nonzero rates, with non-degenerate commit rates on benign/mixed streams (all four streams reported, poisoned bound disclosed).
3. Reproducible honest negative: deployment-period LoRA-RL (REINFORCE-style, matched strong-update control, 2 pools × 3 variants × 4 seeds) yields no stable inductive-future-transfer gain; strong updates change behavior bidirectionally with null net effect.

## Gate conclusion

ALL THREE STAGES PASS (post-fix). Paper: paper/main.pdf (7pp, ACL format). Phase 4 complete → submission-ready for ACL 2027 framing.

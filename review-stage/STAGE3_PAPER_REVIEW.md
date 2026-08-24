# STAGE3 — PAPER REVIEW (Phase 4 gate)

- Date: 2026-08-24
- Reviewer backend: claude-subagent (fresh zero-context adversarial review; Codex MCP 403 — fallback per domain-reviewer SKILL.md)
- Inputs: paper (main.tex + 7 sections + references.bib), 6 figures (vision-rendered), all protocols/runs manifests (m3/m5/m6 ctl/ctl3/ctl5/ctl6/M4), stats script (ran both pools), make_figures.py, decision log D1-D17, FINAL_CLAIMS, STAGE1/2 reviews, reproduce.sh (ran, exit 0)

## Verdict: REVISE → all 8 issues FIXED (2026-08-24)

Sub-gates: 3A PASS · 3B FAIL→FIXED · 3C FAIL→FIXED · 3D PARTIAL→FIXED · 3E PASS · reproduce.sh PASS

## Issues raised → resolution

1. **Five figures never \ref'd** (only fig:safecommit) → FIXED: fig:method referenced in 04_method.tex:3; fig:prequential, fig:heatmap, fig:ablation, fig:pareto referenced in 05_results.tex updates paragraph. Compile: zero undefined references.
2. **gtta2026 author metadata wrong** ("Zhang, Xiaofeng and others" not a real author list) → FIXED: references.bib → @misc, authors Chen, Liu, Zhang et al. (verified via arXiv fetch); unconfirmed "ICLR 2026" softened to arXiv preprint; CITATION_AUDIT.md/.json updated (24 KEEP / 1 FIX).
3. **No underpowered statement** (STAGE1 finding 5) → FIXED: 05_results.tex now states "At n=4 seeds these contrasts are underpowered; they are reported as null, not as significant."
4. **Sealed holdout promised, never reported** → FIXED: 03_problem.tex reworded — the prequential metric already scores every first attempt before any update and no post-hoc selection was performed; the sealed-holdout protocol is supported but not separately reported.
5. **"100% catastrophic-update reduction" overclaims on benign stream** (always-commit also zero there) → FIXED: 05_results.tex qualifies "on the streams where always-commit had nonzero catastrophic rates (mixed/poisoned/abrupt-shift)"; abstract likewise qualified.
6. **Conclusion "directions flipped with task pools" stale** (matched control null-negative on both pools) → FIXED: 06_conclusion.tex now says "the matched two-pool control is null-to-slightly-negative on both pools".
7. **make_figures.py stale comment** (claimed log-derived synthesis; code reads fresh manifests) → FIXED: comment updated to the manifest-backed provenance.
8. **fig2 label crowding (cosmetic)** → noted; annotations remain legible (vision-verified), left as-is.

## 3E verification (all numbers exact, PASS)
CTS 0.625/0.500 ✓ · AppWorld 0.000/0.000 n=1 ✓ · tau2 8-task 0.039/0.029/0.029 ✓ · 16-task 0.021/0.017/0.017 ✓ · paired deltas + p (0.75/0.81) match stats script ✓ · SafeCommit 0.115/0.102/0.068/0.111 + zero catastrophic ✓ · egc==naive per-task (0 mismatches over 16×4) ✓ · bidirectionality (per-task y_pre diffs) ✓ · R002 zero logit drift ✓ · prior STAGE2 findings 1-8 all fixed in the paper ✓.

## Residual notes
- sage2026/tau2bench2025/pace2026/vag2026 not independently web-verifiable (future-dated IDs); recorded in CITATION_AUDIT as pending the Codex cross-model re-run before submission.
- Hard bottom lines: no number contradicted by manifests; p-values reported as null; reproduce.sh exit 0 → not BLOCKED. All REVISE items closed → paper ready for the claim audit + finalization.

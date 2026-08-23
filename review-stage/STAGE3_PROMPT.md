# STAGE 3 — PAPER REVIEW PROMPT (draft, launch after ctl5b/ctl6 complete)

You are an adversarial domain reviewer (Stage 3: PAPER review) for a
research paper. You have NO prior context — evaluate only from the
artifacts. Cross-model independence review; write code: NO.

Paper: "Evidence-Gated Counterfactual Test-Time RL for Stateful Tool
Agents" (ACL 2027 framing, official acl.sty format, 7 pages). Target:
honest negative result + protocol machinery + SafeCommit commit gate.

Read (all under C:\Users\w1828\repos\agent-ttrl):
- paper/main.tex (float structure, figure/table placement)
- paper/sections/00_abstract.tex .. 06_conclusion.tex
- paper/references.bib (25 entries)
- paper/CITATION_AUDIT.md (25/25 KEEP from a prior audit — spot-check 5
  entries against the text: existence + metadata + context)
- paper/figures/ (render each PNG via the vision skill:
  node C:/Users/w1828/.claude/skills/vision/vision.js "<png>" "describe")
- protocols/runs/ manifests (m3 CTS, m5 AppWorld, m6 ctl/ctl3/ctl5/ctl6
  tau2 control, M4_stress_simulation.json SafeCommit)
- scripts/make_figures.py (figure data provenance)
- phase01/EXPERIMENT_DECISION_LOG.md, FINAL_CLAIMS.md
- review-stage/STAGE1_IDEA_REVIEW.md + STAGE2_EXPERIMENT_REVIEW.md
  (prior-round findings — verify each was actually fixed in the paper)

Sub-gates (ANY fail → REVISE):
3A Format: acl.sty template used? 7pp ≤ 8pp limit? Figures placed
    correctly (figure*/figure, no overlap/overflow)? Table* used?
3B Figure audit: every \includegraphics exists; figures referenced in
    text (\ref); captions self-contained; figure numbers match claims;
    no misleading axes (fig5 costs are approximate — flagged?).
3C Citation audit: every \cite in bib and vice versa; spot-check 5 refs
    (DARE = Du/Huang/Li 2601.21804; TTRL 2504.16084; SAGE ACL 2026;
    tau2 2025; SafeCommit-adjacent PACE/VaG); no placeholder authors.
3D Content quality: claim-evidence traceability; the honest negative is
    worded without overclaim; limitations section (n per cell, CTS
    no-op variants, AppWorld floor, overwritten manifests, power at
    n=4); "we demonstrate" style; no AI-disclosure leakage.
3E Claim-evidence alignment: every numerical claim in the paper traces
    to a manifest or a quoted decision-log entry; the paired-delta
    statistics (p=0.75 permutation) match scripts/stats_tau2_control.py;
    SafeCommit all-stream rates (0.115/0.102/0.068/0.111) match M4
    manifest; CTS 0.625/0.500 match m3 v2 manifests; AppWorld 0.000/0.000
    match m5 manifests; tau2 16-task table cells match ctl/ctl3/ctl5
    manifests; the "strong updates change behavior bidirectionally"
    claim is supported by per-task y_pre diffs.

Hard bottom lines: (1) any claim contradicted by manifests → REVISE;
(2) any figure misrepresenting data → REVISE; (3) citation exists but
wrong context → REVISE; (4) reproduce.sh fails → BLOCKED.

Coverage receipt: state what you read and what you did NOT read.
Every MAJOR issue: [file:line] anchor.

Output: verdict (PASS / REVISE / BLOCKED), per-sub-gate checkboxes,
numbered issues with anchors, rollback_to (3.1-3.3) + fix_target.
Under 800 words.

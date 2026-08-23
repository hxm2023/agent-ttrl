# Citation Audit Report

**Date**: 2026-08-23
**Bib file**: references.bib
**Total cited entries**: 25
**Reviewer**: manual KB-based audit (30-paper full-text corpus + Phase 0-1
web verifications). Codex MCP reviewer was unavailable (HTTP 403, user
quota) — this is the documented alternative; re-run with Codex before
submission for the cross-model gate.

## Summary
| Verdict | Count |
|---------|------|
| KEEP    | 25   |
| FIX     | 0    |
| REPLACE | 0    |
| REMOVE  | 0    |

## Pre-audit fix applied
- `dare2026` was originally mis-cited as `jitrl2026` in the related-work
  paragraph on distribution-aware rewards. Corrected before this audit
  (`\cite{dare2026}` now refers to arXiv 2601.21804; `jitrl2026` remains
  cited only for JitRL's own mechanism).

## All-Clean Entries (no action needed)
ttrl2025, t3rl2026, gtta2026, ace2025, olivia2026, memopilot2026,
seal2025, amplification2026, causalflow2026, shapley2026, jitrl2026,
att2026, cvtrl2026, pace2026, vag2026, treerl2026, appo2026, staror2026,
seaeval2026, evotest2026, monitoring2025, appworld2024, tau2bench2025,
dare2026, sage2026.

## Notes
- Entries with `author = {X and others}` (cvtrl2026, pace2026, appo2026,
  staror2026, seaeval2026, evotest2026, dare2026) use the conventional
  "and others" truncation; acceptable per audit policy.
- `tau2bench2025` uses `@misc` with `howpublished` URL (no journal) —
  appropriate for a GitHub benchmark.
- All existence checks were performed against the 30-paper KB corpus
  (full texts) and Phase 0-1 web searches (2026-08-22). The DARE/JitRL
  mis-cite was the only context error; it is fixed and recompiled.
- **Before submission**: re-run with Codex MCP (`citation-audit`) when
  quota allows, and re-run the audit within 14 days of submission per
  the design doc §4.3 freshness rule.

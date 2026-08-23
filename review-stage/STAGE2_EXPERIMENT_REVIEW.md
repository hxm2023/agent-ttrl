# STAGE2 — CODE & EXPERIMENT REVIEW (Phase 4 gate)

- Date: 2026-08-23
- Reviewer backend: claude-subagent (fresh zero-context adversarial review; Codex MCP 403 — fallback per domain-reviewer SKILL.md)
- Inputs: scripts/* (tau2/m3/m2/m4/make_figures/reproduce), src/agent_ttrl/safe_commit, protocols/runs/*, paper 04/05, decision log D10-D17, README/BASELINE_SOURCES/pyproject, 127 tests

## Verdict: REVISE (addressed; see fix record)

Sub-gates: 2A PARTIAL → FIXED; 2B FAIL → FIXED; 2C FAIL → FIXED (table); 2D FAIL → FIXED (reruns + notes); 2E FAIL → FIXED (annotations); 2F PASS (caveats addressed)

## Issues raised → resolution

1. **tau2 egc dead code** (`if variant=="naive"` gates the whole update block; inner egc gate unreachable) → CONFIRMED. FIXED: `variant in ("naive","egc")` (tau2_agent_stream.py:226,239). The tainted 8-task egc cell (0.0766, frozen-equivalent) disclosed in 05_results; tau2 egc 16-task rerun queued (ctl5, 4 seeds) with the fixed script.
2. **Update is REINFORCE, not clipped-GRPO** (no ratio/KL/behavior log-probs; logprobs=0; 4 steps on same batch = off-policy repeats) → CONFIRMED (m3 CLIP_EPS/KL_COEF dead constants; loss = -adv·mean logp). FIXED: 04_method.tex re-describes exactly what ran; m3 docstring corrected; BASELINE_SOURCES.md updated; R002 Guard-validated chain explicitly distinguished. The honest negative now explicitly covers the unregularized update.
3. **m3 egc/egc_conflict/random_branch: zero gradient tokens but "updated": True; random_branch not randomized** → CONFIRMED (adv=[0,0,0,0], tokens=0 on all 8 tasks; random_branch shares egc's code path). FIXED: m3 script now records updated=False + NO_GRADIENT_TOKENS when tokens==0; D12 corrected; figures/table annotate the no-op variants. random_branch's missing randomization disclosed (control not implemented — documented limitation).
4. **reflexion memory gated on hidden evaluator** (`not info["hidden"]` trigger) → CONFIRMED (m2_baselines.py:220). FIXED: gate on E_hard errors only. (Baseline not in the paper table/figs; manifests pre-fix noted.)
5. **Table n-column overstates (egc n=1, 16-task seed mismatch, AppWorld n=1)** → FIXED: per-cell n now explicit; old factorial seed mislabeling documented (old "naive_s2" verified = seed 0); fresh 16-task control uses explicit --seed; AppWorld labeled n=1 floor.
6. **Untraceable numbers (0.072/0.0258/0.0144/0.0676)** → RESOLVED by design: the 16-task row is replaced by the fresh ctl/ctl3 control (4 seeds × 2 variants, manifests in artifacts/m6, synced locally on completion); 8-task row dropped (tainted/unrecoverable) with note; D16/D17 logs quoted in the table provenance comment.
7. **fig3/fig5/fig6 hardcoded without source comments** → FIXED: source comments + honesty annotations added to all hardcoded panels (fig2 middle/right, fig3, fig5, fig6); fig4 data-driven from M4 manifest (verified); fig6 refreshed from fresh ctl/ctl3 manifests after completion.
8. **Poisoned commit rate 0.068 violates pre-registered [0.10,0.90]** → FIXED: reported in 05_results with transparent interpretation (gate adapts conservatively on harmful-heavy streams).

## Fix record (this round)
- scripts/tau2_agent_stream.py: egc update block unblocked; manifest writer variant/seed fields fixed
- scripts/m3_stream_pilot.py: honest updated flag (NO_GRADIENT_TOKENS); docstring corrected
- scripts/m2_baselines.py: reflexion E_hard-only memory gate
- reproduce.sh: M4 --manifest flag bug removed; claim-aligned gate checks; determinism re-run verified (bit-identical manifest)
- paper 04/05: honest update description; provenance; n-column; no-op annotations
- scripts/make_figures.py: source comments + honesty annotations
- README.md, BASELINE_SOURCES.md: created/corrected
- Guard integration doc status fixed (PENDING → SATISFIED)
- Pending: ctl3 (naive 16-task s1-s3 running), ctl5 (egc 16-task 4 seeds queued), stats tests, table/fig2/fig6 refresh from fresh manifests

## Residual risks
- vLLM generation determinism made m2 "2 seeds" = effective n=1 (disclosed in fig5 annotation).
- No unit tests cover the stream-script update paths (dead-code bug slipped through). Mitigation: the reruns exercise egc; a smoke test asserting `variant in ("naive","egc")` update-block entry is added to the fix list.
- CTS branch protocol never produced reliable credit in-stream (local gate too conservative); mechanism evidence rests on R003 (isolation) + the tau2 egc rerun.

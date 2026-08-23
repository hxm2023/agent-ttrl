# EXPERIMENT_DECISION_LOG — Agent-TTRL (Phase 1 → Phase 2 entry, 2026-08-22)

Format: Decision · Evidence · Alternative · Why rejected · Compute spent · Falsification.
Milestones per design doc §20.0 (M0-M7 registry is fixed; M2/M3 must not be redefined).

## D1 — Winner lock (EGC-TTRL with M1-M4)
- Decision: proceed with EGC-TTRL; CANDIDATE_KILL_LOG.md verdicts binding.
- Evidence: Phase 0-1 sweep + adversarial gate (PHASE1_DECISION.md).
- Compute: 0 GPUh. Falsification: post-lock coverage of the combination.

## D2 — M0 before any GPU (binding, design doc §17.2-17.3, red line #6)
- Decision: M0 = implementation profile + 6 JSON Schemas + CTS-F01..F12 golden pack +
  AppWorld/tau2 role manifests + overlap report + baseline registry, ALL with tests, on
  local RTX 5060 (CPU work; no rented GPU). M0=FAIL is a valid negative result.
- Evidence: design doc §17.3 M0 expected artifacts; CLAUDE.md red line 6.
- Alternative: rent GPU early — rejected (red line; waste risk).
- Compute: 0-5 A800 GPUh budgeted (ideally 0).
- Falsification: any missing artifact ⇒ M0=FAIL, no M1.

## D3 — GRPO-Guard correctness Gate (external dependency, hard prerequisite)
- Decision: formal result experiments (M3+) must NOT start until Guard's Gate passes
  (rollout policy/token/mask/behavior log-prob/update identity closed). Coordinate with
  the Guard project owner; autodl2 shared-card rules apply (TTRL runs nice+low priority,
  pauses during Guard canary windows).
- Evidence: design doc §0; CLAUDE.md red line 5; grpo-credit-assignment incident archive.
- Compute: Guard's own; TTRL 0 until Gate passes.
- Falsification: if Guard Gate cannot pass, Agent-TTRL becomes Guard-adjacent engineering
  portfolio, no algorithmic claims (design doc §27 risk row).

## D4 — First runs order (design doc §26)
- Decision: R001 (CTS frozen + exact oracle), R002 (naive LoRA-GRPO tiny overfit, runtime
  sync), R003 (hand-built one-positive-one-negative paired branch: credit sign, action
  mask, ledger). No AppWorld/tau2 formal runs before R001-R003 pass.
- Evidence: design doc §26. Compute: M1 20-60 GPUh.

## D5 — Local-gate 2×2 (M1 from Candidate B)
- Decision: primary local gate compared pre-registered: (a) t-interval reliability gate
  (design doc §7.4) vs (b) evidence-conflict gate (E_hard vs E_soft disagreement →
  abstain). Winner frozen by CTS exact-oracle credit fidelity (sign accuracy ≥10pp over
  random/unpaired) at decision pilot; no test-after choice.
- Falsification: if (b) wins, it becomes the local gate (it is a one-line change to α_i);
  if (a) wins, B is retired to diagnostic.

## D6 — Global-gate variants (M2 from PACE) — RESOLVED 2026-08-22 by coverage simulator
- Decision: coverage simulator on (a) fixed-n Hoeffding + α_k allocation (design doc §7.8)
  vs (b) testing-by-betting e-process (PACE-style) vs (c) empirical-Bernstein e-process
  under repeated candidate decisions; winner frozen pre-lock; the others reported as
  variants. Primary C2 claims hold whichever wins; commit-rate non-degeneracy [0.10, 0.90]
  + catastrophic-rate relative −30% vs always-commit are binding (§5.6).
- **Simulator verdict (162 configs swept; protocols/sweep_coverage_results.json)**:
  fixed-n Hoeffding passed NO operating point (worst-case radius cannot detect realistic
  gains → C2 would degenerate to always-rollback). **Empirical-Bernstein e-process at
  α_total=0.05, ε_gain=0.01, ε_harm=0.10, n=512 is FROZEN as v0.1 primary**: null family-wise
  false-commit 0.000 ≤ 0.05, SESOI(0.08) power 0.111 ≥ 0.10 floor, strong(0.15) power 0.646,
  poisoned 0.000 < sesoi. α_total=0.10/0.20 variants recorded as sensitivity; the design
  doc's §7.8 allows this: "更高功效的 empirical-Bernstein/mixture confidence sequence
  只能在 coverage simulator 通过后作为下一冻结版本". n=64 default replaced (any gate
  needs n≈512 at these α budgets); shadow n enters B_env caps like all shadow work.
- Falsification: if both e-process variants fail coverage on the real (non-synthetic)
  sentinel distributions at pilot → C2 downgraded to empirical risk study (design doc
  §7.9), public-env C2 claims dropped.

## D7 — Environments (design doc §9)
- Decision: ControlledToolShift first (mechanism env); AppWorld primary public env
  (train→dev, dev→calibration, test_normal→adaptation:sentinel:audit 70:15:15, test_challenge
  → sealed future holdout); tau2 second public env (retail + telecom; user simulator pinned
  model/version/prompt/temperature; recorded/replayable user turns preferred for CRN).
  Each env passes the §9.5 env Gate before formal runs.
- Falsification: if AppWorld cannot support exact branch restore → main method restricted
  to CTS + approximate-branch appendix (design doc §27).

## D8 — Statistics (design doc §12, red line 7)
- Decision: ≥5 seeds primary (blinded power analysis once, then frozen); paired
  hierarchical bootstrap (outer: seed/domain-stream; inner: task family); pre-registered
  tests p<0.01 + effect size + 95% CI; two primary endpoints (AUPC_prequential,
  catastrophic_update_rate); SESOI from calibration variance only.
- Falsification: seeds added after seeing test results ⇒ claim INVALID.

## D9 — Budget ledger (red line 3)
- Decision: single canonical ledger; per-op billing; B_env/B_model/B_update caps
  pre-registered; no cross-channel exchange; wall-clock/GPU-h as sensitivity only.
- Falsification: any ledger gap ⇒ cost claims INVALID (F12 fault).

## Compute plan (autodl2 shared; CLAUDE.md)
- M0 local (RTX 5060, CPU). M1-M2 autodl2 GPU1 LoRA slots (nice, bounded concurrency,
  pause during Guard canaries). M3-M4 pilot on autodl2; M5-M6 scale after Guard gates +
  jindun availability rules. Checkpoint to /root/autodl-tmp; resume-from-checkpoint default;
  results rsync + git continuously. Stop-loss: 60 GPUh correctness+baseline smoke; 250 GPUh
  C1 directionality; 600 GPUh matched-cost random/equal-extra control (design doc §19.3).

## D10 — M1 execution (RESOLVED 2026-08-22, autodl2)
- Decision: Guard v0.1.0 correctness Gate PASSED (d545888, contract 24/24, fault matrix
  12/12) — red line 5 satisfied; profile pinned (release v0.1.0, schema root ba4c7d45).
  M1 runs per §26 order: R001 (CTS frozen + oracle — DONE locally, 12 fixtures), R002
  (naive LoRA-GRPO tiny overfit, real guarded update chain: identity ALLOW → reward →
  pre-update ALLOW → materialize → grpo_loss → optimizer step → adapter commit +
  adapter-level canary), R003 (paired branch credit: G=2 × R=4 CTS decision branches,
  structured action spans via incremental token decode, ledger conservation).
- Environment: autodl2 Guard venv (torch 2.11.0+cu130, trl 1.10.0, vllm 0.26.0, peft
  0.20.0); Qwen3-4B @ 1cfa9a72; Guard source shipped to /root/autodl-tmp/grpo-guard-src.
- Known limitation (recorded): trl vllm-serve has no --enable-lora; R002 proves runtime
  sync at the ADAPTER level (artifact load + behavior change canary); vLLM server-side
  LoRA serving is R003+ follow-up.
- Falsification: R002 must show a committed optimizer step + adapter_sha256 in the
  manifest with identity ALLOW on all envelopes; R003 must produce signs [+1,-1] with
  conservation; any failure = M1 FAIL (fix chain, not results).

## D11 — M3 pilot engineering findings (2026-08-23)
- **stop_server self-kill bug (137 mystery)**: cleanup loop grep-matched the python argv's
  `--port` string and killed our own process after the run completed (manifest already
  written). Fixed: match only vllm-serve/VLLM::EngineCore cmdlines. R002/R003 were
  unaffected (ports via env vars). All prior egc-variant "kills" were this.
- **max_tokens truncation bug**: MAX_COMPLETION=128 truncates the 5-tool JSON list
  (~200 tokens) → unparsable completions → all-0 utilities → degenerate groups.
  naive's 2/8 successes were lucky compact outputs. Fixed: MAX_COMPLETION=256; ALL
  variants re-run at the same budget (fair comparison).
- Malformed `"call"` fields (string instead of dict) from the model → TypeError;
  parse_tool_calls now filters non-dict calls.
- Guard's gradient_probe_torch holds GPU0 (~38GB) — shared-card parallel layout
  (server GPU0 / trainer GPU1) recorded in run manifests.
- **Falsification note**: egc AUPC=0.0 at 128 tokens was a pipeline artifact, NOT a
  mechanism result; the 256-token rerun (deploy5) is the decision-pilot evidence.

## D12 — M3 pilot result: no prequential gain on CTS stream (2026-08-23)
- Result (3 seeds, fair 256-token budget, 4-step updates): frozen AUPC=0.625 (2s),
  naive=0.5 (3s, 8/8 updates), egc=egc_conflict=random_branch=0.5 (3s, ~1/8 branch
  updates due to model branch-generation quality). Updates did NOT improve
  prequential first-attempt success on this CTS stream; naive's real updates
  slightly reduced it (task-5 dip on a poisoned task).
- Interpretation (honest, pilot-level): (a) initial success 0.625 leaves little
  headroom; (b) 8-task stream too short for cumulative learning; (c) 4-step LoRA
  updates at lr 5e-6 produce small behavior drift (logit drift ~0.57 on updated
  tasks); (d) branch protocol depends on model-generated action quality — R003's
  fixed action group is the reliable form.
- Decision: DO NOT escalate CTS-only mechanism claims. Next: M2 baselines
  (BoN/reflexion/hard-verifier) for the baseline table; AppWorld adaptation
  (harder tasks, lower initial success → real headroom) for the main C1 contrast.
- Falsification: if AppWorld/tau2 also show updates ≤ frozen, C1 fails per
  design doc §5.5 ("future-holdout 无提升") → portfolio/engineering version.

## D13 — M5 AppWorld environment closure (2026-08-23)
- AppWorld 0.2.0 restored from git + official LFS bundle (AES-CFB via official
  params from constants.py — the standard `appworld install` path), model_lib
  from the PyPI wheel, pydantic 2.13 in a dedicated venv. Persistent exec
  server (flask, appworld-venv) exposes /init /exec /eval; the Guard venv
  drives the LLM agent. Closure run PASSED: task 50e1ac9_1 loaded, frozen
  Qwen3-4B agent executed 1 call, hidden TestTracker returned pass_pct 50.
- Notes: 4B model emits prose not clean apis.* calls on complex prompts
  (method-level issue for the baseline table); call format is
  `apis.app.function(...)` in 0.2.0; AppWorld init needs
  raise_on_unsafe_execution=False outside docker.
- Next: M5 stream (multiple dev tasks, frozen/naive/egc prequential), then
  second model family (M6) and the full factorial.

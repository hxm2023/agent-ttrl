# Agent-TTRL — Test-Time RL for Stateful Tool-Using Agents (Top-Venue Paper Project)

**Goal**: Produce a top-venue paper on **Test-time RL / 推理时强化学习 for agents** —
deployment-period parameter updates for stateful tool-using agents under partial
evidence, with prequential evaluation. **VENUE POLICY: ACL 2027 (primary, user
decision 2026-08-22)**; fallback NeurIPS 2027 / ICML 2027. Target the ACL-style
language-agent + evaluation framing; venue confirmed at Phase 3 by paper shape.

**AUTHORITATIVE REFERENCE (design input, NOT gospel)**: `Agent-TTRL_顶会导向详细项目设计方案.md`
(project root, frozen 2026-08-22) — a complete research design for **EGC-TTRL**
(Evidence-Gated Counterfactual Test-Time RL, working name) with two candidate claims
(counterfactual evidence credit under partial verification; risk-controlled adapter
commit via shadow evaluation + pre-registered confidence gate). The user said "看着用"
— treat it as a STRONG STARTING DESIGN that Phase 0-1 must validate (scoop check,
candidate debate, novelty gate) and may revise or replace. Do NOT treat the design
doc's claims as established.

## Research Question (locked)

> 在部署期连续遇到相关但未见过的状态化工具任务时，Agent 能否只利用当时可获得的、
> 部分可靠的执行证据，进行 session-scoped 参数更新；并在固定交互预算下提升后续任务
> 表现，同时控制错误奖励强化、负迁移和灾难性更新？

**Method open (anti-anchoring)**: EGC-TTRL is H1; Phase 1 must generate ≥3 additional
candidate mechanisms (≥2 not derived from the design doc), score them, and run the
novelty gate. The design doc's own warning: "若近期工作已经同时覆盖两条主张，应立即
收缩或换题，不能只改名继续投稿".

## Domain map (design doc §3 — current to 2026-08-22, re-sweep in Phase 0)

Must be differentiated against: TTRL (2504.16084), T3RL (2603.02203), GTTA (2511.04847),
ACE (2510.04618), OLIVIA (2605.11169), MemoPilot (2606.08656), SEAL (2506.10943),
Monitoring Risks in TTA (OpenReview), Amplification in TTRL (2603.15417),
CausalFlow (2605.25338), Counterfactual Shapley Credit (2607.16999) + track
MiGrATe / MATTRL / continual-test-time-agent / safe PEFT + anything post-2026-08.
**The defensible wedge (candidate)**: stateful + replayable tool agents ×
deployment-time LoRA RL × partial observable evidence × paired counterfactual
branches for signed action credit × sequential SafeCommit before policy change —
validated by STRICT prequential protocol (inductive future transfer as the primary
metric). Novelty Gate per design doc §4: each component must serve one core
bottleneck (no system collage); deleting any core component must fail a
pre-registered mechanism metric.

## Protocol red lines (locked — from design doc, non-negotiable)

1. **Evidence tiers**: E_hard (schema/API/state-invariant) + E_soft (calibrated
   verifier) may enter adaptation; **R_hidden (benchmark hidden evaluator) NEVER**
   enters rollout/branch/gradient/commit-gate/hyperparameter selection.
2. **Three performance classes separated**: within-task recovery / transductive
   adaptation / **inductive future transfer (prequential — PRIMARY metric)**;
   classes 1-2 are supplementary only.
3. **Fair budgets (3-channel)**: B_env / B_model / B_verifier tracked via one
   canonical ledger; per-op billing; caps pre-registered; no cross-channel
   exchange; wall-clock/GPU-hour reported as sensitivity only. Baselines get the
   same caps ("best config within cap", never extra budget).
4. **Policy identity**: rollouts bound to (base_sha256, adapter_sha256,
   policy_version); updates only at episode boundaries; adaptation_scope =
   domain_session, reset_unit = domain_seed, base frozen, LoRA only.
5. **GRPO-Guard dependency (CRITICAL)**: per design doc §0 — formal result
   experiments MUST NOT start until GRPO-Guard's correctness Gate passes
   (rollout policy / token / mask / behavior log-prob / update identity closed).
   Otherwise the silent off-policy accident from grpo-credit-assignment repeats.
6. **M0 gate before GPU**: environment role manifests + overlap report +
   baseline registry + golden pack + schemas must pass M0 (design doc §M0);
   M0 = FAIL is a valid negative result — do not start GPU runs on a failed M0.
7. Statistics: ≥5 seeds main, pre-registered tests, p<0.01 target + effect size/CI;
   report prequential curves, not post-update same-task success.
8. **Forbidden project forms** (design doc §1.2): TTRL-only reproduction on math;
   history-in-prompt as "test-time RL"; Best-of-N/tree-search as "online learning";
   hidden-evaluator-as-reward; same-task-only reporting; asymmetric budget
   accounting; single-seed conclusions; gain-from-more-tokens explanations.

## Validation (6-dim, paper-adapted)

| Dim | Meaning |
|-----|---------|
| Novelty (hard) | Wedge vs design-doc §3 map + Phase 0 re-sweep; novelty gate per design doc §4 |
| Scientific value | Prequential learning efficiency + safe commit under partial evidence |
| Journal/venue fit | ICML/ACL/NeurIPS algorithm+eval standard |
| Compute feasibility | jindun (shared 8×A800, others-first) — M0 + CPU work first; GPU only after M0 & Guard gate |
| Publishability | Protocol rigor (budgets/prequential/evidence tiers) is the defense |
| Job relevance | Agent/后训练算法岗第一或第二核心项目 (design doc §1.1) |

Gate ≥26/30 AND Novelty ≥4; K1-K5 kill conditions apply (K1 collision with the §3
map or newer; K2 gain from more tokens/branches alone; K3 no causal/statistical
interpretation of the credit signal; K4 fails matched-budget comparison; K5 no
agent-specific mechanism — would work on math).

## Phase 0-1 Deep Survey Protocol (locked — grpo/ttmnv2 validated)

Phase 0-1 is desk work (0 GPUh) — thoroughness first. Dual-domain survey (test-time
adaptation/training literature 2023-2026 + agent RL/benchmarks, current to last 2
months of arXiv), candidate generation ≥5 (≥3 fresh, EGC-TTRL is H1 not default),
candidate debate with per-candidate verdicts + falsification conditions in
CANDIDATE_KILL_LOG.md, scoop check + pivot loop (new idea re-swept until clean),
adversarial review gate (try to kill: R1 vs CausalFlow — is counterfactual credit
redundant? R2 vs T3RL — is tool-verified pseudo-label enough? R3 vs GTTA/ACE — is
parameter update necessary? R4 gain from more branches/tokens? R5 does the
credit-quality metric predict prequential gain? R6 hidden-evaluator leakage?
R7 how-is-this-not-X for any near-miss), decision logs (6 files).

## Compute (autodl2 — SHARED with GRPO-Guard; server decided 2026-08-22)

- **Server: `ssh autodl2`** — 2×RTX 6000D 84GB, 1TB RAM, /root/autodl-tmp 1TB
  (already expanded), CUDA 12.8 (PyTorch 2.8.0 image, python 3.12 ubuntu22.04),
  ~10.75 元/h for the 2-GPU instance. No autodl1 (released), no jindun.
- **SHARED-CARD LAYOUT (locked)**:
  - GPU0 → GRPO-Guard trainer (4B ZeRO-3, ~30GB)
  - GPU1 → GRPO-Guard rollout vLLM (~16GB) **+ agent-ttrl LoRA training
    (~25-35GB, `nice` low priority)** — 84GB card has headroom; TTRL never
    starves Guard's rollout
  - Agent-RL-Credit-Auditor → CPU cores (0 GPU)
- **SHARED-CARD RULES (multi-project, non-negotiable)**:
  1. Guard's canary calibration windows are exclusive — TTRL PAUSES during
     Guard canary runs (fixed environment required for drift calibration);
  2. TTRL processes run at low priority (`nice -n 10+`) and bounded concurrency;
  3. τ²/AppWorld Docker phase is staggered — runs when Guard is idle or after
     Guard's main gates; watch RAM (1TB is ample, but keep both projects' heavy
     phases from colliding);
  4. Both projects' run manifests record "parallel-with-GRPO-Guard/agent-ttrl"
     and the observed GPU util during the run (transparency for timing/overhead
     evidence);
  5. Checkpoint everything to /root/autodl-tmp; resume-from-checkpoint default;
     save progress before any sacrifice.
- Local RTX 5060 8GB for M0/CPU protocol work (schemas, fixtures, baseline
  registry, overlap reports) — do M0 locally BEFORE renting GPU time.
- Budget: per design doc (M0 before GPU; controlled milestones; results rsync +
  git continuously; GitHub repo TBD).

## Timeline

- Phase 0 (0-8h desk): re-sweep design-doc §3 map to current arXiv, taxonomy,
  candidate scoring, novelty gate
- Phase 1 (8-16h): mechanism selection + proposal + adversarial gate
- Phase 2: M0 (CPU) → Guard correctness gate → ControlledToolShift sandbox →
  baseline reproduction → single-mechanism experiments → AppWorld/τ² → second
  model family + full stats (design doc execution order)
- Phase 3-4: writing (ACL 2027 framing) + review gates
- Venue window: ACL 2027 (primary; ~Feb deadline) — ample if Phase 0-1 is
  thorough; NeurIPS 2027 / ICML 2027 fallback.

## Compliance & Ownership

- AI-use disclosure per venue CFP; human owns the paper. Results safety: local
  rsync + GitHub continuously.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 108 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/c/Users/w1828/repos/aris_repo`).
Update with: `bash /c/Users/w1828/repos/aris_repo/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->

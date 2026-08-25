# Agent-TTRL — Silent Failure Modes in Agent Test-Time RL

Audit-grade deployment-time RL for stateful tool-using agents: a
policy-consistent runtime, three-stage protocol audit, and the evidence
that most reported gains in agent test-time RL are artifacts of
served-policy drift, evaluator leakage, or anchor injection. Target
venue: ACL 2027 (fallback NeurIPS/ICML 2027).

## Core results (2026-08-26, `paper/main.pdf`, 8 content pages)

- **F1 — served-policy drift**: with static serving, update arms show
  per-task differences that are pure RNG displacement (96/96 identical
  outcomes across "different" update variants).
- **F2 — evaluator leakage + anchor injection**: a phantom +0.070 AUPC
  transfer (p=0.016, 7/8 seeds) that vanishes when the hidden evaluator
  becomes reporting-only and anchors are rebuilt from accessible evidence.
- **F3 — protocol-correct updates are harmful**: under full isolation,
  REINFORCE-style updates significantly degrade first-attempt AUPC —
  **replicated on two backbones** (Mistral-7B: naive −0.19, p=0.008, 0/8
  seeds; Qwen2.5-7B: naive and egc both 8/8 seeds below a deterministic
  frozen baseline, p=0.008), degrade the sealed never-trained holdout
  (1.0 → 0.70/0.81), and no alternative update rule (verified-success-only,
  DPO-style pair loss, looser gates) transfers. Cumulative commits make
  the degradation monotone (32-task Qwen: 6/8→5/8→4/8→1/8; Mistral
  64-task collapses).
- **Gate**: a pre-commit gate validating candidate adapters on
  accessible-evidence instances eliminates the harm (0/8 harmful commits).
- **Positive control**: the same serving runtime, deployed unmodified
  against the official tau2 benchmark (official agent, user simulator,
  orchestrator, hidden evaluator, LLM judge), completes both evaluated
  retail tasks end-to-end with full reward in 5/10 seeded runs using only
  a local 14B model; the stream experiment replicates the no-transfer
  conclusion on the externally-scored environment.

## Quick start

```bash
pip install -e .[dev]
pytest            # schemas, CTS golden pack, stats, integration gates
bash reproduce.sh # manifest verification + figures + paper
```

## Structure

```
src/agent_ttrl/            # protocol machinery: runtime, CTS, ReplayBuffer, gates
scripts/                   # experiments: CTS streams (v3/v3.2/v3.0), tau2 official
                           #   pilots + streams, figure makers, local OpenAI server
protocols/runs/            # all run manifests (v1/v2/v3/v3.1/v3.2/v3q/tau2_official*)
paper/                     # ACL 2027 draft, 8 content pages, 9 figures
portfolio/                 # resume material
审稿意见/                   # five external reviews + REVIEW_RESPONSE.md mapping
```

## Protocol red lines (locked)

1. Evidence tiers: E_hard/E_soft enter adaptation; R_hidden (hidden
   evaluator) never enters rollout/branch/gradient/commit-gate.
2. Primary metric: inductive future transfer (prequential AUPC); the
   hidden evaluator scores first attempts offline, reporting only.
3. 3-channel budgets via one canonical ledger; matched caps for baselines.
4. Policy identity: rollouts bound to (base, adapter, policy_version);
   updates only at episode boundaries; base frozen, LoRA only.
5. ≥5 seeds, exact two-sided tests, common random numbers across arms
   (frozen 8/8 bitwise identical).

## Compute

Experiments ran on autodl2 (2×RTX 6000D 84GB, shared with GRPO-Guard per
the shared-card rules in `CLAUDE.md`). Serving: policy-consistent
`ColocatedPolicy` (HF/PEFT) directly, or through a local OpenAI-compatible
endpoint for the official tau2 pipeline (`scripts/tau2_local_server.py`).

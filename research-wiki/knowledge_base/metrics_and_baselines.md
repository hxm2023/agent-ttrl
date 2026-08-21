# Metrics and Baselines: Test-Time RL for Stateful Tool Agents

## Standard Evaluation Metrics

| Metric | Definition | Typical Range | What It Measures |
|--------|-----------|---------------|-----------------|
| pass@1 / task success | first-attempt task completion | 0–100% (AppWorld ~40–70 for 4-8B; τ²-retail lower) | static capability |
| AUPC_prequential (project) | mean first-attempt hidden score across stream tasks, each task scored before it enters any update | [0,1] | inductive future transfer — PRIMARY |
| sealed_future_holdout_score (project) | score on tasks never used in update/selector/gate/hyperparams | [0,1] | contamination-free confirmatory metric |
| same-task retry recovery | success after retry within the same task | [0,1] | within-task recovery (supplementary only) |
| catastrophic_update_rate (project) | fraction of committed candidates that drop anchor performance beyond pre-registered threshold or violate hard policy | [0,1] | safety of commits |
| false commit / false rollback | commit that hurt / rollback that rejected a genuine gain | [0,1] | gate quality |
| evolutionary gain / stability | SEA-Eval metrics over sequential task streams | unitless | genuine vs pseudo self-evolution |
| cumulative AUC (EvoTest/J-TTL) | area under episode-indexed performance | [0,1] | within-benchmark learning across episodes |
| cost / efficiency | generated tokens, tool calls, env steps, update FLOPs, GPU-h, wall time | — | matched-budget fairness |
| credit fidelity (project, mechanism) | sign accuracy + rank correlation of estimated credit vs exact oracle contribution | sign acc ∈[0,1], rank ∈[-1,1] | whether the credit signal is right |

## SOTA Baselines (by family, for our project)

### Family A — Inference / context adaptation (no parameter update)
| Method | Paper | Year | Key Performance | Code |
|--------|-------|------|-----------------|------|
| Frozen ReAct | — | — | AppWorld ~30–50% (model-dependent) | — |
| Best-of-N + verifier | — | — | gains with matched tokens | — |
| Reflexion / experience replay | — | 2023+ | memory-only gains | public |
| ACE | 2510.04618 | 2025 | AppWorld +10.6% (59.5%), beats GPT-4.1 agent with DeepSeek V3.1 | github.com/ace-agent/ace |
| JitRL | 2601.18510 | 2026 | SOTA training-free WebArena/Jericho; >30× cheaper than WebRL | github.com/liushiliushi/JitRL |

### Family B — Agent test-time adaptation (lightweight online update)
| Method | Paper | Year | Key Performance | Code |
|--------|-------|------|-----------------|------|
| GTTA (SA + DG) | 2511.04847 | 2025 (ICLR26) | WebArena 2%→23% without labels | github.com/r2llab/GTTA |
| OLIVIA (UCB action bandit) | 2605.11169 | 2026 | improves 4 tool-use benchmarks over static/prompt baselines | — |
| MemoPilot (GRPO memory copilot, frozen player) | 2606.08656 | 2026 (ICML26) | LHE Elo 1762; RPS@5 3.28; CoSQL 73.5% | — |
| aTTT (in-episode LoRA TTT) | 2607.03441 | 2026 | ALFWorld +5.0, SWE-bench Lite +4.9, 1.9× overhead | — |
| LEAFE (rollback-rebranch + SFT) | 2603.16843 | 2026 (ICML26) | up to +14% Pass@128 CodeContests/WebShop/ALFWorld; beats outcome RL | — |
| CausalFlow-style minimal repair/reuse | 2605.25338 | 2026 | 4 benchmarks, validated minimal repairs; DPO-style reuse | — |

### Family C — Test-time RL (parameter update)
| Method | Paper | Year | Key Performance | Code |
|--------|-------|------|-----------------|------|
| Naive TTRL (majority-vote GRPO) | 2504.16084 | 2025 (NeurIPS25) | AIME 2024 12.9→40.2% (Qwen2.5-Math-7B); math only | github.com/PRIME-RL/TTRL |
| T3RL (tool-verified weighting) | 2603.02203 | 2026 | +31.6% relative on AIME 2024; math only | — |
| DARE (distribution-aware TTRL reward) | 2601.21804 | 2026 | fixes MV bias/confirmation collapse | — |
| StarOR (test-time LoRA-GRPO + MCTS) | 2606.15197 | 2026 | SOTA on 5 optimization benchmarks, 4B backbone | — |
| SAGE (skill library GRPO, continual) | ACL 2026 | 2026 | AppWorld +8.9% goal completion, −26% steps, −59% tokens | github.com/amazon-science/SAGE |
| Prove / Synthesize-and-Reward | 2606.03892 | 2026 | 4 models on BFCL Multi-Turn, τ²-bench, T-Eval | — |

### Commit-gate / risk baselines (for our C2)
| Method | Paper | Year | Mechanism |
|--------|-------|------|-----------|
| Greedy always-commit | — | — | uncontrolled adaptive multiple testing (degrades) |
| PACE | 2606.08106 | 2026 | anytime-valid e-process gate, per-decision false-commit control |
| VaG | 2608.05810 | 2026 | verifier-as-gatekeeper pre-commit skill admission |
| Drift2Act | OpenReview | 2026 | online risk certificates gate TTA actions |
| STABLE | 2510.16089 | 2025 | gated LoRA merge (EM/bits/KL gates) for continual learning |
| Monitoring Risks in TTA | Schirmer et al. | 2025 | confidence sequences for changing models, no labels |

### Upper bounds / negative controls (project)
- hidden-oracle reward update (dev-only), all-turn exact branch, oracle commit, random labels/shuffled evidence, always-rollback.

## Evaluation Protocols (field conventions)

- **TTRL protocol**: unlabeled test set; sample n rollouts; majority-vote pseudo-label;
  GRPO on match/mismatch; iterate. Same-task evaluation only — reviewers criticize this.
- **Prequential/stream protocols**: SEA-Eval sequential task streams (evolutionary gain +
  stability); EvoTest J-TTL repeated-episode cumulative AUC; EvoPolicyGym fixed interaction
  budget then unseen generalization; Continual Learning Bench multi-episode shared env.
- **Our project protocol (design doc §8.4)**: first attempt with current adapter → hidden
  score recorded (algorithm-blind) → task completes → evidence allowed → branch/update →
  SafeCommit → next never-updated task. Roles: dev / calibration / adaptation_stream /
  sentinel_stream / candidate_audit / future_holdout, manifest-separated.
- **Budget accounting**: 3-channel hard caps B_env / B_model / B_update; one canonical
  ledger; no cross-channel exchange; baselines get the same caps.
- **Statistics**: ≥5 seeds (blinded power analysis), paired hierarchical bootstrap /
  mixed-effects, p<0.01 + effect size + CI, pre-registered primary endpoints.

## Known Traps (from field failures)

- Majority-vote self-consistency rewards amplify wrong consensus (T3RL §2; Amplification 2603.15417).
- In-episode self-training drifts when the agent is stuck (aTTT repetition filter is a symptom, not a gate).
- Post-hoc rollback of contaminated skills recovers little (VaG: contamination structurally irreversible) → pre-commit gating is necessary.
- Greedy "score went up" acceptance is uncontrolled adaptive multiple testing (PACE) → agents p-hack themselves.
- Same-task-only reporting = test-set memorization (design doc §1.2 forbidden forms).
- Parameter updates break static safety assumptions (TTT undermines guardrails 2605.22984; ASR 93–95% under few-shot/generation threat models).

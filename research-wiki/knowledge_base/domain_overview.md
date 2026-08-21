# Domain Overview: Test-Time RL for Stateful Tool-Using Agents

> KB built 2026-08-22 as Phase 0 of Agent-TTRL. Re-swept to current arXiv (last 2 months:
> 2026-06-22 → 2026-08-22). 30 papers downloaded and full-text indexed in `papers/`.
> This file is the field map; see `metrics_and_baselines.md` for protocols and
> `novelty/claim_matrix.csv` for the machine-readable claim matrix.

## What is this field?

Test-time learning for LLM agents asks: when a deployed agent continuously encounters
related-but-unseen stateful tool tasks, can it use only the evidence available at that
moment (partial, unreliable execution feedback) to improve on **future** tasks, within a
fixed interaction budget, without amplifying wrong rewards, negative transfer, or
catastrophic updates? The design space spans (a) whether parameters change (none /
activation / LoRA / memory text / prompt), (b) what the learning signal is (self-consistency,
tool verification, execution evidence, hidden outcome), (c) whether updates are gated before
they affect future behavior, and (d) how evaluation separates same-task recovery from
inductive future transfer.

The 2025–2026 field has moved from "can TTRL improve reasoning on unlabeled math" (TTRL,
Zuo et al. 2504.16084; T3RL, Liao et al. 2603.02203) toward **deployment-oriented
adaptation of agents in interactive environments** — but with a striking gap: most
mechanisms are validated at training time (Tree-RL, APPO, BiPACE, CVT-RL, CRAFT), most
deployment-time adaptation avoids gradient updates (JitRL, OLIVIA, ACE, MemoPilot), and
the few works that do update parameters in live episodes (aTTT: TTT-style self-supervised;
StarOR: per-instance LoRA-GRPO in optimization modeling) lack risk-controlled commit
protocols and prequential future-transfer evaluation. Commit gates exist (PACE, VaG,
Drift2Act, STABLE) but are applied to prompt/skill/context-level self-modification or
general TTA, not to parameter-level RL candidates under partial evidence.

## Key Concepts

- **Test-time inference/scaling** (no parameter change): more sampling, search, voting,
  verification. Baseline in our project.
- **Test-time context learning** (no parameter change): memory / playbook / prompt updates.
  Strong baselines: Reflexion, ACE (Agentic Context Engineering, 2510.04618), MemoPilot's
  player side (2606.08656).
- **Test-time adaptation/training** (parameter or activation change): TTT (aTTT 2607.03441,
  In-Place TTT 2604.06169), GTTA's adaptation vector (2511.04847), LoRA-TTT family,
  SEAL's self-edits (2506.10943).
- **Test-time reinforcement learning** (parameter update driven by interaction/reward):
  TTRL/T3RL (math), StarOR (2606.15197, optimization), MemoPilot's memory-copilot GRPO,
  our target setting.
- **Evidence tiers** (project-specific, from design doc §2.3): E_hard (schema/API/state
  invariants/receipts), E_soft (calibrated verifier), R_hidden (benchmark hidden evaluator,
  NEVER enters adaptation). T3RL is the canonical E_soft-weighted work; our E_hard/E_soft
  split for stateful agents is not covered elsewhere.
- **Counterfactual/paired-branch credit**: comparing rollouts from a shared decision state
  (fork/sibling/branch) to compute relative advantage. Tree-RL (2605.10913), APPO
  (2606.12384), CVT-RL PCCC (2606.05263), CRAFT (2606.29476), BiPACE (2606.25556),
  CausalFlow (2605.25338, offline repair), Counterfactual Shapley (2607.16999, general RL).
- **Commit gate / safe update**: deciding whether a candidate change may take effect.
  PACE e-process gate (2606.08106), VaG pre-commit verifier gating (2608.05810),
  Drift2Act risk certificates (OpenReview), STABLE gated LoRA merge (2510.16089),
  Monitoring Risks in TTA confidence sequences (Schirmer et al., ICML25 oral).
- **Prequential evaluation**: measure first-attempt performance on tasks before they enter
  any update; primary metric is inductive future transfer. Benchmarks: SEA-Eval (2604.08988),
  EvoTest/J-TTL (2510.13220), EvoPolicyGym (2607.02440), Continual Learning Bench (2606.05661).
- **Contamination/poisoning**: wrong-but-frequent evidence being reinforced into policy
  (T3RL's unverified-consensus bias; Amplification in TTRL 2603.15417; VaG's structural
  irreversibility of skill contamination 2608.05810; TTT undermining safety guardrails 2605.22984).

## Standard Methods (by adaptation substrate)

1. **Memory/prompt/context evolution** (frozen weights): Reflexion, ACE deltas, EvoTest
   Actor–Evolver, SEA-Eval-style self-evolving agents. Cheap; no risk to base weights;
   contamination lives in memory (can be reverted).
2. **Action-layer bandit** (frozen weights): OLIVIA (2605.11169) — UCB linear bandit over
   LLM hidden states at the action layer. Lightweight online update, no gradient.
3. **Non-parametric test-time RL** (frozen weights): JitRL (2601.18510) — memory-retrieved
   trajectory advantages modulate logits via closed-form KL-constrained solution; SOTA
   training-free on WebArena/Jericho; >30× cheaper than WebRL.
4. **LoRA parameter RL at test time**: StarOR (2606.15197) — transient LoRA updated by
   GRPO at each MCTS node, per instance, optimization modeling.
5. **TTT-style in-episode LoRA** (self-supervised, no reward): aTTT (2607.03441) — LoRA
   updates inside live episodes via vLLM runtime API; repetition-filtered reweighting to
   fight drift; ALFWorld +5.0, SWE-bench Lite +4.9.
6. **Training-time counterfactual credit RL** (our mechanism cell, but training): Tree-RL
   fork advantage, APPO branch-score + procedure-level advantage, CVT-RL PCCC
   (policy-conditioned counterfactual contribution + validity gating + doubly robust
   estimator), CRAFT sibling-rollout counterfactual credit, BiPACE action-conditioned
   Q̂−V̂. All improve GRPO for agents at training time.
7. **Offline counterfactual repair**: CausalFlow (2605.25338) — CRS via step-level
   counterfactual intervention, minimal repair, reuse as DPO/preference supervision.
8. **Skill-library / harness evolution**: SAGE (ACL 2026, GRPO + skill library, AppWorld
   +8.9% goal completion), Evo-Harness (2608.15071), EvoHarness-RL (2608.05446),
   SLAaaT (2608.17034, switching LoRA adapters as a tool).
9. **Self-modification commit gates**: PACE (2606.08106, anytime-valid e-process gate over
   prompt-level proposals), VaG (2608.05810, verifier-as-gatekeeper skill admission).

## Active Research Directions (post-2026-08 map)

- In-episode parameter updates for agents (aTTT, StarOR, TMEM 2606.04536) — hot, July 2026.
- Statistical commit gates for self-evolving agents (PACE, VaG, Drift2Act) — June–Aug 2026.
- Counterfactual credit assignment for agentic RL (CVT-RL, CRAFT, Tree-RL, APPO, BiPACE) —
  2026 wave; training-time so far.
- Prequential/streaming evaluation of self-improving agents (SEA-Eval, EvoTest, EvoPolicyGym).
- Safety of adaptive weights (Amplification in TTRL, TTT undermines guardrails).
- Deployment RL infrastructure (LEGO-RL 2608.17393, Agentic ESOpt 2608.17310,
  Wuying/DAO-GRPO 2608.17319).

## Key Venues

- ICML/NeurIPS/ICLR main: TTRL (NeurIPS25), SEAL (NeurIPS25), Monitoring Risks (ICML25
  oral + NeurIPS25), MemoPilot (ICML26), EvoTest (ICLR26), In-Place TTT (ICLR26),
  AutoTool (ICLR26), ParetoPO (ICML26 Spotlight), LEAFE (ICML26), Iterative RMFT (ICML26).
- ACL: SAGE (ACL 2026) — same-venue near-miss for our project (ACL 2027 primary).
- RLJ/RLC: Counterfactual Shapley (2607.16999).
- TMLR: Marvel safe O2O RL.
- OpenReview: Drift2Act, Monitoring Risks forum threads.

## Open wedge (candidate, pending Phase 1 debate)

No work found (through 2026-08-22) that combines ALL of:
deployment-stream **LoRA RL** × **partial/conflicting evidence** (E_hard+E_soft, R_hidden
excluded) × **paired counterfactual branches for signed action credit** × **risk-controlled
commit before policy change** × **inductive future transfer as primary prequential metric**
for **stateful tool agents** (AppWorld/τ²-class). The closest single works cover one or two
of these cells; see `novelty/claim_matrix.csv` for the per-work verdict.

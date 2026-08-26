# Stage 3 pre-registered predictions (written BEFORE the runs)

Written 2026-08-24 before launching the 18-run matrix, derived from the
Stage-2 evidence bridge (docs/evidence_bridge.md) and the M0/V001 exact
worlds. The predictions are falsifiable claims about the real-training
outcomes; the results section of the Stage 3 report compares them honestly.

## The mapping

| Stage-2 estimator (exact/MC) | Stage-3 credit scheme (real loop) |
|---|---|
| dense (unbiased, high cycle var 14.2 vs 1.5 at H=4) | dense GRPO (group advantage, all masked tokens) |
| local sibling (biased for full gradient, low var) | local-decision credit (advantage at tool-call tokens only) |
| paired replay (low var 1.5, small bias in observation-dependent worlds) | paired-branch gated credit (per-decision signed credit) |
| pc_rsg (biased + 1/q variance amplification) | not run (V001 already demonstrates its failure) |

## Predictions

1. **Ordering of final success (both tasks)**: paired >= dense > local.
   - Rationale: Stage 2 at horizon 12 (the real horizon, ~20-60 tokens of
     tool calls) showed paired-replay winning matched-budget MSE (0.011 vs
     0.046 on tool_selection_large) thanks to its cycle-variance advantage;
     dense is unbiased but high-variance (slowest stable progress); local is
     biased for the full gradient (T003) so it should make the least progress
     toward the verifiable reward.
2. **Gradient variance (grad_l2)**: dense > local >= paired.
   - The Stage-2 cycle variances rank dense 14.2 > paired 1.5 > sibling 0.55;
     in the real loop the per-step gradient norm is a proxy for that ordering.
3. **KL drift vs base**: local < dense < paired.
   - Local updates only a few positions per trajectory (fewest effective
     updates per step), so the smallest KL drift; paired's concentrated
     gated updates give the largest per-step drift.
4. **The paired gate will be closed (zero credit) in early epochs** when all
   branches succeed or all fail (degenerate groups), recovering later —
   a real-world manifestation of the reliability gate, NOT a bug.
5. **Invalid-tool-call rate**: dense and paired should converge to lower
   invalid rates than local (which only pressures the decision positions).

## Falsification conditions

- If dense beats paired on final success on BOTH tasks, prediction 1 is
  falsified (the unbiased-but-high-variance estimator wins in the real loop).
- If local achieves the highest final success on either task, the T003-style
  bias concern does not bite in this regime (prediction 1 partially void).
- If grad_l2 ordering does not match prediction 2, the Stage-2 cycle-variance
  ranking does not transfer to the real loop.

"""Stage 3 credit schemes: dense GRPO / local-decision / paired-branch.

The three schemes mirror the Auditor's Stage-2 estimator set (dense, local
sibling, paired replay) with the honest real-rollout mapping:

- dense:  standard GRPO — every masked token carries the group-relative
          advantage (R - group_mean) / group_std      [unbiased, high var]
- local:  the SAME advantage applied ONLY at decision-token positions (the
          tool-name tokens) — updates only the decision steps, never the
          intermediate prose  [biased for the full gradient, low var]
- paired: per-decision signed credit from the paired-branch reliability gate
          (agent_ttrl.credit.paired_credit over the branch-utility matrix),
          applied at the decision positions of each sample  [gated contrast]

The credit enters the loss as (mask, per-position advantage) over the
materialized ValidatedBatchHandle tensors; the ratio clipping and KL term
follow the Guard's grpo_loss conventions so the guarded-update contract is
unchanged.
"""

from __future__ import annotations

import re

import numpy as np


def decision_positions(text: str, tokenizer, completion_ids) -> list[int]:
    """Token indices where a tool-call decision starts: the offset of each
    tool name inside the completion, mapped through the tokenizer's
    offsets."""
    if not completion_ids:
        return []
    toks = tokenizer.convert_ids_to_tokens(completion_ids)
    # char offset of each token start within the decoded string
    decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)
    pos = 0
    token_starts: list[int] = []
    token_texts: list[str] = []
    for t in toks:
        t2 = t.replace("Ġ", " ")
        idx = decoded.find(t2, pos)
        if idx < 0:
            token_starts.append(pos)
        else:
            token_starts.append(idx)
            pos = idx + len(t2)
        token_texts.append(t2)
    # decision markers: tool names after '"tool"' or bare tool-name starts
    markers = []
    for m in re.finditer(r'"(?:tool|name)"\s*:\s*"', decoded):
        markers.append(m.end())
    for m in re.finditer(r"\b(reserve_item|create_order|charge|ship|complete_task|use_tool|lookup|refund|notify)\b", decoded):
        markers.append(m.start())
    # map each marker char offset to the token that contains it
    out = []
    for off in markers:
        for i, start in enumerate(token_starts):
            if start <= off < (token_starts[i + 1] if i + 1 < len(token_starts) else 10**9):
                out.append(i)
                break
    return sorted(set(out))


def dense_credit(utils_group: np.ndarray) -> np.ndarray:
    """Group-relative advantage per sample (the GRPO baseline)."""
    g = utils_group.reshape(-1, utils_group.shape[1] if utils_group.ndim > 1 else 1)
    mean = g.mean(axis=1, keepdims=True)
    std = g.std(axis=1, keepdims=True) + 1e-3
    return ((g - mean) / std).reshape(-1)


def local_credit(utils_group: np.ndarray) -> np.ndarray:
    """Local-decision variant: same advantage formula, applied only at
    decision positions (mask construction in train.py)."""
    return dense_credit(utils_group)


def paired_credit(U: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-decision signed credit from the paired-branch reliability gate.

    U: (decisions x branches) utility matrix. Returns (A per decision slot,
    gate info). A closed gate yields zero credit (honest no-update mode).
    """
    from agent_ttrl.credit.paired_credit import paired_credit as gate

    U = np.asarray(U, dtype=float)
    verdict = gate(U)
    info = {"status": verdict.status, "reason": verdict.reason_code}
    if verdict.status != "OK":
        return np.zeros(U.shape[0]), info
    credits = np.array([row.credit for row in verdict.rows])
    return credits, info

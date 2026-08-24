"""v2 P0 integration tests: policy-consistent runtime (A0/A1/A2).

A0: request-scoped RNG — same RequestSeed yields bitwise-identical
    generations; extra update-arm rollouts do NOT change future production
    first attempts (zero-LR matched control).
A1: served policy changes after commit — a real gradient step + atomic
    commit observably changes deterministic canary output.
A2: rollback restores the parent behavior.

These fix the v1 STATIC_SERVED_POLICY defect (trainer model never served).
Requires a GPU + small local model (default Qwen2.5-0.5B-Instruct).
"""
from __future__ import annotations

import hashlib
import os
import pytest
import torch

from agent_ttrl.runtime.request_seed import RequestSeed
from agent_ttrl.runtime.served_policy import ColocatedPolicy

MODEL = os.environ.get("ATTRL_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROTO = hashlib.sha256(b"agent-ttrl-v2-a0").hexdigest()
PROMPT = "List the tools you would call to refund order o1:"


def _seed(task: str, turn: int, purpose: str, v: int = 0, branch_group: int = 0) -> RequestSeed:
    return RequestSeed(PROTO, 0, task, turn, v, purpose, branch_group=branch_group)


def _policy():
    return ColocatedPolicy(MODEL, lora_rank=8, lora_alpha=16, device="cuda:0")


@pytest.mark.integration
def test_a0_same_seed_bitwise_identical():
    pol = _policy()
    s = _seed("t1", 0, "production_first_attempt")
    a, _ = pol.generate(s, PROMPT, max_tokens=32)
    b, _ = pol.generate(s, PROMPT, max_tokens=32)
    assert a == b, "same RequestSeed must be bitwise identical"
    del pol
    torch.cuda.empty_cache()


@pytest.mark.integration
def test_a0_extra_rollouts_do_not_change_production():
    pol = _policy()
    prod = _seed("t2", 0, "production_first_attempt")
    before, _ = pol.generate(prod, PROMPT, max_tokens=32)
    # update arm: consume extra RNG draws via credit rollouts (lr=0)
    for g in range(8):
        r = _seed("t2", 0, "credit_action_proposal", branch_group=g)
        pol.generate(r, PROMPT, max_tokens=16)
    after, _ = pol.generate(prod, PROMPT, max_tokens=32)
    assert before == after, "extra rollout RNG draws must not shift production first attempts"
    del pol
    torch.cuda.empty_cache()


def _overfit_step(pol):
    """Generate a completion, then train the model to RE-EMIT that same
    completion with high probability (a real overfit step on real tokens)."""
    s = RequestSeed(PROTO, 0, "overfit", 0, pol.policy_version, "canary")
    ids = pol.tokenizer(PROMPT, return_tensors="pt").input_ids.tolist()[0]
    cid, _ = pol.generate(s, PROMPT, max_tokens=24)
    assert cid, "overfit source completion must be non-empty"
    pol.train_step(ids, cid, advantage=1.0, lr=3e-3)


@pytest.mark.integration
def test_a1_commit_changes_served_output():
    pol = _policy()
    s = _seed("t3", 0, "production_first_attempt")
    before, _ = pol.generate(s, PROMPT, max_tokens=32)
    _overfit_step(pol)
    cand = pol.freeze_candidate()
    canary = RequestSeed(PROTO, 0, "canary1", 0, pol.policy_version, "canary")
    res = pol.commit(cand, PROMPT, canary)
    assert res.passed, f"canary failed: {res.reason}"
    after, _ = pol.generate(s, PROMPT, max_tokens=32)
    assert before != after, "committed adapter must observably change served output"
    del pol
    torch.cuda.empty_cache()


@pytest.mark.integration
def test_a2_rollback_restores_parent():
    pol = _policy()
    s = _seed("t4", 0, "production_first_attempt")
    before, _ = pol.generate(s, PROMPT, max_tokens=32)
    _overfit_step(pol)
    cand = pol.freeze_candidate()
    canary = RequestSeed(PROTO, 0, "canary2", 0, pol.policy_version, "canary")
    assert pol.commit(cand, PROMPT, canary).passed
    pol.rollback()
    after, _ = pol.generate(s, PROMPT, max_tokens=32)
    assert before == after, "rollback must restore parent behavior"
    del pol
    torch.cuda.empty_cache()

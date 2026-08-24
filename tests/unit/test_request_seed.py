"""Request-scoped RNG namespace: seeds isolate purposes and identities."""
import hashlib

import pytest

from agent_ttrl.runtime.request_seed import PURPOSES, RequestSeed

PROTO = hashlib.sha256(b"agent-ttrl-v2-unit").hexdigest()


def _s(**kw):
    base = dict(protocol_hash=PROTO, stream_seed=0, task_id="t1", turn_id=0,
                policy_version=0, purpose="production_first_attempt")
    base.update(kw)
    return RequestSeed(**base)


def test_purpose_namespace():
    assert len(PURPOSES) == 7
    a = _s().seed()
    b = _s(purpose="credit_action_proposal").seed()
    assert a != b


def test_identity_components_change_seed():
    base = _s()
    assert base.seed() != _s(task_id="t2").seed()
    assert base.seed() != _s(stream_seed=1).seed()
    assert base.seed() != _s(turn_id=1).seed()
    assert base.seed() != _s(branch_group=1).seed()
    assert base.seed() != _s(action_id=1).seed()
    assert base.seed() != _s(continuation_id=1).seed()


def test_policy_version_never_enters_seed():
    """v3 exogenous CRN: treatment (policy_version) must NOT change the
    production seed — frozen and update arms draw the same sequence."""
    assert _s().seed() == _s(policy_version=1).seed()
    assert _s().seed() == _s(policy_version=7).seed()


def test_extra_branches_do_not_change_production_seed():
    """The core v2 invariant: update-arm draws must not touch production."""
    prod = _s().seed()
    for g in range(16):
        _s(purpose="credit_continuation", branch_group=g, continuation_id=g % 2).seed()
    assert _s().seed() == prod


def test_invalid_purpose_rejected():
    with pytest.raises(ValueError):
        _s(purpose="not_a_purpose")

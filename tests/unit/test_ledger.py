"""CostLedger property tests (design doc §10.5, §18.2; fault F12)."""
import pytest

from agent_ttrl.cost.ledger import Channel, CostLedger


def make_ledger(caps=None):
    return CostLedger(caps=caps or {Channel.ENV: 100, Channel.MODEL: 10_000, Channel.UPDATE: 5_000})


def test_bill_and_totals():
    L = make_ledger()
    L.bill("op1", Channel.ENV, 4, "production")
    L.bill("op2", Channel.MODEL, 512, "branch")
    assert L.totals() == {"B_env": 4.0, "B_model": 512.0, "B_update": 0.0}
    assert L.within_caps()


def test_duplicate_op_rejected():
    L = make_ledger()
    L.bill("op1", Channel.ENV, 1, "production")
    with pytest.raises(ValueError, match="DUPLICATE_LEDGER_EVENT"):
        L.bill("op1", Channel.ENV, 1, "production")


def test_negative_bill_rejected():
    L = make_ledger()
    with pytest.raises(ValueError, match="NEGATIVE_BILL"):
        L.bill("op1", Channel.ENV, -1, "production")


def test_cap_violation_detected():
    L = make_ledger(caps={Channel.ENV: 10, Channel.MODEL: 10_000, Channel.UPDATE: 5_000})
    L.bill("op1", Channel.ENV, 11, "production")
    assert not L.within_caps()


def test_conservation_ok_with_external_tally():
    L = make_ledger()
    L.bill("op1", Channel.ENV, 4, "production")
    L.bill("op2", Channel.MODEL, 512, "branch")
    assert L.conservation_ok({"B_env": 4.0, "B_model": 512.0, "B_update": 0.0})
    # missed billing (external saw more) or double billing (external saw less) -> False
    assert not L.conservation_ok({"B_env": 5.0, "B_model": 512.0, "B_update": 0.0})
    assert not L.conservation_ok({"B_env": 4.0, "B_model": 0.0, "B_update": 0.0})


def test_no_cross_channel_exchange():
    """A channel over cap cannot be offset by another channel under cap."""
    L = make_ledger(caps={Channel.ENV: 10, Channel.MODEL: 10_000, Channel.UPDATE: 5_000})
    L.bill("op1", Channel.ENV, 12, "production")   # over cap
    L.bill("op2", Channel.MODEL, 1, "production")  # under cap
    assert not L.within_caps()


def test_event_log_hash_stable():
    L1, L2 = make_ledger(), make_ledger()
    for L in (L1, L2):
        L.bill("op1", Channel.ENV, 4, "production")
        L.bill("op2", Channel.MODEL, 512, "branch")
    assert L1.event_log_sha256() == L2.event_log_sha256()
    L2.bill("op3", Channel.UPDATE, 8, "update")
    assert L1.event_log_sha256() != L2.event_log_sha256()

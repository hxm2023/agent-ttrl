"""UpdateRow materialization tests (design doc §7.6; fault F04-analog)."""
import pytest

from agent_ttrl.credit.paired_credit import CreditRow
from agent_ttrl.credit.update_row import ActionSpan, PolicyIdentity, UpdateRowMaterializer, validate_batch_identity
from benchmarks.controlled_tool_shift.fixtures import HAND_U

IDENTITY = PolicyIdentity(base_sha256="a" * 64, adapter_sha256="b" * 64, policy_version="v1")


def span(start=10, end=20):
    return ActionSpan(producer="structured_recorder", start=start, end=end,
                      token_ids_ref="c" * 64, text_hash="d" * 64)


def mask_for(s: ActionSpan, length=30):
    return [1 if s.start <= i < s.end else 0 for i in range(length)]


def test_materialize_valid_row():
    m = UpdateRowMaterializer(IDENTITY, local_gate_kind="reliability_t")
    s = span()
    row = m.materialize("e" * 64, s, "f" * 64, mask_for(s),
                        CreditRow(0, 0.4, 0.4, True), ["g" * 64], "h" * 64)
    assert row["advantage"] == 0.4
    assert row["text_fallback"] is False
    assert row["policy_identity"]["base_sha256"] == "a" * 64
    assert len(row["row_id"]) == 64


def test_mask_scope_violation_rejected():
    """Observation/prefix tokens in the loss mask -> REJECT / ACTION_MASK_SCOPE."""
    m = UpdateRowMaterializer(IDENTITY, local_gate_kind="reliability_t")
    s = span()
    bad_mask = mask_for(s)
    bad_mask[3] = 1  # prefix token trained
    with pytest.raises(ValueError, match="ACTION_MASK_SCOPE"):
        m.materialize("e" * 64, s, "f" * 64, bad_mask, CreditRow(0, 0.4, 0.4, True), [], "h" * 64)


def test_gated_row_nonzero_advantage_rejected():
    m = UpdateRowMaterializer(IDENTITY, local_gate_kind="reliability_t")
    s = span()
    with pytest.raises(ValueError, match="GATED_ROW_NONZERO_ADVANTAGE"):
        m.materialize("e" * 64, s, "f" * 64, mask_for(s),
                      CreditRow(0, 0.4, 0.4, False), [], "h" * 64)


def test_empty_span_rejected():
    m = UpdateRowMaterializer(IDENTITY, local_gate_kind="reliability_t")
    s = span(10, 10)
    with pytest.raises(ValueError, match="EMPTY_ACTION_SPAN"):
        m.materialize("e" * 64, s, "f" * 64, mask_for(span()), CreditRow(0, 0.0, 0.0, True), [], "h" * 64)


def test_batch_identity_validation():
    m = UpdateRowMaterializer(IDENTITY, local_gate_kind="reliability_t")
    s = span()
    rows = [m.materialize("e" * 64, s, "f" * 64, mask_for(s),
                          CreditRow(0, 0.4, 0.4, True), [], "h" * 64) for _ in range(2)]
    validate_batch_identity(rows, IDENTITY)
    rows[1]["policy_identity"]["adapter_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="POLICY_IDENTITY_MISMATCH"):
        validate_batch_identity(rows, IDENTITY)


def test_hand_fixture_materialization_signs():
    """§7.5: pre-registered intervals -> alpha=[1,0,1] -> rows: +, dropped, -."""
    from benchmarks.controlled_tool_shift.fixtures import HAND_EXPECTED_ALPHA, HAND_INTERVALS
    means = HAND_U.mean(axis=1)
    raw = means - means.mean()                     # [+0.4, +0.1, -0.5]
    alpha = [1 if (lo > 0.05 or hi < -0.05) else 0 for lo, hi in HAND_INTERVALS]
    assert alpha == HAND_EXPECTED_ALPHA
    m = UpdateRowMaterializer(IDENTITY, local_gate_kind="reliability_t")
    signs = []
    for i, (a, c) in enumerate(zip(alpha, raw)):
        s = span(10 + i, 12 + i)
        if a:
            credit = CreditRow(i, c, c, True)
            row = m.materialize("e" * 64, s, "f" * 64, mask_for(s), credit, [], "h" * 64)
            signs.append(1 if row["advantage"] > 0 else (-1 if row["advantage"] < 0 else 0))
        else:
            signs.append(0)
    assert signs == [1, 0, -1]

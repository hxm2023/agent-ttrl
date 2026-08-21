"""CTS golden acceptance (design doc §9.1.2): F01-F09 credit signs correct;
F10-F12 fail closed with fixed reason codes; hand fixture token-level signs.
"""
import numpy as np
import pytest

from agent_ttrl.credit.paired_credit import ETA_CREDIT, paired_credit
from agent_ttrl.environments.cts_runner import run_fixture
from benchmarks.controlled_tool_shift.fixtures import (
    FIXTURES, HAND_EXPECTED_ALPHA, HAND_INTERVALS, HAND_U, CTSFixture,
)

F01_F09 = [f for f in FIXTURES if f.fid in {f"CTS-F0{i}" for i in range(1, 10)}]
FAIL_CLOSED = [f for f in FIXTURES if f.fid in {"CTS-F10", "CTS-F11", "CTS-F12"}]


@pytest.mark.parametrize("fx", F01_F09, ids=lambda f: f.fid)
def test_credit_signs(fx: CTSFixture):
    run = run_fixture(fx)
    assert run.verdict.status in ("OK", "NO_RELIABLE_CREDIT"), \
        f"{fx.fid}: {run.verdict.status}/{run.verdict.reason_code} errors={run.errors}"
    signs = [int(np.sign(r.credit)) for r in (run.verdict.rows or [])]
    assert signs == fx.expected_signs, f"{fx.fid}: signs={signs} raw={[r.raw_credit for r in (run.verdict.rows or [])]}"
    if run.oracle_canonical is not None:
        assert run.oracle_canonical == fx.hidden_outcome, f"{fx.fid}: oracle={run.oracle_canonical}"


def test_f04_duplicate_penalized():
    fx = next(f for f in FIXTURES if f.fid == "CTS-F04")
    run = run_fixture(fx)
    assert run.verdict.rows[1].credit < run.verdict.rows[0].credit


def test_f08_conflict_recorded():
    fx = next(f for f in FIXTURES if f.fid == "CTS-F08")
    run = run_fixture(fx)
    assert fx.assert_conflict
    assert run.conflicts, "poisoned receipt vs DB projection conflict must be recorded"


def test_f08_proxy_hidden_disagreement():
    """Trusting the poisoned receipt must fail the hidden oracle (receipt vs DB invariant)."""
    fx = next(f for f in FIXTURES if f.fid == "CTS-F08")
    run = run_fixture(fx)
    assert run.oracle_canonical is False  # naive (trust-receipt) branch fails hidden oracle
    assert run.U[1].mean() > run.U[0].mean()  # charging actually pays -> higher evidence utility


@pytest.mark.parametrize("fx", FAIL_CLOSED, ids=lambda f: f.fid)
def test_fail_closed(fx: CTSFixture):
    run = run_fixture(fx, pollute_parent=(fx.fid == "CTS-F12"))
    assert fx.fail_closed_reason in f"{run.verdict.status}/{run.verdict.reason_code}", \
        f"{fx.fid}: got {run.verdict.status}/{run.verdict.reason_code}"


def test_branch_restore_deterministic():
    """Same sealed snapshot + same action/seed -> identical U (design doc §18.2)."""
    fx = next(f for f in FIXTURES if f.fid == "CTS-F01")
    u1 = run_fixture(fx).U
    u2 = run_fixture(fx).U
    assert np.array_equal(u1, u2)


def test_hand_fixture_alpha():
    """§7.5 three-action fixture: pre-registered intervals -> alpha=[1,0,1] at eta=0.05."""
    eta = 0.05
    means = HAND_U.mean(axis=1)
    np.testing.assert_allclose(means, [0.9, 0.6, 0.0], atol=1e-9)
    np.testing.assert_allclose(means - means.mean(), [0.4, 0.1, -0.5], atol=1e-9)
    alpha = []
    for (lo, hi) in HAND_INTERVALS:
        # gate: interval disjoint from [-eta, +eta]
        alpha.append(1 if (lo > eta or hi < -eta) else 0)
    assert alpha == HAND_EXPECTED_ALPHA, f"alpha={alpha}"


def test_noop_group_no_update():
    """F10/F11 must not produce any update rows."""
    for fid in ("CTS-F10", "CTS-F11"):
        fx = next(f for f in FIXTURES if f.fid == fid)
        run = run_fixture(fx)
        assert run.verdict.status == "DEGENERATE_GROUP"
        assert run.verdict.rows is None or all(r.credit == 0.0 for r in run.verdict.rows)

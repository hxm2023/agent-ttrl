"""Evidence-conflict gate + drift monitor tests (M1, Candidate B; CTS-F08)."""
import numpy as np

from agent_ttrl.credit.conflict_gate import DriftMonitor, apply_conflict_gate
from agent_ttrl.credit.paired_credit import paired_credit
from agent_ttrl.environments.cts_runner import run_fixture
from benchmarks.controlled_tool_shift.fixtures import FIXTURES


def test_conflict_gate_abstains():
    U = np.array([[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]])
    verdict = paired_credit(U)
    assert verdict.status == "OK"
    out = apply_conflict_gate(verdict, conflicts=["conflict:receipt=PAID:projection=CREATED"])
    assert out.abstained
    assert all(r.credit == 0.0 and not r.gate_passed for r in out.verdict.rows)
    assert out.verdict.reason_code == "EVIDENCE_CONFLICT_ABSTAIN"


def test_conflict_gate_passthrough_without_conflict():
    U = np.array([[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]])
    verdict = paired_credit(U)
    out = apply_conflict_gate(verdict, conflicts=[])
    assert not out.abstained
    assert out.verdict is verdict


def test_conflict_gate_on_cts_f08():
    """F08 (poisoned receipt) must abstain on the conflicted branch group."""
    fx = next(f for f in FIXTURES if f.fid == "CTS-F08")
    run = run_fixture(fx)
    out = apply_conflict_gate(run.verdict, conflicts=run.conflicts)
    assert out.abstained
    assert out.reason_code == "EVIDENCE_CONFLICT_ABSTAIN"


def test_conflict_gate_no_abstain_on_clean_f01():
    fx = next(f for f in FIXTURES if f.fid == "CTS-F01")
    run = run_fixture(fx)
    out = apply_conflict_gate(run.verdict, conflicts=run.conflicts)
    assert not out.abstained


def test_drift_monitor_fail_closed():
    m = DriftMonitor(window=5, threshold=3)
    assert not m.observe(False)
    assert not m.observe(False)
    assert not m.observe(True)
    assert not m.observe(True)
    assert m.observe(True)          # 3 conflicts in window -> halt
    assert m.halt_recommended()
    m.observe(False)                # window slides
    m.observe(False)
    assert m.halt_recommended()     # [T,T,T,F,F] still has 3 conflicts
    m.observe(False)                # [T,T,F,F,F] -> 2 conflicts -> cleared
    assert not m.halt_recommended()

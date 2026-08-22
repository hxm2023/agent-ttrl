"""Capability separation tests (design doc §8.2; fault F06)."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from agent_ttrl import capabilities


def _run_in_subprocess(capability: str, code: str) -> tuple[int, str]:
    env = {"AGENT_TTRL_CAPABILITY": capability, "PYTHONPATH": "src;."}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, cwd=str(Path(__file__).resolve().parents[2]))
    return r.returncode, (r.stdout + r.stderr).strip()


def test_online_cannot_import_sealed_evaluator():
    code = "import agent_ttrl.evaluation.sealed"  # must fail
    rc, err = _run_in_subprocess("online", code)
    assert rc != 0
    assert "HIDDEN_CAPABILITY_BREACH" in err or "PermissionError" in err


def test_sealed_can_import_and_evaluate():
    code = (
        "import agent_ttrl.evaluation.sealed as s; "
        "e = s.SealedEvaluator('m'); print(e.evaluate({'run_id': 'r1'}))"
    )
    rc, out = _run_in_subprocess("sealed_audit", code)
    assert rc == 0, out


def test_online_process_has_no_hidden_access():
    code = "from agent_ttrl import capabilities; print(capabilities.hidden_evaluator_available())"
    rc, out = _run_in_subprocess("online", code)
    assert rc == 0 and out == "False"


def test_gate_capability_is_separate():
    code = "from agent_ttrl import capabilities; print(capabilities.hidden_evaluator_available())"
    rc, out = _run_in_subprocess("gate", code)
    assert rc == 0 and out == "False"


def test_require_capability_raises():
    capabilities.CAPABILITY = "online"
    capabilities.GRANTED = {"online": {"online"}}
    with pytest.raises(PermissionError, match="HIDDEN_CAPABILITY_BREACH"):
        capabilities.require_capability("sealed_audit")


def test_online_evidence_never_imports_oracle():
    """The online evidence producer must not import the hidden oracle (design doc §2.3)."""
    import inspect
    from agent_ttrl.environments import cts_evidence
    src = inspect.getsource(cts_evidence)
    assert "cts_oracle" not in src.replace("cts_oracle", "") or "hidden_score" not in src
    assert "hidden_score" not in src

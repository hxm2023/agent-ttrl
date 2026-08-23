"""Regression guards for stream-script audit findings (2026-08-23):

1. tau2 egc dead-code bug: the update block was gated on variant=="naive",
   making egc a frozen-equivalent no-op. The gate must cover ("naive","egc").
2. m2 reflexion leak: memory formation must never be gated on the hidden
   evaluator verdict (protocol red line 1).
3. m3 no-op honesty: a zero-gradient-token update must be recorded as
   updated=False, not updated=True.
"""
import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_tau2_update_block_covers_egc():
    src = (SCRIPTS / "tau2_agent_stream.py").read_text(encoding="utf-8")
    assert re.search(r'if args\.variant in \("naive", "egc"\)', src), (
        "tau2 update block must run for both naive and egc variants")


def test_reflexion_memory_not_gated_on_hidden():
    src = (SCRIPTS / "m2_baselines.py").read_text(encoding="utf-8")
    assert 'if VARIANT == "reflexion" and info["errors"]:' in src, (
        "reflexion memory must be gated on E_hard errors only")
    leak = re.search(r'if VARIANT == "reflexion".*?not info\["hidden"\].*?info\["errors"\]', src)
    assert leak is None, "reflexion memory must never reference the hidden verdict"


def test_m3_noop_update_recorded_honestly():
    src = (SCRIPTS / "m3_stream_pilot.py").read_text(encoding="utf-8")
    assert 'if metrics["tokens"] > 0:' in src
    assert '"reason": "NO_GRADIENT_TOKENS"' in src, (
        "zero-gradient-token updates must record updated=False with a reason")

"""Sealed hidden evaluator (offline-only; fault F06 guard on import).

Only importable in a sealed_audit (or dev) process. The online process that
imports this module must fail closed, and any attempt to evaluate a candidate
from an online process is a HIDDEN_CAPABILITY_BREACH.
"""
from __future__ import annotations

from agent_ttrl import capabilities

capabilities.require_capability("sealed_audit")


class SealedEvaluator:
    """Runs the hidden evaluator over candidate adapters / final adapters.

    Instantiation checks the capability at call time as well as at import, so a
    stale process cannot evaluate after capability revocation.
    """

    def __init__(self, manifest_ref: str):
        capabilities.require_capability("sealed_audit")
        if capabilities.hidden_evaluator_available() is False:
            raise PermissionError("HIDDEN_CAPABILITY_BREACH")
        self.manifest_ref = manifest_ref

    def evaluate(self, run_manifest: dict) -> dict:
        capabilities.require_capability("sealed_audit")
        return {"run_id": run_manifest.get("run_id"), "sealed": True}

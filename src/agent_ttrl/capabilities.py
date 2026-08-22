"""Capability separation (design doc §8.2, fault F06).

The online/training process must never be able to import or invoke the hidden
evaluator. Capabilities are granted by the process environment:
  AGENT_TTRL_CAPABILITY = online | gate | sealed_audit | dev
The online entry point sets "online"; the sealed evaluator module refuses to
load without "sealed_audit" (or "dev" for local correctness work).
"""
from __future__ import annotations

import os

CAPABILITY = os.environ.get("AGENT_TTRL_CAPABILITY", "online")

GRANTED = {
    "online": {"online"},
    "gate": {"gate"},
    "sealed_audit": {"sealed_audit"},
    "dev": {"online", "gate", "sealed_audit"},
}


def require_capability(name: str) -> None:
    if name not in GRANTED.get(CAPABILITY, set()):
        raise PermissionError(
            f"HIDDEN_CAPABILITY_BREACH: capability '{name}' required but process "
            f"capability is '{CAPABILITY}'")


def hidden_evaluator_available() -> bool:
    """True only in sealed_audit (or dev) processes."""
    return "sealed_audit" in GRANTED.get(CAPABILITY, set())


def online_capability() -> bool:
    return "online" in GRANTED.get(CAPABILITY, set())

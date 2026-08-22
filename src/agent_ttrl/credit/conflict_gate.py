"""Evidence-conflict local gate (M1, Candidate B) + drift monitor.

When E_hard internal evidence conflicts (tool receipt vs state projection,
e.g., poisoned success receipts), the local gate abstains from gradient for
the whole branch group and feeds a drift counter that can halt further
adaptation (domain-level alarm). Falls back to the t-interval reliability
gate when no conflict is present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_ttrl.credit.paired_credit import GroupVerdict


@dataclass
class ConflictGateOutcome:
    abstained: bool
    reason_code: str | None
    verdict: GroupVerdict
    conflicts: list[str] = field(default_factory=list)


def apply_conflict_gate(verdict: GroupVerdict, conflicts: list[str],
                        abstain_on_conflict: bool = True) -> ConflictGateOutcome:
    """Zero out all credit rows when the decision-state evidence is conflicted."""
    if abstain_on_conflict and conflicts:
        rows = []
        for r in (verdict.rows or []):
            rows.append(type(r)(action_idx=r.action_idx, credit=0.0,
                                raw_credit=r.raw_credit, gate_passed=False,
                                reason="EVIDENCE_CONFLICT_ABSTAIN"))
        zeroed = GroupVerdict(status="OK", reason_code="EVIDENCE_CONFLICT_ABSTAIN", rows=rows)
        return ConflictGateOutcome(abstained=True, reason_code="EVIDENCE_CONFLICT_ABSTAIN",
                                   verdict=zeroed, conflicts=conflicts)
    return ConflictGateOutcome(abstained=False, reason_code=None,
                               verdict=verdict, conflicts=conflicts)


@dataclass
class DriftMonitor:
    """Domain-level alarm: accumulated evidence conflicts can halt adaptation.

    fail-closed semantics (design doc §6.4): when the verifier/evidence is
    unreliable, stop parameter updates instead of reinforcing wrong consensus.
    """
    window: int = 5
    threshold: int = 3
    _recent: list[bool] = field(default_factory=list)

    def observe(self, conflicted: bool) -> bool:
        """Record one decision's conflict state; return True when adaptation
        should halt (fail closed)."""
        self._recent.append(conflicted)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        return sum(self._recent) >= self.threshold

    def halt_recommended(self) -> bool:
        return sum(self._recent) >= self.threshold

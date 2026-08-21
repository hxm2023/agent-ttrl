"""Accessible evidence producers for ControlledToolShift.

Online agent sees tool-returned receipts + an allowed state projection; NEVER the
oracle. Evidence utility g(e) = w^T e, normalized to [0,1] on calibration split
(λ_c=0). Conflict detection compares tool receipts against state projections
(poisoned receipts vs DB invariants).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cts_oracle import GoalSpec
from .cts_world import OrderStatus, WorldState


@dataclass
class EvidenceBundle:
    hard_evidence: list[dict] = field(default_factory=list)   # receipts + projections
    soft_evidence: list[dict] = field(default_factory=list)   # calibrated verifier scores

    @property
    def raw(self) -> dict:
        return {"hard_evidence": self.hard_evidence, "soft_evidence": self.soft_evidence}


class AccessibleEvidence:
    """What the agent can observe: status receipts + partial projections."""

    def __init__(self, state: WorldState, goal: GoalSpec):
        self.state = state
        self.goal = goal

    def collect(self, tool_receipts: list[dict] | None = None) -> EvidenceBundle:
        bundle = EvidenceBundle()
        for oid in sorted(self.state.order_status):
            st = self.state.order_status[oid]
            bundle.hard_evidence.append({
                "producer": "state_projection",
                "value": f"status:{st}",
                "unit": None,
                "allowed_in_gradient": True,
            })
            if st == OrderStatus.SHIPPED.value:
                bundle.hard_evidence.append({
                    "producer": "state_projection",
                    "value": f"shipped:{self.state.reservation.get(oid)}",
                    "unit": None,
                    "allowed_in_gradient": True,
                })
                bundle.hard_evidence.append({
                    "producer": "state_projection",
                    "value": f"shipped_address:{self.state.address.get(self.goal.user_id)}",
                    "unit": None,
                    "allowed_in_gradient": True,
                })
        for flag in self.state.audit_flags:
            if flag.startswith("duplicate_reserve"):
                bundle.hard_evidence.append({
                    "producer": "state_projection",
                    "value": flag,
                    "unit": None,
                    "allowed_in_gradient": True,
                })
        for r in tool_receipts or []:
            bundle.hard_evidence.append({
                "producer": "tool_receipt",
                "value": r.get("status", r.get("type", "receipt")),
                "unit": None,
                "allowed_in_gradient": True,
                "receipt_type": r.get("type"),
            })
        # verifier: goal-relevant soft signal derived ONLY from observable projections
        observed_shipped = [self.state.reservation.get(oid) for oid in self.state.order_status
                            if self.state.order_status[oid] == OrderStatus.SHIPPED.value]
        soft = 1.0 if self.goal.want_item in observed_shipped else 0.0
        bundle.soft_evidence.append({
            "producer": "cts_verifier_v1",
            "score": soft,
            "calibration_version": "cal:cts:v1",
            "allowed_in_gradient": True,
        })
        return bundle


def evidence_utility(bundle: EvidenceBundle, goal: GoalSpec) -> float:
    """g(e) in [0,1]: goal-relevant hard evidence dominates; verifier adds agreement."""
    hard = bundle.hard_evidence
    shipped_entries = [e for e in hard if e["producer"] == "state_projection"
                       and str(e["value"]).startswith("shipped:")]
    goal_ok = any(goal.want_item in str(e["value"]) for e in shipped_entries)
    if goal.want_cancelled:
        goal_ok = any(str(e["value"]) == "status:CANCELLED" for e in hard)
    if goal.want_address is not None:
        addr_ok = any(str(e["value"]) == f"shipped_address:{goal.want_address}" for e in hard)
        goal_ok = goal_ok and addr_ok
    shipped_count = sum(1 for e in hard if str(e["value"]) == "status:SHIPPED")
    collateral_proxy = sum(1 for e in hard if str(e["value"]) == "status:CANCELLED")
    duplicate_proxy = sum(1 for e in hard
                          if e["producer"] == "state_projection" and "duplicate_reserve" in str(e["value"]))
    soft = bundle.soft_evidence[0]["score"] if bundle.soft_evidence else 0.0
    u = (0.5 * float(goal_ok) + 0.25 * min(shipped_count, 2) / 2 + 0.25 * soft
         - 0.1 * collateral_proxy - 0.3 * duplicate_proxy)
    return max(0.0, min(1.0, u))


def conflict_flags(bundle: EvidenceBundle) -> list[str]:
    """E_hard internal conflicts: tool-returned receipt status vs state projection."""
    flags = []
    projections = {str(e["value"]) for e in bundle.hard_evidence
                   if e["producer"] == "state_projection" and str(e["value"]).startswith("status:")}
    for e in bundle.hard_evidence:
        if e["producer"] != "tool_receipt":
            continue
        rtype = e.get("receipt_type")
        if rtype not in ("status_receipt", "charge_receipt", "cancel_receipt", "order_receipt"):
            continue
        rstatus = str(e["value"])
        if rstatus.startswith("status:"):
            rstatus = rstatus.split(":", 1)[1]
        if rtype == "charge_receipt" and rstatus == "PENDING":
            continue  # delay_v1: PENDING is not a conflict
        if rtype == "charge_receipt" and rstatus == "PAID":
            if "status:CREATED" in projections and "status:PAID" not in projections:
                flags.append("conflict:receipt=PAID:projection=CREATED")
        if rtype == "status_receipt":
            # poison signature: receipt claims PAID while the DB projection is CREATED
            if rstatus == "PAID" and "status:CREATED" in projections and "status:PAID" not in projections:
                flags.append("conflict:receipt=PAID:projection=CREATED")
    return flags

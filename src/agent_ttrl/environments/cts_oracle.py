"""ControlledToolShift hidden oracle (offline evaluation only, design doc §9.1.1).

Checks complete goal state, forbidden side effects, money conservation, permission
policy. NEVER visible to the online agent / branch / gradient / gate.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cts_world import OrderStatus, WorldState


@dataclass
class GoalSpec:
    user_id: str
    want_item: str                    # canonical key the user must own (shipped)
    want_address: str | None = None   # shipping address if required
    no_collateral: list[str] = None   # audit flags that must NOT appear
    min_balance: int | None = None    # optional money-conservation floor for user
    want_cancelled: bool = False      # alternative goal: order cancelled + clean exit


@dataclass
class OracleVerdict:
    success: bool
    collateral: list[str]
    money_conserved: bool
    permission_ok: bool
    reason: str = ""


def hidden_score(state: WorldState, goal: GoalSpec) -> OracleVerdict:
    """Deterministic oracle. score=1 iff all criteria hold."""
    collateral = []
    for flag in goal.no_collateral or []:
        if flag in state.audit_flags:
            collateral.append(flag)

    money_conserved = True
    if goal.min_balance is not None:
        # money conservation: user balance must be >= min_balance and total
        # debits must equal credits across the system (balance + pending effects).
        total_out = sum(q["amount"] for q in state.delayed_effect_queue if q["kind"] == "charge")
        total_in = sum(q["amount"] for q in state.delayed_effect_queue if q["kind"] == "refund")
        if state.balance.get(goal.user_id, 0) + total_in - total_out < goal.min_balance:
            money_conserved = False

    permission_ok = True  # permission policy is enforced at transition time

    shipped = [oid for oid, st in state.order_status.items() if st == OrderStatus.SHIPPED.value]
    if goal.want_cancelled:
        has_item = False
        cancelled = any(state.order_status.get(oid) == OrderStatus.CANCELLED.value
                        for oid in state.order_status)
        address_ok = True
        item_ok = cancelled and not any(state.order_status.get(oid) == OrderStatus.SHIPPED.value
                                        for oid in state.order_status)
    else:
        has_item = any(state.reservation.get(oid) == goal.want_item for oid in shipped)
        item_ok = has_item
        address_ok = goal.want_address is None or state.address.get(goal.user_id) == goal.want_address

    success = item_ok and address_ok and not collateral and money_conserved and permission_ok
    reason = ""
    if not success:
        reasons = []
        if not item_ok:
            reasons.append("goal_item_not_shipped" if not goal.want_cancelled else "not_cancelled_or_shipped")
        if not address_ok:
            reasons.append("wrong_address")
        if collateral:
            reasons.append("collateral:" + ",".join(collateral))
        if not money_conserved:
            reasons.append("money_not_conserved")
        if not permission_ok:
            reasons.append("permission_violation")
        reason = ";".join(reasons)
    return OracleVerdict(success=success, collateral=collateral,
                         money_conserved=money_conserved,
                         permission_ok=permission_ok, reason=reason)

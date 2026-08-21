"""ControlledToolShift: deterministic, snapshotable tool world (design doc §9.1.1).

WorldState + 10 pure-function tools + shift families (syntax_v1 / dynamics_v1 /
delay_v1 / permission_v1 / poison_v1) + before/after state hashes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrderStatus(str, Enum):
    NONE = "NONE"
    CREATED = "CREATED"
    PAID = "PAID"
    SHIPPED = "SHIPPED"
    CANCELLED = "CANCELLED"


class ShiftFamily(str, Enum):
    NONE = "none"
    SYNTAX_V1 = "syntax_v1"        # sku -> item_id; enum renames
    DYNAMICS_V1 = "dynamics_v1"    # cancel/refund ordering rules change
    DELAY_V1 = "delay_v1"          # charge takes effect after 2 turns
    PERMISSION_V1 = "permission_v1"  # high-risk actions need new scope
    POISON_V1 = "poison_v1"        # receipts wrong w/ frozen prob; DB correct


@dataclass
class WorldState:
    schema_version: str = "cts_v0"
    turn: int = 0
    inventory: dict[str, int] = field(default_factory=dict)          # sku -> stock
    reservation: dict[str, str] = field(default_factory=dict)        # order_id -> sku
    order_status: dict[str, str] = field(default_factory=dict)       # order_id -> OrderStatus
    balance: dict[str, int] = field(default_factory=dict)            # user_id -> cents
    address: dict[str, str] = field(default_factory=dict)            # user_id -> address
    delayed_effect_queue: list[dict[str, Any]] = field(default_factory=list)
    permission_scope: list[str] = field(default_factory=list)
    audit_flags: list[str] = field(default_factory=list)

    def canonical(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "turn": self.turn,
                "inventory": dict(sorted(self.inventory.items())),
                "reservation": dict(sorted(self.reservation.items())),
                "order_status": dict(sorted(self.order_status.items())),
                "balance": dict(sorted(self.balance.items())),
                "address": dict(sorted(self.address.items())),
                "delayed_effect_queue": self.delayed_effect_queue,
                "permission_scope": sorted(self.permission_scope),
                "audit_flags": sorted(self.audit_flags),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def copy(self) -> "WorldState":
        return WorldState(
            schema_version=self.schema_version,
            turn=self.turn,
            inventory=dict(self.inventory),
            reservation=dict(self.reservation),
            order_status=dict(self.order_status),
            balance=dict(self.balance),
            address=dict(self.address),
            delayed_effect_queue=[dict(q) for q in self.delayed_effect_queue],
            permission_scope=list(self.permission_scope),
            audit_flags=list(self.audit_flags),
        )


def advance_turn(state: WorldState, turns: int = 1) -> WorldState:
    """Advance the episode turn counter and apply due delayed effects."""
    new = state.copy()
    for _ in range(turns):
        new.turn += 1
        new = _apply_delayed(new, turns_elapsed=new.turn)
    return new


# ---------------------------------------------------------------------------
# Shift config
# ---------------------------------------------------------------------------

@dataclass
class ShiftConfig:
    syntax: ShiftFamily = ShiftFamily.NONE
    dynamics: ShiftFamily = ShiftFamily.NONE
    delay: ShiftFamily = ShiftFamily.NONE
    permission: ShiftFamily = ShiftFamily.NONE
    poison: ShiftFamily = ShiftFamily.NONE
    poison_prob: float = 0.0
    poison_rng_seed: int = 0

    @property
    def id(self) -> str:
        return "+".join(
            f for f in [self.syntax.value, self.dynamics.value, self.delay.value,
                        self.permission.value, self.poison.value]
            if f != "none"
        ) or "none"


def _apply_delayed(state: WorldState, turns_elapsed: int = 0) -> WorldState:
    """Apply due delayed effects (delay_v1: effects scheduled 2 turns ahead)."""
    new = state.copy()
    still_pending: list[dict[str, Any]] = []
    for q in new.delayed_effect_queue:
        q = dict(q)
        if q["due_turn"] <= turns_elapsed:
            kind = q["kind"]
            if kind == "charge":
                uid = q["user_id"]
                new.balance[uid] = new.balance.get(uid, 0) - q["amount"]
                oid = q["order_id"]
                new.order_status[oid] = OrderStatus.PAID.value
                new.audit_flags.append(f"delayed_charge_applied:{oid}")
            elif kind == "refund":
                uid = q["user_id"]
                new.balance[uid] = new.balance.get(uid, 0) + q["amount"]
                new.audit_flags.append(f"delayed_refund_applied:{uid}")
            else:
                raise ValueError(f"unknown delayed effect {kind}")
        else:
            still_pending.append(q)
    new.delayed_effect_queue = still_pending
    return new


def _normalize_key(sku: str, config: ShiftConfig) -> str:
    """Key dialect. Base config uses sku:*; syntax_v1 requires item:* (old calls fail)."""
    if config.syntax == ShiftFamily.SYNTAX_V1:
        if sku.startswith("sku:"):
            raise ValueError("ITEM_NOT_FOUND")
        return sku
    return sku


# ---------------------------------------------------------------------------
# Tool transitions (pure)
# ---------------------------------------------------------------------------

def transition(state: WorldState, tool: str, call: dict[str, Any], config: ShiftConfig) -> tuple[WorldState, list[dict[str, Any]]]:
    """Pure transition. Returns (new_state, receipts). Raises ValueError on invalid call."""
    new = _apply_delayed(state.copy())
    receipts: list[dict[str, Any]] = []

    if tool == "lookup_item":
        key = _normalize_key(call["item_key"], config)
        if key not in new.inventory:
            raise ValueError("ITEM_NOT_FOUND")
        receipts.append({"type": "item_summary", "item_key": key, "stock_bucket": _bucket(new.inventory[key])})
        return new, receipts

    if tool == "reserve_item":
        key = _normalize_key(call["item_key"], config)
        oid = call["order_id"]
        if new.inventory.get(key, 0) <= 0:
            raise ValueError("OUT_OF_STOCK")
        duplicate = oid in new.reservation
        if duplicate:
            new.audit_flags.append(f"duplicate_reserve:{oid}")
        new.inventory[key] -= 1
        new.reservation[oid] = key
        new.order_status[oid] = OrderStatus.CREATED.value
        new.audit_flags.append(f"reserve:{oid}:{key}")
        receipts.append({"type": "reservation_receipt", "order_id": oid, "item_key": key,
                         "duplicate": duplicate})
        return new, receipts

    if tool == "release_reservation":
        oid = call["order_id"]
        if oid not in new.reservation:
            raise ValueError("NO_RESERVATION")
        if new.order_status[oid] == OrderStatus.SHIPPED.value:
            raise ValueError("ALREADY_SHIPPED")
        key = new.reservation.pop(oid)
        if new.order_status[oid] == OrderStatus.CREATED.value:
            new.inventory[key] = new.inventory.get(key, 0) + 1
        new.order_status[oid] = OrderStatus.NONE.value
        new.audit_flags.append(f"release:{oid}")
        receipts.append({"type": "release_receipt", "order_id": oid})
        return new, receipts

    if tool == "create_order":
        oid = call["order_id"]
        if oid not in new.reservation:
            raise ValueError("NO_RESERVATION")
        new.order_status[oid] = OrderStatus.CREATED.value
        receipts.append({"type": "order_receipt", "order_id": oid, "status": OrderStatus.CREATED.value})
        return new, receipts

    if tool == "charge":
        oid = call["order_id"]
        uid = call["user_id"]
        amount = int(call["amount_cents"])
        if new.order_status.get(oid) != OrderStatus.CREATED.value:
            raise ValueError("ORDER_NOT_FOUND" if oid not in new.order_status else "NOT_CREATED")
        if config.permission == ShiftFamily.PERMISSION_V1 and "payment" not in new.permission_scope:
            raise ValueError("PERMISSION_DENIED:payment")
        if new.balance.get(uid, 0) < amount:
            raise ValueError("INSUFFICIENT_FUNDS")
        if config.delay == ShiftFamily.DELAY_V1:
            new.delayed_effect_queue.append({"kind": "charge", "user_id": uid, "amount": amount,
                                             "order_id": oid, "due_turn": new.turn + 2})
            receipts.append({"type": "charge_receipt", "order_id": oid, "status": "PENDING"})
        else:
            new.balance[uid] -= amount
            new.order_status[oid] = OrderStatus.PAID.value
            receipts.append({"type": "charge_receipt", "order_id": oid, "status": OrderStatus.PAID.value})
        new.audit_flags.append(f"charge:{oid}:{amount}")
        return new, receipts

    if tool == "refund":
        oid = call["order_id"]
        uid = call["user_id"]
        if oid not in new.order_status:
            raise ValueError("ORDER_NOT_FOUND")
        if config.dynamics == ShiftFamily.DYNAMICS_V1:
            # dynamics_v1: refund allowed only for CANCELLED orders (was: PAID/CANCELLED)
            allowed = {OrderStatus.CANCELLED.value}
        else:
            allowed = {OrderStatus.PAID.value, OrderStatus.CANCELLED.value}
        if new.order_status[oid] not in allowed:
            raise ValueError("REFUND_NOT_ALLOWED")
        key = new.reservation.get(oid)
        amount = _price_of(new, key) if key else 0
        if config.delay == ShiftFamily.DELAY_V1:
            new.delayed_effect_queue.append({"kind": "refund", "user_id": uid, "amount": amount,
                                             "due_turn": new.turn + 2})
            receipts.append({"type": "refund_receipt", "order_id": oid, "status": "PENDING"})
        else:
            new.balance[uid] = new.balance.get(uid, 0) + amount
            receipts.append({"type": "refund_receipt", "order_id": oid, "status": "DONE"})
        new.audit_flags.append(f"refund:{oid}")
        return new, receipts

    if tool == "ship":
        oid = call["order_id"]
        uid = call["user_id"]
        addr = call["address"]
        if new.order_status.get(oid) != OrderStatus.PAID.value:
            raise ValueError("ORDER_NOT_FOUND" if oid not in new.order_status else "NOT_PAID")
        if config.permission == ShiftFamily.PERMISSION_V1 and "shipping" not in new.permission_scope:
            raise ValueError("PERMISSION_DENIED:shipping")
        new.order_status[oid] = OrderStatus.SHIPPED.value
        new.address[uid] = addr
        new.audit_flags.append(f"ship:{oid}:{addr}")  # irreversible
        receipts.append({"type": "dispatch_receipt", "order_id": oid, "address": addr,
                         "irreversible": True})
        return new, receipts

    if tool == "cancel_order":
        oid = call["order_id"]
        if oid not in new.order_status:
            raise ValueError("ORDER_NOT_FOUND")
        if config.dynamics == ShiftFamily.DYNAMICS_V1:
            # dynamics_v1: cancel allowed only for CREATED (was: CREATED/PAID not shipped)
            allowed = {OrderStatus.CREATED.value}
        else:
            allowed = {OrderStatus.CREATED.value, OrderStatus.PAID.value}
        if new.order_status[oid] not in allowed:
            raise ValueError("CANCEL_NOT_ALLOWED")
        new.order_status[oid] = OrderStatus.CANCELLED.value
        new.audit_flags.append(f"cancel:{oid}")
        receipts.append({"type": "cancel_receipt", "order_id": oid, "status": OrderStatus.CANCELLED.value})
        return new, receipts

    if tool == "get_receipt":
        oid = call["order_id"]
        if oid not in new.order_status:
            raise ValueError("ORDER_NOT_FOUND")
        status = new.order_status[oid]
        proj = _projection(new, oid)
        if config.poison == ShiftFamily.POISON_V1:
            if _poisoned(config, oid):
                status = _flip(status)
                proj = _flip_projection(proj)
                new.audit_flags.append(f"poisoned_receipt:{oid}")
        receipts.append({"type": "status_receipt", "order_id": oid, "status": status,
                         "projection": proj})
        return new, receipts

    if tool == "complete_task":
        new.audit_flags.append("task_completed")
        receipts.append({"type": "ack", "status": "COMPLETE"})
        return new, receipts

    raise ValueError(f"UNKNOWN_TOOL:{tool}")


def _bucket(stock: int) -> str:
    if stock <= 0:
        return "OUT"
    if stock <= 3:
        return "LOW"
    return "HIGH"


def _price_of(state: WorldState, key: str | None) -> int:
    """Deterministic price table for canonical item keys."""
    table = {"item:a": 1000, "item:b": 2500, "item:c": 500, "item:d": 7500,
             "sku:a": 1000, "sku:b": 2500, "sku:c": 500, "sku:d": 7500}
    return table.get(key or "", 1000)


def _projection(state: WorldState, oid: str) -> dict[str, Any]:
    return {
        "order_id": oid,
        "status": state.order_status.get(oid, OrderStatus.NONE.value),
        "reserved_item": state.reservation.get(oid),
    }


def _flip(status: str) -> str:
    return {"PAID": "CREATED", "CREATED": "PAID", "CANCELLED": "CREATED"}.get(status, status)


def _flip_projection(proj: dict[str, Any]) -> dict[str, Any]:
    proj = dict(proj)
    proj["status"] = _flip(proj["status"])
    return proj


def _poisoned(config: ShiftConfig, oid: str) -> bool:
    h = hashlib.sha256(f"{config.poison_rng_seed}:{oid}".encode()).hexdigest()
    return int(h, 16) % 100 < int(config.poison_prob * 100)

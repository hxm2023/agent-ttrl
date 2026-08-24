"""CTS-v2: latent skill families for learnability experiments (v2 §13.1).

The v1 CTS had one workflow (ship an order). v2 introduces multiple LATENT
SKILL FAMILIES — repeatable workflows instantiated over unseen entities —
so that cross-task inductive transfer has a real signal source:

  F1 policy-rule transfer: apply the refund/exchange/cancel rule to unseen
     user/item/order instances (the rule is latent, the entities differ).
  F2 tool-composition transfer: search -> verify -> mutate -> confirm
     workflow over unseen item ids.
  F3 error-recovery transfer: after a permission failure, apply the common
     recovery path (correct arguments / authorized alternative).

Streams are LEAVE-ONE-TEMPLATE-OUT: the update policy sees adaptation
templates, first attempts on sealed templates are the prequential target.
The hidden oracle (goal satisfaction on final world state) is used ONLY for
estimator-fidelity diagnostics (L0); L1+ uses accessible evidence.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum


class OrderStatus(Enum):
    CREATED = "created"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


@dataclass
class Item:
    sku: str
    price: int
    stock: int


@dataclass
class User:
    user_id: str
    address: str
    balance: int = 0


@dataclass
class Order:
    order_id: str
    user_id: str
    sku: str
    status: OrderStatus
    address: str = ""


@dataclass
class WorldV2:
    items: dict[str, Item]
    users: dict[str, User]
    orders: dict[str, Order]
    permissions: set[str] = field(default_factory=lambda: {"payment"})

    def snapshot_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for o in sorted(self.orders):
            h.update(f"{o}:{self.orders[o].status.value}".encode())
        for u in sorted(self.users):
            h.update(f"{u}:{self.users[u].balance}".encode())
        return h.hexdigest()[:16]


@dataclass
class ToolResult:
    ok: bool
    error: str = ""
    data: dict = field(default_factory=dict)


# ------------------------------------------------------------------ tools
def _lookup_order(w: WorldV2, order_id: str) -> ToolResult:
    o = w.orders.get(order_id)
    if o is None:
        return ToolResult(False, "order not found")
    return ToolResult(True, data={"order_id": o.order_id, "user_id": o.user_id,
                                  "sku": o.sku, "status": o.status.value})


def _lookup_user(w: WorldV2, user_id: str) -> ToolResult:
    u = w.users.get(user_id)
    if u is None:
        return ToolResult(False, "user not found")
    return ToolResult(True, data={"user_id": u.user_id, "address": u.address,
                                  "balance": u.balance})


def _refund_order(w: WorldV2, order_id: str, user_id: str, reason: str = "") -> ToolResult:
    o = w.orders.get(order_id)
    if o is None:
        return ToolResult(False, "order not found")
    if o.user_id != user_id:
        return ToolResult(False, "user mismatch")
    if o.status not in (OrderStatus.PAID, OrderStatus.DELIVERED):
        return ToolResult(False, f"cannot refund order in state {o.status.value}")
    if "payment" not in w.permissions:
        return ToolResult(False, "permission denied: payment")
    item = w.items.get(o.sku)
    w.users[user_id].balance += item.price if item else 0
    o.status = OrderStatus.REFUNDED
    return ToolResult(True, data={"refunded": item.price if item else 0})


def _cancel_order(w: WorldV2, order_id: str, reason: str = "") -> ToolResult:
    o = w.orders.get(order_id)
    if o is None:
        return ToolResult(False, "order not found")
    if o.status not in (OrderStatus.CREATED, OrderStatus.PAID):
        return ToolResult(False, f"cannot cancel order in state {o.status.value}")
    o.status = OrderStatus.CANCELLED
    return ToolResult(True)


def _exchange_item(w: WorldV2, order_id: str, old_item_id: str, new_item_id: str) -> ToolResult:
    o = w.orders.get(order_id)
    if o is None:
        return ToolResult(False, "order not found")
    if o.sku != old_item_id:
        return ToolResult(False, "sku mismatch")
    item = w.items.get(new_item_id)
    if item is None:
        return ToolResult(False, "item not found")
    if item.stock <= 0:
        return ToolResult(False, "out of stock")
    if o.status not in (OrderStatus.DELIVERED, OrderStatus.PAID):
        return ToolResult(False, f"cannot exchange order in state {o.status.value}")
    item.stock -= 1
    o.sku = new_item_id
    return ToolResult(True)


def _ship_order(w: WorldV2, order_id: str, address: str) -> ToolResult:
    o = w.orders.get(order_id)
    if o is None:
        return ToolResult(False, "order not found")
    if "shipping" not in w.permissions:
        return ToolResult(False, "permission denied: shipping")
    if o.status != OrderStatus.PAID:
        return ToolResult(False, f"cannot ship order in state {o.status.value}")
    o.address = address
    o.status = OrderStatus.SHIPPED
    return ToolResult(True)


TOOLS = {
    "lookup_order": _lookup_order,
    "lookup_user": _lookup_user,
    "refund_order": _refund_order,
    "cancel_order": _cancel_order,
    "exchange_item": _exchange_item,
    "ship_order": _ship_order,
}


# ------------------------------------------------------------------ task templates
@dataclass
class TaskTemplate:
    family: str
    name: str
    goal: str
    tools: list[str]
    success: callable  # (world) -> bool
    init: callable  # (rng, world) -> None  (materializes order/user/item)
    perms: set = None  # permissions available at start (None = per-init)

    def __post_init__(self):
        if self.perms is None:
            self.perms = {"payment"}

    def instantiate(self, rng: random.Random) -> "TaskV2":
        w = WorldV2(items={}, users={}, orders={}, permissions=set(self.perms))
        self.init(rng, w)
        return TaskV2(self, w)


def _mk_item(rng, sku):
    return Item(sku=sku, price=rng.choice([50, 100, 200]), stock=rng.randint(2, 9))


def _init_f1_refund(rng, w):
    sku = f"sku-{rng.randint(1, 99)}"
    user = f"user-{rng.randint(1, 99)}"
    oid = f"order-{rng.randint(1, 99)}"
    w.items[sku] = _mk_item(rng, sku)
    w.users[user] = User(user, f"addr-{rng.randint(1, 99)}")
    w.orders[oid] = Order(oid, user, sku, OrderStatus.PAID)
    w._goal = {"order": oid, "user": user}


def _ok_f1_refund(w):
    o = list(w.orders.values())[0]
    return o.status == OrderStatus.REFUNDED


def _init_f1_cancel(rng, w):
    sku = f"sku-{rng.randint(1, 99)}"
    user = f"user-{rng.randint(1, 99)}"
    oid = f"order-{rng.randint(1, 99)}"
    w.items[sku] = _mk_item(rng, sku)
    w.users[user] = User(user, f"addr-{rng.randint(1, 99)}")
    w.orders[oid] = Order(oid, user, sku, OrderStatus.PAID)
    w._goal = {"order": oid}


def _ok_f1_cancel(w):
    o = list(w.orders.values())[0]
    return o.status == OrderStatus.CANCELLED


def _init_f1_exchange(rng, w):
    sku = f"sku-{rng.randint(1, 99)}"
    new_sku = f"sku-{rng.randint(100, 199)}"
    user = f"user-{rng.randint(1, 99)}"
    oid = f"order-{rng.randint(1, 99)}"
    w.items[sku] = _mk_item(rng, sku)
    w.items[new_sku] = _mk_item(rng, new_sku)
    w.users[user] = User(user, f"addr-{rng.randint(1, 99)}")
    w.orders[oid] = Order(oid, user, sku, OrderStatus.DELIVERED)
    w._goal = {"order": oid, "old": sku, "new": new_sku}


def _ok_f1_exchange(w):
    o = list(w.orders.values())[0]
    return o.status == OrderStatus.DELIVERED and o.sku != w._goal["old"]


def _init_f3_recover(rng, w):
    sku = f"sku-{rng.randint(1, 99)}"
    user = f"user-{rng.randint(1, 99)}"
    oid = f"order-{rng.randint(1, 99)}"
    w.items[sku] = _mk_item(rng, sku)
    w.users[user] = User(user, f"addr-{rng.randint(1, 99)}")
    w.orders[oid] = Order(oid, user, sku, OrderStatus.PAID)
    # shipping permission MISSING at start: agent must discover/use the
    # recovery path (e.g., request permission then ship)
    w.permissions = set()
    w._goal = {"order": oid, "user": user}


def _ok_f3_recover(w):
    o = list(w.orders.values())[0]
    return o.status == OrderStatus.SHIPPED


TEMPLATES = {
    "F1_refund": TaskTemplate("F1", "refund", "Refund order {order} for user {user}",
                              ["lookup_order", "lookup_user", "refund_order"],
                              _ok_f1_refund, _init_f1_refund),
    "F1_cancel": TaskTemplate("F1", "cancel", "Cancel order {order}",
                              ["lookup_order", "cancel_order"],
                              _ok_f1_cancel, _init_f1_cancel),
    "F1_exchange": TaskTemplate("F1", "exchange", "Exchange item {old} for {new} on order {order}",
                                ["lookup_order", "exchange_item"],
                                _ok_f1_exchange, _init_f1_exchange),
    "F3_recover": TaskTemplate("F3", "recover", "Ship order {order} to {user} (may need permission)",
                               ["lookup_order", "lookup_user", "ship_order"],
                               _ok_f3_recover, _init_f3_recover),
}


@dataclass
class TaskV2:
    template: TaskTemplate
    world: WorldV2

    @property
    def goal(self) -> str:
        g = getattr(self.world, "_goal", {})
        return self.template.goal.format(**g) if g else self.template.goal

    @property
    def tool_descriptions(self) -> str:
        return "\n".join(f"- {name}({', '.join(f'{p}' for p in ['order_id', 'user_id'])}): {doc}"
                         for name, doc in [
                             ("lookup_order", "get order status"),
                             ("lookup_user", "get user balance"),
                             ("refund_order", "refund a paid/delivered order"),
                             ("cancel_order", "cancel a created/paid order"),
                             ("exchange_item", "swap item on a delivered order"),
                             ("ship_order", "ship a paid order (needs shipping permission)"),
                         ])

    def exec_call(self, name: str, kwargs: dict) -> ToolResult:
        fn = TOOLS.get(name)
        if fn is None:
            return ToolResult(False, f"unknown tool {name}")
        try:
            return fn(self.world, **kwargs)
        except TypeError as e:
            return ToolResult(False, f"bad arguments: {e}")

    @property
    def hidden_success(self) -> bool:
        return bool(self.template.success(self.world))

"""CTS-F01..F12 golden fixtures + three-action hand fixture (design doc §9.1.2, §7.5).

Each fixture fixes: initial state, setup steps (to the decision state), goal,
G candidate actions at the decision state, per-action continuation scripts,
expected credit signs (or fail-closed reason code).
U_{i,r} is produced by executing action i + continuation with seed r, then
evidence_utility on the accessible evidence bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from agent_ttrl.environments.cts_oracle import GoalSpec
from agent_ttrl.environments.cts_world import ShiftConfig, ShiftFamily, WorldState

Action = dict
Continuation = Callable[[WorldState, GoalSpec], list[tuple[str, Action]]]


def _base_inventory() -> dict[str, int]:
    return {"sku:a": 5, "sku:b": 3, "sku:c": 1, "sku:d": 0,
            "item:a": 5, "item:b": 3, "item:c": 1, "item:d": 0}


def base_user_state() -> WorldState:
    return WorldState(
        inventory=_base_inventory(),
        balance={"u1": 100_000},
        address={"u1": "addr-1"},
        permission_scope=["payment", "shipping"],
    )


def _setup_reserve_charge() -> list[tuple[str, Action]]:
    """reserve sku:a, create order o1, charge o1 -> decision state."""
    return [
        ("reserve_item", {"item_key": "sku:a", "order_id": "o1"}),
        ("create_order", {"order_id": "o1"}),
        ("charge", {"order_id": "o1", "user_id": "u1", "amount_cents": 1000}),
    ]


def _ship_and_complete(state: WorldState, goal: GoalSpec) -> list[tuple[str, Action]]:
    return [("ship", {"order_id": "o1", "user_id": goal.user_id, "address": "addr-1"}),
            ("complete_task", {})]


@dataclass
class CTSFixture:
    fid: str
    title: str
    config: ShiftConfig
    initial: WorldState
    goal: GoalSpec
    group_actions: list[Action]                     # G alternative actions at decision state
    continuations: list[Continuation]               # one per action; None -> no continuation
    setup_steps: list[tuple[str, Action]] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    expected_signs: list[int] | None = None         # +1/-1/0 per action; None if fail-closed
    fail_closed_reason: str | None = None           # reason code if fail-closed
    hidden_outcome: bool | None = None              # oracle success for the "canonical" branch
    failure_reason: str | None = None               # why the naive/negative action fails
    assert_conflict: bool = False                   # F08: assert E_hard conflict recorded


def _continue_order(state: WorldState, goal: GoalSpec) -> list[tuple[str, Action]]:
    return [
        ("create_order", {"order_id": "o1"}),
        ("charge", {"order_id": "o1", "user_id": goal.user_id, "amount_cents": 1000}),
        ("ship", {"order_id": "o1", "user_id": goal.user_id, "address": "addr-1"}),
        ("complete_task", {}),
    ]


def _continue_noop(state: WorldState, goal: GoalSpec) -> list[tuple[str, Action]]:
    return [("complete_task", {})]


def _wait_then_ship(state: WorldState, goal: GoalSpec) -> list[tuple[str, Action]]:
    return [("get_receipt", {"order_id": "o1"}),
            ("ship", {"order_id": "o1", "user_id": goal.user_id, "address": "addr-1"}),
            ("complete_task", {})]


def _cancel_then_refund(state: WorldState, goal: GoalSpec) -> list[tuple[str, Action]]:
    return [("cancel_order", {"order_id": "o1"}),
            ("refund", {"order_id": "o1", "user_id": goal.user_id}),
            ("complete_task", {})]


def _ship_then_cancel(state: WorldState, goal: GoalSpec) -> list[tuple[str, Action]]:
    return [("cancel_order", {"order_id": "o1"}), ("complete_task", {})]


def _build_goal(want_item: str, **kw) -> GoalSpec:
    return GoalSpec(user_id="u1", want_item=want_item, **kw)


FIXTURES: list[CTSFixture] = [
    CTSFixture(
        fid="CTS-F01", title="correct reserve vs wrong sku",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a"),
        group_actions=[
            {"tool": "reserve_item", "call": {"item_key": "sku:a", "order_id": "o1"}},
            {"tool": "reserve_item", "call": {"item_key": "sku:b", "order_id": "o1"}},
        ],
        continuations=[_continue_order, _continue_order],
        expected_signs=[+1, -1], hidden_outcome=True,
        failure_reason="wrong item reserved -> goal_item_not_shipped",
    ),
    CTSFixture(
        fid="CTS-F02", title="correct address vs wrong address (irreversible)",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a", want_address="addr-1"),
        group_actions=[
            {"tool": "ship", "call": {"order_id": "o1", "user_id": "u1", "address": "addr-1"}},
            {"tool": "ship", "call": {"order_id": "o1", "user_id": "u1", "address": "addr-999"}},
        ],
        continuations=[_continue_noop, _continue_noop],
        setup_steps=_setup_reserve_charge(),
        expected_signs=[+1, -1], hidden_outcome=True,
        failure_reason="wrong irreversible shipment -> wrong_address; dispatch receipt irreversible",
    ),
    CTSFixture(
        fid="CTS-F03", title="schema-valid no-op vs effective call",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a"),
        group_actions=[
            {"tool": "lookup_item", "call": {"item_key": "sku:a"}},
            {"tool": "reserve_item", "call": {"item_key": "sku:a", "order_id": "o1"}},
        ],
        continuations=[_continue_order, _continue_order],
        expected_signs=[-1, +1], hidden_outcome=False,
        failure_reason="validity alone insufficient; state evidence required",
    ),
    CTSFixture(
        fid="CTS-F04", title="repeated reserve (duplicate side effect)",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a", no_collateral=["duplicate_reserve:o1"]),
        group_actions=[
            {"tool": "charge", "call": {"order_id": "o1", "user_id": "u1", "amount_cents": 1000}},
            {"tool": "reserve_item", "call": {"item_key": "sku:a", "order_id": "o1"}},  # duplicate
        ],
        continuations=[_ship_and_complete, _continue_order],
        setup_steps=[("reserve_item", {"item_key": "sku:a", "order_id": "o1"}),
                     ("create_order", {"order_id": "o1"})],
        expected_signs=[+1, -1], hidden_outcome=True,
        failure_reason="duplicate reserve -> collateral:duplicate_reserve:o1 (stock consumed twice)",
    ),
    CTSFixture(
        fid="CTS-F05", title="delayed charge: receipt != final success",
        config=ShiftConfig(delay=ShiftFamily.DELAY_V1), initial=base_user_state(),
        goal=_build_goal("sku:a"),
        group_actions=[
            {"tool": "complete_task", "call": {}},
            {"tool": "get_receipt", "call": {"order_id": "o1"}},
        ],
        continuations=[None, _wait_then_ship],
        setup_steps=_setup_reserve_charge(),
        expected_signs=[-1, +1], hidden_outcome=False,
        failure_reason="task completed before delayed charge applied -> NOT_PAID at ship time",
    ),
    CTSFixture(
        fid="CTS-F06", title="partial verifier misses collateral (money)",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a", want_cancelled=True, min_balance=100_000),
        group_actions=[
            {"tool": "complete_task", "call": {}},                # cancel w/o refund
            {"tool": "refund", "call": {"order_id": "o1", "user_id": "u1"}},
        ],
        continuations=[None, _continue_noop],
        setup_steps=_setup_reserve_charge() + [("cancel_order", {"order_id": "o1"})],
        expected_signs=[0, 0], hidden_outcome=False,
        failure_reason="verifier blind to balance; proxy/hidden disagreement recorded",
    ),
    CTSFixture(
        fid="CTS-F07", title="syntax_v1 parameter migration",
        config=ShiftConfig(syntax=ShiftFamily.SYNTAX_V1), initial=base_user_state(),
        goal=_build_goal("item:a"),
        group_actions=[
            {"tool": "reserve_item", "call": {"item_key": "sku:a", "order_id": "o1"}},
            {"tool": "reserve_item", "call": {"item_key": "item:a", "order_id": "o1"}},
        ],
        continuations=[_continue_order, _continue_order],
        expected_signs=[-1, +1], hidden_outcome=False,
        failure_reason="old syntax fails (ITEM_NOT_FOUND); new syntax passes",
    ),
    CTSFixture(
        fid="CTS-F08", title="poisoned success receipt vs DB invariant",
        config=ShiftConfig(poison=ShiftFamily.POISON_V1, poison_prob=1.0, poison_rng_seed=7),
        initial=base_user_state(), goal=_build_goal("sku:a"),
        group_actions=[
            {"tool": "complete_task", "call": {}},
            {"tool": "charge", "call": {"order_id": "o1", "user_id": "u1", "amount_cents": 1000}},
        ],
        continuations=[None, _ship_and_complete],
        setup_steps=[("reserve_item", {"item_key": "sku:a", "order_id": "o1"}),
                     ("create_order", {"order_id": "o1"}),
                     ("get_receipt", {"order_id": "o1"})],   # poisoned: says PAID, DB CREATED
        expected_signs=[-1, +1], hidden_outcome=False,
        failure_reason="receipt says PAID but projection CREATED -> conflict; DB state decides",
        assert_conflict=True,
    ),
    CTSFixture(
        fid="CTS-F09", title="cancel-after-ship is irreversible",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a"),
        group_actions=[
            {"tool": "ship", "call": {"order_id": "o1", "user_id": "u1", "address": "addr-1"}},
            {"tool": "cancel_order", "call": {"order_id": "o1"}},
        ],
        continuations=[_ship_then_cancel, None],
        setup_steps=_setup_reserve_charge(),
        expected_signs=[+1, -1], hidden_outcome=True,
        failure_reason="cancel after ship raises CANCEL_NOT_ALLOWED; restore sandbox-only",
    ),
    CTSFixture(
        fid="CTS-F10", title="all-success proposal group",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a"),
        group_actions=[
            {"tool": "complete_task", "call": {}},
            {"tool": "complete_task", "call": {}},
            {"tool": "complete_task", "call": {}},
        ],
        continuations=[None, None, None],
        fail_closed_reason="DEGENERATE_GROUP/ALL_SAME_OUTCOME", hidden_outcome=True,
    ),
    CTSFixture(
        fid="CTS-F11", title="all-fail proposal group",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a"),
        group_actions=[
            {"tool": "lookup_item", "call": {"item_key": "sku:zz"}},
            {"tool": "lookup_item", "call": {"item_key": "sku:yy"}},
            {"tool": "lookup_item", "call": {"item_key": "sku:xx"}},
        ],
        continuations=[None, None, None],
        fail_closed_reason="DEGENERATE_GROUP/ALL_SAME_OUTCOME", hidden_outcome=False,
    ),
    CTSFixture(
        fid="CTS-F12", title="branch writes parent world",
        config=ShiftConfig(), initial=base_user_state(),
        goal=_build_goal("sku:a"),
        group_actions=[
            {"tool": "reserve_item", "call": {"item_key": "sku:a", "order_id": "o1"}},
        ],
        continuations=[_continue_order],
        fail_closed_reason="INVALID/BRANCH_ISOLATION_BREACH", hidden_outcome=True,
    ),
]


# Three-action hand fixture (design doc §7.5): frozen U means [0.9, 0.6, 0.0].
# Pre-registered t-intervals [0.30,0.50],[-0.05,0.25],[-0.65,-0.35] with eta=0.05
# must yield alpha = [1, 0, 1].
HAND_U = np.array([
    [0.90, 0.90, 0.90, 0.90],
    [0.60, 0.60, 0.60, 0.60],
    [0.00, 0.00, 0.00, 0.00],
], dtype=float)
HAND_INTERVALS = [(0.30, 0.50), (-0.05, 0.25), (-0.65, -0.35)]
HAND_EXPECTED_ALPHA = [1, 0, 1]

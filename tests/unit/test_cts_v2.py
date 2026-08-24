"""CTS-v2 environment + replay buffer unit tests (CPU, fast)."""
import random

from agent_ttrl.environments.cts_v2 import TEMPLATES, WorldV2
from agent_ttrl.optimization.replay_buffer import EvidenceRow, ReplayBuffer


def _tpl(name):
    return TEMPLATES[name]


def test_f1_refund_success_path():
    rng = random.Random(0)
    task = _tpl("F1_refund").instantiate(rng)
    g = task.world._goal
    assert task.exec_call("lookup_order", {"order_id": g["order"]}).ok
    assert task.exec_call("lookup_user", {"user_id": g["user"]}).ok
    r = task.exec_call("refund_order", {"order_id": g["order"], "user_id": g["user"]})
    assert r.ok and task.hidden_success


def test_f1_refund_wrong_user_fails():
    rng = random.Random(1)
    task = _tpl("F1_refund").instantiate(rng)
    g = task.world._goal
    r = task.exec_call("refund_order", {"order_id": g["order"], "user_id": "user-999"})
    assert not r.ok and not task.hidden_success


def test_f1_cancel_and_exchange():
    rng = random.Random(2)
    t = _tpl("F1_cancel").instantiate(rng)
    assert t.exec_call("cancel_order", {"order_id": t.world._goal["order"]}).ok
    assert t.hidden_success
    rng = random.Random(3)
    t = _tpl("F1_exchange").instantiate(rng)
    g = t.world._goal
    assert t.exec_call("exchange_item", {"order_id": g["order"], "old_item_id": g["old"],
                                         "new_item_id": g["new"]}).ok
    assert t.hidden_success


def test_f3_recover_permission_gate():
    rng = random.Random(4)
    t = _tpl("F3_recover").instantiate(rng)
    g = t.world._goal
    # shipping blocked without permission
    r = t.exec_call("ship_order", {"order_id": g["order"], "address": "addr-1"})
    assert not r.ok and r.error.startswith("permission denied")
    # recovery: grant permission (simulated admin action) then ship
    t.world.permissions.add("shipping")
    assert t.exec_call("ship_order", {"order_id": g["order"], "address": "addr-1"}).ok
    assert t.hidden_success


def test_templates_are_leave_one_template_out_ready():
    assert set(TEMPLATES) == {"F1_refund", "F1_cancel", "F1_exchange", "F3_recover"}
    assert {t.family for t in TEMPLATES.values()} == {"F1", "F3"}


def test_replay_buffer_dedup_and_capacity():
    rb = ReplayBuffer(capacity=16, anchor_fraction=0.2)
    for i in range(30):
        rb.add(EvidenceRow(f"t{i}", "F1_refund", list(range(5)), list(range(5, 9)),
                           advantage=1.0 if i % 2 else -1.0, policy_version=0))
    # same row twice -> dedup
    rb.add(EvidenceRow("t1", "F1_refund", list(range(5)), list(range(5, 9)),
                       advantage=1.0, policy_version=0))
    assert len(rb.rows) <= 16
    st = rb.stats()
    assert st["rows"] == len(rb.rows)
    assert rb._seen


def test_replay_buffer_anchors_never_evicted():
    rb = ReplayBuffer(capacity=8, anchor_fraction=0.25)
    rb.set_anchor(EvidenceRow("anchor", "F1_refund", list(range(3)), list(range(3, 6)),
                              advantage=1.0, policy_version=0))
    for i in range(50):
        rb.add(EvidenceRow(f"t{i}", "F1_refund", [i], [i + 1], advantage=0.5, policy_version=0))
    assert any(r.task_id == "anchor" for r in rb.rows)
    batch = rb.sample_update_batch(8)
    assert any(r.task_id == "anchor" for r in batch)

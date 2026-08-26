"""Stage 3 tasks: two tool-use environments for the real closed loop.

Task A `cts_order` — the CTS evidence utility world (agent-ttrl's existing
environment, same reward as r002): the agent must call tools in the right
order (reserve -> create_order -> charge -> ship -> complete); reward =
evidence utility u in [0, 1].

Task B `tau2_retail` — a TAU2 retail subset (server-backed): the agent
receives a retail task instruction, emits function calls, the tau2 server
executes them and evaluates the call history against evaluation_criteria;
reward = match score in [0, 1].

Both tasks expose: prompt(text, prompt_id), reward_fn(text) -> (u, info),
evaluate(completions) -> {success_rate, mean_u, mean_len, invalid_rate}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- CTS order

CTS_TASK_PROMPT = """You are an order assistant. Available tools (JSON list of {"tool": ..., "call": {...}}):
- reserve_item {item_key, order_id}
- create_order {order_id}
- charge {order_id, user_id, amount_cents}
- ship {order_id, user_id, address}
- complete_task {}
Task: user u1 wants item sku:a shipped to address addr-1. Call the tools in order.
Return ONLY the JSON list of tool calls."""


def parse_tool_calls(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        calls = json.loads(m.group(0))
        return [c for c in calls if isinstance(c, dict) and "tool" in c]
    except Exception:
        return []


def cts_reward(text: str) -> tuple[float, dict]:
    from agent_ttrl.environments.cts_evidence import AccessibleEvidence, evidence_utility
    from agent_ttrl.environments.cts_oracle import GoalSpec, hidden_score
    from agent_ttrl.environments.cts_world import ShiftConfig, WorldState, advance_turn, transition

    calls = parse_tool_calls(text)
    if not calls:
        return 0.0, {"u": 0.0, "hidden_success": False, "state_sha": None, "errors": ["NO_PARSE"]}
    state = WorldState(
        inventory={"sku:a": 5, "sku:b": 3},
        balance={"u1": 100_000},
        address={"u1": "addr-1"},
        permission_scope=["payment", "shipping"],
    )
    errors = []
    config = ShiftConfig()
    for call in calls:
        tool = call.get("tool", "")
        c = call.get("call", {})
        try:
            state, _ = transition(state, tool, c, config)
            state = advance_turn(state)
        except ValueError as e:
            errors.append(str(e))
    goal = GoalSpec(user_id="u1", want_item="sku:a", want_address="addr-1")
    u = evidence_utility(AccessibleEvidence(state, goal).collect(), goal)
    info = {"u": u, "hidden_success": hidden_score(state, goal).success, "errors": errors[:3]}
    return float(u), info


class CtsTask:
    task_id = "cts_order"

    def prompts(self, n: int, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            out.append({"text": CTS_TASK_PROMPT, "prompt_id": f"cts-order-{i:04d}"})
        return out

    def reward(self, text: str) -> tuple[float, dict]:
        return cts_reward(text)

    def evaluate(self, completions: list[str]) -> dict:
        utils = [self.reward(t)[0] for t in completions]
        calls = [len(parse_tool_calls(t)) for t in completions]
        return {
            "n": len(completions),
            "success_rate": float(np.mean([u >= 0.99 for u in utils])),
            "mean_u": float(np.mean(utils)),
            "mean_len": float(np.mean(calls)),
            "invalid_rate": float(np.mean([c == 0 for c in calls])),
        }


# ---------------------------------------------------------------- TAU2 retail

class Tau2Task:
    """TAU2 retail subset via the tau2 server (POST /init /exec /eval).

    The server is started once by run_matrix.sh; this task only talks HTTP.
    """

    task_id = "tau2_retail"
    server_url = "http://127.0.0.1:8800"
    _task_ids: list[str] | None = None

    def __init__(self, n: int = 32, seed: int = 0, split: str = "base") -> None:
        self.n = n
        self.seed = seed
        self.split = split

    # frozen ids of the tau2 base retail split (read 2026-08-24 via the
    # tau2 package; the server has no /tasks endpoint)
    FROZEN_IDS = [str(i) for i in range(40)]

    def _all_task_ids(self) -> list[str]:
        if Tau2Task._task_ids is None:
            try:
                import urllib.request

                req = urllib.request.Request(f"{self.server_url}/tasks")
                with urllib.request.urlopen(req, timeout=30) as r:
                    Tau2Task._task_ids = json.loads(r.read())["task_ids"]
            except Exception:
                Tau2Task._task_ids = list(Tau2Task.FROZEN_IDS)
        return Tau2Task._task_ids

    def prompts(self, n: int, seed: int) -> list[dict]:
        import urllib.request

        ids = self._all_task_ids()
        rng = np.random.default_rng(seed)
        picked = [ids[i % len(ids)] for i in rng.choice(len(ids), size=n, replace=False)]
        out = []
        for task_id in picked:
            req = urllib.request.Request(
                f"{self.server_url}/init",
                data=json.dumps({"task_id": task_id}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
            if not resp.get("ok"):
                continue
            out.append({"text": resp["instruction"], "prompt_id": f"tau2-{task_id}"})
        return out

    def reward(self, text: str) -> tuple[float, dict]:
        import urllib.request

        def post(path: str, payload: dict) -> dict:
            req = urllib.request.Request(
                f"{self.server_url}{path}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())

        post("/reset", {})  # isolate this generation's call history
        ev = post("/exec", {"code": text})
        if not ev.get("ok"):
            return 0.0, {"u": 0.0, "errors": [str(ev.get("error", "exec_fail"))[:80]]}
        evl = post("/eval", {})
        u = float(evl.get("pass_pct", 0.0)) / 100.0
        return u, {"u": u, "errors": []}

    def evaluate(self, completions: list[str]) -> dict:
        utils = [self.reward(t)[0] for t in completions]
        calls = [len(re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", t)) for t in completions]
        return {
            "n": len(completions),
            "success_rate": float(np.mean([u >= 0.99 for u in utils])),
            "mean_u": float(np.mean(utils)),
            "mean_len": float(np.mean(calls)),
            "invalid_rate": float(np.mean([c == 0 for c in calls])),
        }


TASKS = {"cts_order": CtsTask, "tau2_retail": Tau2Task}

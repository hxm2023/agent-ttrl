"""tau2 retail exec server (persistent environment; runs with tau2 deps).

POST /init {"task_id": ...}  -> load retail task + environment
POST /exec  {"code": "..."} -> parse func(args) calls, use_tool, return observation
POST /eval  {}               -> match call history against evaluation_criteria.actions
POST /reset {}               -> reset history
"""
import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")

from flask import Flask, request  # noqa: E402

from tau2.data_model.tasks import Task  # noqa: E402
from tau2.domains.retail.environment import get_environment, get_tasks  # noqa: E402

app = Flask(__name__)
env = None
task = None
call_history: list[dict] = []


@app.post("/init")
def init_world():
    global env, task, call_history
    data = request.get_json(force=True)
    task_id = data["task_id"]
    tasks = get_tasks("base")
    task = next((t for t in tasks if t.id == task_id), None)
    if task is None:
        # try the train split ids too
        tasks = get_tasks(None)
        task = next((t for t in tasks if t.id == task_id), None)
    if task is None:
        return {"ok": False, "error": f"task {task_id} not found"}, 404
    env = get_environment()
    call_history = []
    return {"ok": True, "task_id": task_id,
            "instruction": task.user_scenario.instructions.task_instructions[:300]}


def _parse_calls(code: str) -> list[dict]:
    """Parse func(k=v, ...) lines into {name, arguments}; tolerant of prose."""
    import re
    calls = []
    for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^\n]{0,200})\)", code):
        name, argstr = m.group(1), m.group(2)
        arguments = {}
        for kw in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,\)]+)", argstr):
            key, val = kw.group(1), kw.group(2).strip().strip('"').strip("'")
            arguments[key] = val
        calls.append({"name": name, "arguments": arguments})
    return calls


@app.post("/exec")
def exec_code():
    global env, call_history
    data = request.get_json(force=True)
    code = data.get("code", "")
    calls = _parse_calls(code)
    outputs = []
    for c in calls:
        try:
            out = env.use_tool(c["name"], **c["arguments"])
            outputs.append({"call": c, "output": str(out)[:300], "ok": True})
        except Exception as e:  # noqa: BLE001
            outputs.append({"call": c, "error": str(e)[:200], "ok": False})
    call_history.extend(calls)
    return {"ok": True, "outputs": outputs, "n_ok": sum(1 for o in outputs if o["ok"])}


@app.post("/eval")
def eval_task():
    global task, call_history
    if task is None or task.evaluation_criteria is None:
        return {"ok": True, "success": False, "note": "no criteria", "pass_pct": 0.0}
    actions = task.evaluation_criteria.actions
    matched = 0
    for req in actions:
        for call in call_history:
            if call["name"] != req.name:
                continue
            req_args = req.arguments or {}
            call_args = call.get("arguments") or {}
            if all(call_args.get(k) == v for k, v in req_args.items()):
                matched += 1
                break
    total = len(actions)
    pass_pct = 100.0 * matched / total if total else 100.0
    return {"ok": True, "success": matched == total, "pass_pct": pass_pct,
            "matched": matched, "total": total, "calls": len(call_history)}


@app.post("/reset")
def reset():
    global call_history
    call_history = []
    return {"ok": True}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("TAU2_SERVER_PORT", "8800")), threaded=False)

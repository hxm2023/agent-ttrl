"""AppWorld exec server (persistent environment; runs in appworld-venv).

POST /init {"task_id": ...}  -> load task, initialize world
POST /exec  {"code": "..."} -> execute API calls, return observation
POST /eval  {}               -> hidden evaluation (TestTracker summary)
POST /reset {}               -> close world
"""
import json
import os
import sys

sys.path.insert(0, "/root/autodl-tmp/appworld/src")
os.environ["APPWORLD_ROOT"] = "/root/autodl-tmp/appworld"

from flask import Flask, request  # noqa: E402

from appworld.environment import AppWorld  # noqa: E402
from appworld.task import Task  # noqa: E402

app = Flask(__name__)
world = None
task_id = None


@app.post("/init")
def init_world():
    global world, task_id
    data = request.get_json(force=True)
    task_id = data["task_id"]
    world = AppWorld(task_id=task_id, raise_on_unsafe_execution=False)
    world.initialize()
    return {"ok": True, "task_id": task_id, "instruction": world.task.instruction[:200]}


@app.post("/exec")
def exec_code():
    global world
    data = request.get_json(force=True)
    code = data.get("code", "")
    try:
        result = world.execute(code)
        completed = bool(world.task_completed())
        return {"ok": True, "output": str(result)[:1000], "completed": completed}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:500]}, 400


@app.post("/eval")
def eval_task():
    global world
    try:
        world.task_completed()
        tracker = world.evaluate()
        return {"ok": True, "success": bool(tracker.success),
                "pass_pct": tracker.pass_percentage,
                "passes": len(tracker.passes), "failures": len(tracker.failures)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:500]}, 400


@app.post("/reset")
def reset():
    global world
    if world is not None:
        try:
            world.close()
        except Exception:  # noqa: BLE001
            pass
        world = None
    return {"ok": True}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("APWORLD_SERVER_PORT", "8700")))

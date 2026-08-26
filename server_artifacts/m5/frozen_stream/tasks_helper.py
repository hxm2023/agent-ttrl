
import json, sys, os
sys.path.insert(0, "/root/autodl-tmp/appworld/src")
os.environ["APPWORLD_ROOT"] = "/root/autodl-tmp/appworld"
from appworld.task import Task, load_task_ids
from appworld.api_docs import prepare_api_docs
ids = load_task_ids("dev")
out = []
for tid in ids[:6]:
    try:
        t = Task.load(tid)
        apps = getattr(t, "allowed_apps", None) or []
        docs = {}
        for a in apps[:4]:
            try:
                docs[a] = prepare_api_docs(a, format="function_calling")
            except Exception as e:
                docs[a] = {"error": str(e)[:80]}
        out.append({"id": tid, "instruction": t.instruction, "allowed_apps": apps, "docs": docs})
    except Exception as e:
        out.append({"id": tid, "error": str(e)[:200]})
print(json.dumps(out))

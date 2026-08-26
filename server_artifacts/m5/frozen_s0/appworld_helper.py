
import json, sys
sys.path.insert(0, '/root/autodl-tmp/appworld/src')
sys.path.insert(0, '/root/autodl-tmp/appworld/src/appworld')
from appworld.environment import AppWorld
from appworld.api_docs import prepare_api_docs
from appworld.task import Task, load_task_ids
task_ids = load_task_ids("dev")
task_id_arg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else task_ids[0]
task = Task.load(task_id_arg)
apps = getattr(task, "allowed_apps", None) or ["amazon", "booking", "facebook", "gmail", "linkedin", "quora", "slack", "spotify", "uber", "venmo", "whatsapp", "yelp"]
docs = {}
for app in apps:
    try:
        docs[app] = prepare_api_docs(app, format="function_calling")
    except Exception as e:
        docs[app] = {"error": str(e)[:100]}
print(json.dumps({"instruction": task.instruction, "api_docs": docs, "task_id": task.id}))

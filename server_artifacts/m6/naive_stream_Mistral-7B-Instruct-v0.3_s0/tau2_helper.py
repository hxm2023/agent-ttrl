
import json, sys, os
sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
from tau2.domains.retail.environment import get_environment, get_tasks
env = get_environment()
tools = env.get_tools_description("assistant")
tasks = get_tasks("base")[:20]
out = []
for t in tasks:
    out.append({"id": t.id, "instruction": t.user_scenario.instructions.task_instructions,
                "known": t.user_scenario.instructions.known_info,
                "policy": None})
print(json.dumps({"tools": str(tools)[:4000], "tasks": out}))

"""tau2 v3 pilot: official retail environment + ColocatedTau2Agent.

Runs a single official tau2 task through the official orchestrator and
evaluator (hidden scoring, reporting only). Frozen agent for now.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
sys.path.insert(0, "/root/autodl-tmp/agent-ttrl/src")

from agent_ttrl.runtime.served_policy import ColocatedPolicy
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.runner.simulation import run_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from scripts.tau2_v3_agent import ColocatedTau2Agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--model", default="/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = get_environment()
    tasks = get_tasks("base")
    task = tasks[args.task_idx]
    print(f"task {task.id}: {task.user_scenario.instructions.task_instructions[:100]}",
          flush=True)

    policy = ColocatedPolicy(args.model, lora_rank=8, lora_alpha=16, device=args.device)
    tools = env.get_tools()
    agent = ColocatedTau2Agent(tools=tools, domain_policy=env.policy,
                               policy=policy, stream_seed=args.seed,
                               task_idx=args.task_idx)
    # colocated user simulator (transformers, no external LLM)
    from scripts.tau2_v3_user import ColocatedUserSimulator
    ins = task.user_scenario.instructions
    known = ins.known_info.replace("You are ", "I am ", 1)
    reason = (ins.reason_for_call.replace("You received", "I received")
              .replace(" you ", " I ").replace("you wish", "I'd like", 1))
    scenario = f"{known} {reason}"
    user = ColocatedUserSimulator(instructions=None, tools=None,
                                  policy=policy, stream_seed=args.seed,
                                  task_idx=args.task_idx, scenario_str=scenario)
    orch = Orchestrator(domain="retail", agent=agent, user=user,
                        environment=env, task=task, max_steps=30)
    result = run_simulation(orch)
    r = result.reward_info
    print("reward:", r.reward, "| score:", getattr(r, "score", None),
          "| success:", getattr(r, "success", None), flush=True)
    for m in result.messages[-10:]:
        who = type(m).__name__
        content = str(getattr(m, "content", ""))[:100]
        tcs = getattr(m, "tool_calls", None)
        tcs_s = f" TOOLS={[ (tc.name, str(tc.arguments)[:60]) for tc in tcs ]}" if tcs else ""
        print(f"  {who}: {content}{tcs_s}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

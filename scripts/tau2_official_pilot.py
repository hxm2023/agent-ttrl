"""tau2 official pilot: OFFICIAL LLMAgent + OFFICIAL UserSimulator, both
driven via litellm against our local OpenAI-compatible server (which wraps
ColocatedPolicy). The orchestrator, environment, and hidden evaluator are
the official tau2 components, untouched."""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
sys.path.insert(0, "/root/autodl-tmp/agent-ttrl/src")

from tau2.agent.llm_agent import create_llm_agent
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator import UserSimulator
import tau2.config as tau2_config

BASE = "openai/local-model"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    env = get_environment()
    tasks = get_tasks("base")
    task = tasks[args.task_idx]
    ins = task.user_scenario.instructions
    print(f"task {task.id}: {ins.reason_for_call[:90]}", flush=True)

    # FULLY OFFICIAL pipeline: LLMAgent + UserSimulator, both via litellm
    # against the local OpenAI-compatible server (ColocatedPolicy backend).
    llm_args = {"api_base": args.base_url, "api_key": "sk-local",
                "temperature": 0.3, "seed": args.seed, "max_tokens": 256}
    agent = create_llm_agent(env.get_tools(), env.policy, llm=BASE,
                             llm_args=dict(llm_args))
    try:
        user_tools = env.get_user_tools()
    except Exception:
        user_tools = None
    user = UserSimulator(llm=BASE, instructions=task.user_scenario,
                         tools=user_tools, llm_args=dict(llm_args))
    # Point the evaluator's LLM judge (NL assertions) at the local server.
    # Patch module attributes directly: the evaluator bound these at import.
    import tau2.evaluator.evaluator_nl_assertions as _nl
    _nl.DEFAULT_LLM_NL_ASSERTIONS = BASE
    _nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
        "temperature": 0.0, "api_base": args.base_url, "api_key": "sk-local",
        "max_tokens": 256}
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = BASE
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = _nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS
    orch = Orchestrator(domain="retail", agent=agent, user=user,
                        environment=env, task=task, max_steps=args.max_steps)
    result = run_simulation(orch)
    r = result.reward_info
    print("reward:", r.reward, "| score:", getattr(r, "score", None),
          "| success:", getattr(r, "success", None), flush=True)
    print("breakdown:", getattr(r, "reward_breakdown", None), flush=True)
    print("env_assertions:", getattr(r, "env_assertions", None), flush=True)
    print("nl_assertions:", getattr(r, "nl_assertions", None), flush=True)
    print("termination:", getattr(result, "termination_reason", None),
          "| n_steps:", getattr(result, "n_steps", None), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

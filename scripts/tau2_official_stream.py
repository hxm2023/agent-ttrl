"""tau2 OFFICIAL test-time RL stream (v3 protocol on the official benchmark).

Prequential tasks through the official tau2 orchestrator + hidden evaluator
(reporting only). Arms:
  frozen — no updates (deterministic baseline, CRN across seeds);
  pair   — DPO-style pair updates (verified-success positives vs raw-text
           negatives) with the pre-commit gate, exactly the v3.2 rule.

Credit uses ACCESSIBLE evidence only: a task is a success iff the trajectory
contains a mutation tool call whose id-like arguments all appear in the
tool results of the same conversation (real ids come from tool results).
The hidden official evaluator's reward is recorded for reporting only and
never enters the update/gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
sys.path.insert(0, "/root/autodl-tmp/agent-ttrl/src")

from agent_ttrl.runtime.request_seed import RequestSeed
from agent_ttrl.runtime.served_policy import ColocatedPolicy
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.simulation import run_simulation

from scripts.tau2_v3_agent import ColocatedTau2Agent, _parse_calls
from scripts.tau2_v3_user import ColocatedUserSimulator

MUTATION_TOOLS = {
    "exchange_delivered_order_items", "return_delivered_order_items",
    "cancel_pending_order", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_pending_order_address",
    "modify_user_address",
}
ID_RE = re.compile(r"(#?W\d{7}|\b\d{10}\b|[a-z]+_\d+)")
OUT_ROOT = Path("/root/autodl-tmp/agent-ttrl/protocols/runs/tau2_official_stream")


def log(msg: str) -> None:
    print(f"[stream] {msg}", flush=True)


# ---------------------------------------------------------------- accessible credit
def trajectory_calls(trajectory) -> list[tuple[dict, str]]:
    """Executed (tool_call, result_content) pairs from the trajectory."""
    out = []
    pending = {}
    for m in trajectory:
        if isinstance(m, AssistantMessage) and m.tool_calls:
            for tc in m.tool_calls:
                pending[tc.id] = tc
        elif isinstance(m, ToolMessage):
            tc = pending.pop(m.id, None)
            if tc is not None:
                out.append((tc, m.content or ""))
    return out


def accessible_success(trajectory) -> dict:
    """Rule-based success from accessible evidence (tool results only).
    Requires properly typed args: list-valued args must be real lists and
    every id-like value must appear in the tool results."""
    calls = trajectory_calls(trajectory)
    results_text = " ".join(r for _, r in calls)
    for tc, _ in calls:
        if tc.name not in MUTATION_TOOLS:
            continue
        ok = True
        id_vals = []
        for v in tc.arguments.values():
            if isinstance(v, list):
                if not v:
                    ok = False
                    break
                id_vals.extend(str(x) for x in v)
            elif isinstance(v, str):
                id_vals.append(v)
            else:
                ok = False
                break
        if not ok:
            continue
        real = [v for v in id_vals if v in results_text]
        if id_vals and len(real) == len(id_vals):
            return {"success": True, "call": tc.name, "args": tc.arguments}
    return {"success": False, "call": None, "args": None}


MAX_SEQ = 768  # training sequences are truncated to fit 14B activations


def render_prompt_ids(policy, messages: list[dict]) -> list[int]:
    prompt = policy.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    ids = policy.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    return ids[-MAX_SEQ:]


def completion_ids(policy, text: str) -> list[int]:
    return policy.tokenizer(text, return_tensors="pt").input_ids[0].tolist()[:128]


# ---------------------------------------------------------------- stream
def run_stream(args) -> int:
    env = get_environment()
    tasks = get_tasks("base")
    task_idxs = [int(x) for x in args.tasks.split(",")]
    out_dir = OUT_ROOT / args.arm / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = ColocatedPolicy(args.model, lora_rank=8, lora_alpha=16,
                             device=args.device)
    log(f"arm={args.arm} seed={args.seed} tasks={task_idxs} "
        f"model={Path(args.model).name}")

    pos_rows, neg_rows = [], []
    per_task = []
    for t_idx, task_idx in enumerate(task_idxs):
        task = tasks[task_idx]
        ins = task.user_scenario.instructions
        known = ins.known_info.replace("You are ", "I am ", 1)
        reason = (ins.reason_for_call.replace("You received", "I received")
                  .replace(" you ", " I ").replace("you wish", "I'd like", 1))
        scenario = f"{known} {reason}"
        agent = ColocatedTau2Agent(tools=env.get_tools(), domain_policy=env.policy,
                                   policy=policy, stream_seed=args.seed,
                                   task_idx=task_idx)
        user = ColocatedUserSimulator(instructions=None, tools=None,
                                      policy=policy, stream_seed=args.seed,
                                      task_idx=task_idx, scenario_str=scenario)
        orch = Orchestrator(domain="retail", agent=agent, user=user,
                            environment=env, task=task, max_steps=args.max_steps)
        try:
            result = run_simulation(orch)
            reward = float(result.reward_info.reward)
        except Exception as e:
            reward = -1.0
            log(f"task {task_idx} crashed: {e}")
            result = None
        acc = accessible_success(result.messages) if result else {"success": False}
        per_task.append({"task": task_idx, "reward": reward, **acc})

        # accessible-evidence row building (update arms only)
        if args.arm != "frozen" and result is not None:
            if acc["success"]:
                prompt_ids = render_prompt_ids(policy, agent._history_to_messages(
                    _state_before_call(agent, result.messages, acc["call"])))
                call_text = f"{acc['call']}({_args_str(acc['args'])})"
                pos_rows.append((prompt_ids, completion_ids(policy, call_text)))
                log(f"task {task_idx}: SUCCESS row (reward {reward})")
            else:
                # negative: the agent's last generated text (raw, unparseable)
                last_text = _last_agent_text(result.messages)
                if last_text:
                    hist = agent._history_to_messages(
                        _state_before_last(agent, result.messages))
                    neg_rows.append((render_prompt_ids(policy, hist),
                                     completion_ids(policy, last_text[:128])))
                log(f"task {task_idx}: failure row (reward {reward})")

        # update + gate + commit (prequential: after the task's first attempt)
        if args.arm == "pair" and (pos_rows or neg_rows):
            _pair_update(policy, pos_rows, neg_rows, args,
                         validate_tasks=[t for t in task_idxs[:t_idx + 1]][-3:])
        elif args.arm == "naive" and pos_rows:
            _naive_update(policy, pos_rows, args)

        entry = {"task": task_idx, "reward": reward, "success": acc["success"]}
        with open(out_dir / f"task_{task_idx}.json", "w") as f:
            json.dump(entry, f, indent=2)
        log(f"task {task_idx} done: reward={reward} accessible_success={acc['success']}")
        import torch
        torch.cuda.empty_cache()

    summary = {"arm": args.arm, "seed": args.seed, "tasks": per_task,
               "n_success": sum(1 for p in per_task if p["success"]),
               "n_reward1": sum(1 for p in per_task if p["reward"] == 1.0)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"SUMMARY {summary}")
    return 0


def _args_str(args_dict: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args_dict.items())


def _state_before_call(agent, messages, call_name: str):
    """Agent state truncated right before the first mutation call."""
    from scripts.tau2_v3_agent import Tau2AgentState
    state = Tau2AgentState([], [])
    for m in messages:
        if isinstance(m, AssistantMessage) and m.tool_calls:
            if any(tc.name == call_name for tc in m.tool_calls):
                break
        if isinstance(m, AssistantMessage):
            state.messages.append(m)
        elif isinstance(m, ToolMessage):
            state.messages.append(m)
        elif hasattr(m, "tool_messages"):
            state.messages.extend(m.tool_messages)
        elif hasattr(m, "content"):
            state.messages.append(m)
    return state


def _state_before_last(agent, messages):
    from scripts.tau2_v3_agent import Tau2AgentState
    state = Tau2AgentState([], [])
    for m in messages:
        if isinstance(m, AssistantMessage) and m.tool_calls is None:
            pass  # include up to the last assistant message's predecessors
    # take everything except the last assistant text message
    items = list(messages)
    last_ai = None
    for i, m in enumerate(items):
        if isinstance(m, AssistantMessage) and not m.tool_calls and (m.content or "").strip():
            last_ai = i
    for i, m in enumerate(items):
        if i == last_ai:
            break
        if isinstance(m, AssistantMessage):
            state.messages.append(m)
        elif isinstance(m, ToolMessage):
            state.messages.append(m)
        elif hasattr(m, "tool_messages"):
            state.messages.extend(m.tool_messages)
        elif hasattr(m, "content"):
            state.messages.append(m)
    return state


def _last_agent_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AssistantMessage) and not m.tool_calls:
            c = (m.content or "").strip()
            if c:
                return c
    return ""


def _pair_update(policy, pos_rows, neg_rows, args, validate_tasks) -> None:
    if not pos_rows or not neg_rows:
        return
    policy.begin_candidate()
    n_steps = min(2, len(pos_rows), len(neg_rows))
    for i in range(n_steps):
        pp, pc = pos_rows[i]
        np_, nc = neg_rows[i]
        try:
            r = policy.train_pair_step(pp, pc, np_, nc, lr=args.lr)
        except Exception as e:
            log(f"pair step failed: {e}")
            return
        log(f"pair step {i}: loss={r['loss']:.4f} logit={r['logit']:.4f}")

    gate_rate = policy.gate_validate(
        lambda state, i: _gate_success(policy, state, i, args), n_per_intent=2)
    log(f"gate improvement rate: {gate_rate:.2f}")
    if gate_rate >= args.gate_threshold:
        rs = RequestSeed("tau2-stream", args.seed, "canary", 0, "canary",
                         policy_version=policy.policy_version)
        cr = policy.commit_candidate("find_user_id_by_name_zip(\"Yusuf\", \"Rossi\", \"19122\")",
                                     rs)
        log(f"commit: passed={cr.passed} version={policy.policy_version} kl={cr.parent_logit_kl:.5f}")
    else:
        policy.rollback()
        log("gate failed: candidate discarded")


def _naive_update(policy, pos_rows, args) -> None:
    policy.begin_candidate()
    for i in range(min(2, len(pos_rows))):
        try:
            r = policy.train_step(pos_rows[i][0], pos_rows[i][1],
                                  advantage=1.0, lr=args.lr)
        except Exception as e:
            log(f"naive step failed: {e}")
            return
        log(f"naive step {i}: loss={r.get('loss', '?')}")
    rs = RequestSeed("tau2-stream", args.seed, "canary", 0, "canary",
                     policy_version=policy.policy_version)
    cr = policy.commit_candidate("find_user_id_by_name_zip(\"Yusuf\", \"Rossi\", \"19122\")",
                                 rs)
    log(f"naive commit: passed={cr.passed}")


def _gate_success(policy, state, i, args) -> int:
    """1 if generating with `state` on a validation prompt yields a tool call."""
    prompt = ("You are a customer service agent. Available tools: "
              "find_user_id_by_name_zip, get_order_details, get_user_details, "
              "return_delivered_order_items, exchange_delivered_order_items. "
              "User: Hi, I am Yusuf Rossi in zip 19122. I received my order "
              "#W2378156 and I'd like to return my headphones, vacuum cleaner, "
              "and smart watch.\nReply with a single tool call.")
    rs = RequestSeed("tau2-stream", args.seed + i, "gate", 0,
                     "production_first_attempt",
                     policy_version=policy.policy_version)
    _, text = policy.generate_with(state, rs, prompt, max_tokens=64,
                                   temperature=0.0)
    return 1 if _parse_calls(text) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["frozen", "naive", "pair"], default="frozen")
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-14B-Instruct")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--gate-threshold", type=float, default=0.5,
                    help="commit if gate improvement rate >= threshold; "
                         "0.0 = always commit (gate-sensitivity arm)")
    args = ap.parse_args()
    return run_stream(args)


if __name__ == "__main__":
    sys.exit(main())

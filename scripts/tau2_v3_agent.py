"""tau2 v3 wrapper: ColocatedTau2Agent (official HalfDuplexAgent) driven by
our policy-consistent ColocatedPolicy. Request-scoped exogenous seeds;
training uses accessible tool observations only; hidden official evaluator
is invoked by the tau2 runner after the episode (reporting only)."""
from __future__ import annotations

import sys

from agent_ttrl.runtime.request_seed import RequestSeed
from tau2.agent.base_agent import HalfDuplexAgent, is_valid_agent_history_message
from tau2.data_model.message import (
    AssistantMessage, MultiToolMessage, SystemMessage, UserMessage, ToolCall,
)

AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the "
    "<policy> provided below. In each turn you can either: send a message to "
    "the user, or make a tool call. You cannot do both at the same time. "
    "Try to be helpful and always follow the policy. "
    "Return ONLY a tool call like: tool_name(arg1=\"v1\", arg2=\"v2\"). "
    "Use EXACT values from the conversation, never invent IDs."
)
SYSTEM_PROMPT = (
    "<instructions>\n{agent_instruction}\n</instructions>\n"
    "<policy>\n{domain_policy}\n</policy>"
)


class Tau2AgentState:
    def __init__(self, system_messages, messages):
        self.system_messages = system_messages
        self.messages = messages


class ColocatedTau2Agent(HalfDuplexAgent):
    def __init__(self, tools, domain_policy, policy, stream_seed, task_idx,
                 turn_seed_offset=0):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.policy = policy
        self.stream_seed = stream_seed
        self.task_idx = task_idx
        self.turn = 0
        self.policy_version = policy.policy_version

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION)

    def get_init_state(self, message_history=None):
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history)
        return Tau2AgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history))

    def _history_to_text(self, state) -> str:
        parts = []
        for m in state.system_messages + state.messages:
            if isinstance(m, SystemMessage):
                parts.append(f"[SYSTEM]\n{m.content}")
            elif isinstance(m, UserMessage):
                parts.append(f"[USER]\n{m.content}")
            elif isinstance(m, MultiToolMessage):
                for tm in m.tool_messages:
                    parts.append(f"[TOOL:{getattr(tm, 'name', '?')}]\n"
                                 f"{getattr(tm, 'content', '')}")
            elif hasattr(m, "content"):
                parts.append(f"[AGENT]\n{m.content}")
        return "\n".join(parts)

    def generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        prompt = self._history_to_text(state)
        seed = RequestSeed("tau2-v3", self.stream_seed, f"t{self.task_idx}",
                           self.turn, "production_first_attempt",
                           policy_version=self.policy.policy_version)
        self.turn += 1
        cid, text = self.policy.generate(seed, prompt, max_tokens=128, temperature=0.3)
        calls = _parse_calls(text)
        tool_calls = [ToolCall(name=n, arguments=kw) for n, kw in calls[:1]]
        msg = AssistantMessage.text(text, tool_calls=tool_calls or None)
        state.messages.append(msg)
        return msg, state


def _parse_calls(text: str) -> list[tuple[str, dict]]:
    import re
    out = []
    for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)", text):
        name, args = m.group(1), m.group(2)
        if name.lower() in {"think", "thought", "let", "i", "the"}:
            continue
        kwargs = {}
        for am in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)=\"?([^,\")]*)\"?", args):
            kwargs[am.group(1)] = am.group(2).strip('"')
        if kwargs:
            out.append((name, kwargs))
    return out

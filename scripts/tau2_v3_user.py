"""Colocated user simulator for tau2: drives the official half-duplex user
with our ColocatedPolicy (transformers, no external LLM). Official
environment/evaluator/orchestrator are used as-is."""
from __future__ import annotations

from agent_ttrl.runtime.request_seed import RequestSeed
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.user.user_simulator_base import HalfDuplexUser


class ColocatedUserState:
    def __init__(self, system_messages, messages):
        self.system_messages = system_messages
        self.messages = messages


class ColocatedUserSimulator(HalfDuplexUser):
    def __init__(self, instructions, tools, policy, stream_seed, task_idx,
                 scenario_str="", llm_args=None):
        super().__init__(instructions=instructions, tools=tools)
        self.policy = policy
        self.stream_seed = stream_seed
        self.task_idx = task_idx
        self.turn = 0
        self.scenario_str = scenario_str

    @property
    def system_prompt(self) -> str:
        return (
            "You are the USER in a customer-service conversation. "
            "Respond as the user naturally, briefly, continuing the conversation. "
            "If the agent asks for information, provide the exact details from your situation. "
            "Say nothing else, never add labels like [AGENT] or [USER]."
        )

    def _first_user_message(self) -> str:
        """Scenario as the user's opening message (in-context persona)."""
        return ("Hi, I need help. " + self.scenario_str).strip()

    def get_init_state(self, message_history=None):
        if message_history is None:
            message_history = []
        return ColocatedUserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history))

    def _history_to_messages(self, state) -> list[dict]:
        """The user sees only agent text messages (tool results are hidden)."""
        msgs = [{"role": "system", "content": self.system_prompt}]
        for m in state.messages:
            if isinstance(m, UserMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AssistantMessage):
                msgs.append({"role": "assistant", "content": m.content or ""})
        msgs = _merge_roles(msgs)
        # Mistral chat template requires the first real turn to be a user turn;
        # the agent always speaks first, so open with the scenario as a user turn.
        if len(msgs) > 1 and msgs[1]["role"] == "assistant":
            msgs.insert(1, {"role": "user", "content": self._first_user_message()})
        return msgs

    def generate_next_message(self, message, state):
        state.messages.append(message)
        # Scripted first turn: the scenario verbatim, so the agent always
        # receives the user's identity (the LLM tends to rephrase it away).
        if self.turn == 0:
            self.turn += 1
            text = self._first_user_message()
            msg = UserMessage.text(text)
            state.messages.append(msg)
            return msg, state
        messages = self._history_to_messages(state)
        seed = RequestSeed("tau2-v3-user", self.stream_seed, f"u{self.task_idx}",
                           self.turn, "production_first_attempt",
                           policy_version=self.policy.policy_version)
        self.turn += 1
        cid, text = self.policy.generate_chat(
            seed, messages, max_tokens=96, temperature=0.5)
        text = text.strip() or "ok."
        if text.lower().startswith(("[agent]", "[user]", "(tool")):
            text = "I'm sorry, what did you say?"
        msg = UserMessage.text(text)
        state.messages.append(msg)
        return msg, state


def _merge_roles(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] = (out[-1]["content"] + "\n" + m["content"]).strip()
        else:
            out.append(dict(m))
    return out

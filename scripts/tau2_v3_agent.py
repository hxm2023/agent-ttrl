"""tau2 v3 wrapper: ColocatedTau2Agent (official HalfDuplexAgent) driven by
our policy-consistent ColocatedPolicy. Request-scoped exogenous seeds;
training uses accessible tool observations only; hidden official evaluator
is invoked by the tau2 runner after the episode (reporting only)."""
from __future__ import annotations

import sys

from agent_ttrl.runtime.request_seed import RequestSeed
from tau2.agent.base_agent import HalfDuplexAgent, is_valid_agent_history_message
from tau2.data_model.message import (
    AssistantMessage, MultiToolMessage, SystemMessage, ToolMessage, UserMessage,
    ToolCall,
)

AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the "
    "<policy> provided below. In each turn you can either: send a message to "
    "the user, or make a tool call. You cannot do both at the same time. "
    "ALWAYS actually call the tool before confirming anything to the user. "
    "Never claim an action was done unless a tool call for it succeeded. "
    "The LAST message is the latest user/tool message. Reply to it now with "
    "ONLY ONE of the following: (a) a short message to the user, or (b) a "
    "single tool call of the form tool_name(arg1=\"value\", arg2=\"value\") "
    "using EXACTLY one of the tool names listed in <tools>, with EXACT "
    "argument names and EXACT values from the conversation. Never invent "
    "IDs, never invent tool names, never wrap the call in parentheses, "
    "never include any other text next to the call.\n"
    "\n"
    "Rules:\n"
    "- Use EXACTLY the name, zip, and order number the user stated. Never "
    "invent an email address, name, or ID.\n"
    "- If the user gave name + zip code, use find_user_id_by_name_zip. "
    "Only use find_user_id_by_email if the user actually stated an email.\n"
    "- Call get_order_details(order_id=...) BEFORE any exchange/return/"
    "cancel/modify tool, to learn the real item ids.\n"
    "- To count how many options exist for a product type (e.g. tshirts), "
    "call list_all_product_types() and count the matching entries; do NOT "
    "pass words like \"tshirt\" as an id.\n"
    "- Never invent ids: every id must come from a tool result or the user's "
    "message. get_product_details/get_item_details take NUMERIC ids found in "
    "earlier tool results.\n"
    "- If a tool returns an Error, NEVER repeat the same call with the same "
    "arguments — fix the arguments or pick a different tool.\n"
    "- NEVER call the same tool twice in a row.\n"
    "- Ask the user for explicit confirmation (yes/no) before any action "
    "that changes the order.\n"
    "\n"
    "Example (ALL example values are fake — use ONLY values from THIS "
    "conversation, never angle brackets, never the example's ids):\n"
    "User: Hi, I'm Jane Smith in zip 94041. I received my order #W1234567 "
    "and I'd like to exchange the mouse.\n"
    "Agent: get_order_details(order_id=\"#W1234567\")\n"
    "Tool: order found for user user_jane_smith_001\n"
    "Agent: find_user_id_by_name_zip(first_name=\"Jane\", last_name=\"Smith\", "
    "zip=\"94041\")\n"
    "Tool: user found: user_jane_smith_001\n"
    "Agent: get_user_details(user_id=\"user_jane_smith_001\")\n"
    "Tool: payment methods: gift_card_0000001\n"
    "Agent: exchange_delivered_order_items(order_id=\"#W1234567\", "
    "item_ids=[\"1000000001\"], new_item_ids=[\"1000000002\"], "
    "payment_method_id=\"gift_card_0000001\")\n"
    "Tool: exchange succeeded"
)
SYSTEM_PROMPT = (
    "<instructions>\n{agent_instruction}\n</instructions>\n"
    "<tools>\n{tools_text}\n</tools>\n"
    "<policy>\n{domain_policy}\n</policy>"
)


def _tools_to_text(tools) -> str:
    """Serialize Tool objects (openai_schema) into a plain-text tool list.
    Descriptions are trimmed to the first sentence to keep the prompt small."""
    lines = []
    if isinstance(tools, dict):
        tool_items = [(name, tool) for name, tool in tools.items()]
    else:
        tool_items = [(getattr(t, "name", None), t) for t in tools]
    for name, tool in tool_items:
        schema = getattr(tool, "openai_schema", None)
        if isinstance(schema, dict) and "function" in schema:
            fn = schema["function"]
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            required = set(params.get("required", []) or [])
        else:
            fn = {"name": name, "description": str(getattr(tool, "description", ""))}
            params, props, required = {}, {}, set()
        desc = fn.get("description", "")
        if desc:
            desc = desc.split(". ")[0] + ("." if desc else "")
        arg_lines = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "")
            pdesc = pinfo.get("description", "")
            pdesc = pdesc.split(". ")[0] if pdesc else ""
            opt = "" if pname in required else " (optional)"
            arg_lines.append(f"    {pname}{opt}: {ptype} — {pdesc}")
        lines.append(f"tool {fn.get('name', name)}")
        if desc:
            lines.append(f"  description: {desc}")
        if arg_lines:
            lines.append("  arguments:")
            lines.extend(arg_lines)
        else:
            lines.append("  arguments: none")
    return "\n".join(lines)


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
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION,
            tools_text=_tools_to_text(self.tools))

    def get_init_state(self, message_history=None):
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history)
        return Tau2AgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history))

    def _history_to_messages(self, state) -> list[dict]:
        """Chat messages with tool results paired to their calls by id.
        ToolMessage does not carry the tool name; we resolve it from the
        preceding AssistantMessage's ToolCall list (matched by id)."""
        msgs = [{"role": "system", "content": self.system_prompt}]
        pending: dict[str, str] = {}
        for m in state.messages:
            if isinstance(m, UserMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AssistantMessage):
                content = m.content or ""
                if m.tool_calls:
                    for tc in m.tool_calls:
                        pending[tc.id] = tc.name
                    call_lines = [f"{tc.name}({_args_str(tc.arguments)})"
                                  for tc in m.tool_calls]
                    content = (content + "\n" + "\n".join(call_lines)).strip()
                msgs.append({"role": "assistant", "content": content})
            elif isinstance(m, ToolMessage):
                name = pending.pop(m.id, "?")
                body = _digest_tool_result(name, m.content or "") if name != "?" else (m.content or "")
                body = f"Error: {body}" if m.error else body
                msgs.append({"role": "user", "content": f"[TOOL {name}]\n{body}"})
            elif isinstance(m, MultiToolMessage):
                for tm in m.tool_messages:
                    name = pending.pop(tm.id, "?")
                    body = tm.content or ""
                    body = f"Error: {body}" if tm.error else body
                    msgs.append({"role": "user",
                                 "content": f"[TOOL {name}]\n{body}"})
        msgs = _merge_roles(msgs)
        # Drop the orchestrator's seeded default greeting (leading assistant
        # turn) — Mistral requires user first, and the greeting is trivial.
        while len(msgs) > 1 and msgs[1]["role"] == "assistant":
            del msgs[1]
        return msgs

    def _track_last_result(self, message) -> None:
        """Record whether the most recent tool result was an error (used by
        the deterministic anti-loop valve below)."""
        if isinstance(message, MultiToolMessage):
            errs = [tm.error for tm in message.tool_messages]
            self.last_error = bool(errs) and any(errs)
        elif isinstance(message, ToolMessage):
            self.last_error = bool(getattr(message, "error", False))
        else:
            self.last_error = False

    def generate_next_message(self, message, state):
        self._track_last_result(message)
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = self._history_to_messages(state)
        seed = RequestSeed("tau2-v3", self.stream_seed, f"t{self.task_idx}",
                           self.turn, "production_first_attempt",
                           policy_version=self.policy.policy_version)
        self.turn += 1
        cid, text = self.policy.generate_chat(
            seed, messages, max_tokens=128, temperature=0.3)
        calls = _parse_calls(text)
        if calls:
            name, kw = calls[0]
            repeated = (name, kw) == getattr(self, "last_call", None)
            if repeated:
                # Deterministic anti-loop valve: the model repeated the exact
                # last call; force a text reply instead. last_call persists so
                # the valve keeps blocking until the model proposes something
                # new (it only resets when a different call is emitted).
                calls = []
                text = ("I need a moment to review what we know so far. "
                        "Could you confirm your order number and what you'd "
                        "like to do?")
        if not calls and not text.strip():
            text = "I'm sorry, could you repeat that?"
        if calls:
            self.last_call = calls[0]
        tool_calls = [ToolCall(name=n, arguments=kw) for n, kw in calls[:1]]
        msg = AssistantMessage.text(text, tool_calls=tool_calls or None)
        state.messages.append(msg)
        return msg, state


def _opts_str(options) -> str:
    if not isinstance(options, dict):
        return ""
    return ", ".join(f"{k}={v}" for k, v in options.items())


def _digest_tool_result(name: str, content: str) -> str:
    """Compact digest of large tool results (same serving-side compression
    as tau2_local_server.py)."""
    import json
    try:
        d = json.loads(content)
    except Exception:
        return content
    try:
        if name == "get_order_details" and isinstance(d, dict):
            lines = [f"order {d.get('order_id')} | status: {d.get('status')} | "
                     f"user: {d.get('user_id')}"]
            for it in d.get("items", []) or []:
                if not isinstance(it, dict):
                    continue
                opts = _opts_str(it.get("options"))
                lines.append(f"  item {it.get('item_id')}: {it.get('name')} "
                             f"(${it.get('price')}){(' ' + opts) if opts else ''}")
            return "\n".join(lines)
        if name == "get_user_details" and isinstance(d, dict):
            nm = d.get("name") or {}
            lines = [f"user {d.get('user_id')} | "
                     f"{nm.get('first_name') if isinstance(nm, dict) else ''} "
                     f"{nm.get('last_name') if isinstance(nm, dict) else ''} "
                     f"| email: {d.get('email')}"]
            pm = d.get("payment_methods") or {}
            if isinstance(pm, dict) and pm:
                lines.append(f"  payment methods: {', '.join(pm.keys())}")
            lines.append(f"  orders: {', '.join(d.get('orders') or [])}")
            return "\n".join(lines)
        if name in ("get_product_details", "get_item_details") and isinstance(d, dict):
            variants = d.get("variants")
            if isinstance(variants, dict):
                lines = [f"product {d.get('name')} ({d.get('product_id')}) | "
                         f"{len(variants)} variants:"]
                for iid, v in variants.items():
                    if not isinstance(v, dict):
                        continue
                    avail = "available" if v.get("available") else "unavailable"
                    lines.append(f"  item {iid}: {_opts_str(v.get('options'))} "
                                 f"(${v.get('price')}, {avail})")
                return "\n".join(lines)
            if isinstance(d.get("options"), dict):
                return (f"item {d.get('item_id')} of product "
                        f"{d.get('product_name') or d.get('name')}: "
                        f"{_opts_str(d.get('options'))} "
                        f"(${d.get('price')}, "
                        f"{'available' if d.get('available') else 'unavailable'})")
    except Exception:
        pass
    return content


def _args_str(arguments: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in arguments.items())


def _merge_roles(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] = (out[-1]["content"] + "\n" + m["content"]).strip()
        else:
            out.append(dict(m))
    return out


def _parse_kwargs(args: str) -> dict:
    import json
    import re
    s = args.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    kwargs = {}
    for am in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)=\"?([^,\")]*)\"?", args):
        kwargs[am.group(1)] = am.group(2).strip('"')
    return kwargs


def _parse_calls(text: str) -> list[tuple[str, dict]]:
    import re
    out = []
    # tau-bench style: (tool name(args)) or (tool name)
    for m in re.finditer(r"\(tool\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\(([^)]*)\))?\)",
                         text):
        out.append((m.group(1), _parse_kwargs(m.group(2) or "")))
    if out:
        return out
    for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)", text):
        name, args = m.group(1), m.group(2)
        if name.lower() in {"think", "thought", "let", "i", "the", "tool"}:
            continue
        kwargs = _parse_kwargs(args)
        if kwargs:
            out.append((name, kwargs))
    return out

"""OpenAI-compatible local endpoint wrapping ColocatedPolicy.

Lets the OFFICIAL tau2 pipeline (LLMAgent + UserSimulator, both driven via
litellm) run against our local transformers model: litellm sends OpenAI-format
chat completions with function schemas; this server renders them through the
model's chat template and returns OpenAI-format assistant messages with
tool_calls parsed from the generated text.

Request-scoped determinism: each request draws its RNG seed from the litellm
``seed`` kwarg (set via llm_args) or from a server-side counter; the torch
RNG state is restored after every generation.
"""
from __future__ import annotations

import argparse
import json
import re
import threading

import torch
import uvicorn
from fastapi import FastAPI, Request

from agent_ttrl.runtime.request_seed import RequestSeed
from agent_ttrl.runtime.served_policy import ColocatedPolicy

app = FastAPI()
POLICY: ColocatedPolicy = None
MODEL_NAME = "local-model"
_counter = 0
_lock = threading.Lock()

STOP_NAMES = {"think", "thought", "let", "i", "the", "tool"}


def _next_seed(seed_arg) -> int:
    """Distinct seed per request; reproducible sequence for the same base."""
    global _counter
    with _lock:
        _counter += 1
        base = seed_arg if isinstance(seed_arg, int) else 0
        return base * 1000000 + _counter


def _value(v: str):
    v = v.strip().strip('"')
    try:
        return json.loads(v)
    except Exception:
        return v


def _parse_calls(text: str) -> list[tuple[str, dict]]:
    out = []
    for m in re.finditer(r"\(tool\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\(([^)]*)\))?\)",
                         text):
        out.append((m.group(1), _parse_args(m.group(2) or "")))
    if out:
        return out
    for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)", text):
        name, args = m.group(1), m.group(2)
        if name.lower() in STOP_NAMES:
            continue
        out.append((name, _parse_args(args)))
    return out


def _parse_args(args: str) -> dict:
    """Parse ``k1="v1", k2=[...]`` OR JSON ``{"k1": "v1"}`` into typed dict."""
    s = args.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    out = {}
    for am in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)=\"?([^,\")]*)\"?", args):
        out[am.group(1)] = _value(am.group(2))
    return out


def _last_executed_call(messages: list[dict]):
    """(name, args) of the last assistant tool call in the conversation."""
    for m in reversed(messages):
        tcs = m.get("tool_calls") or []
        if tcs:
            tc = tcs[-1]
            fn = tc.get("function", {})
            try:
                return (fn.get("name"), json.loads(fn.get("arguments") or "{}"))
            except Exception:
                return (fn.get("name"), fn.get("arguments"))
    return None


def _first_json_object(text: str):
    """Extract the first balanced {...} object from text (brace counting)."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def _parse_bare_json_calls(text: str) -> list[tuple[str, dict]]:
    """Qwen2.5 native tool call emitted as a bare JSON object:
    {"name": "get_product_details", "arguments": {...}}"""
    obj = _first_json_object(text)
    if not obj:
        return []
    try:
        d = json.loads(obj)
    except Exception:
        return []
    if isinstance(d, dict) and d.get("name") and isinstance(d.get("arguments"), dict):
        return [(d["name"], d["arguments"])]
    return []


def _parse_qwen_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Parse native Qwen tool-call blocks: <tool_call> / <|tool_call|> followed
    by a JSON object {"name":.., "arguments":..}."""
    out = []
    for m in re.finditer(r"<\|?tool_call\|?>(.*?)(?:<\|?/?\|?tool_call\|?>|$)",
                         text, flags=re.S):
        try:
            d = json.loads(m.group(1).strip())
        except Exception:
            continue
        name = d.get("name")
        args = d.get("arguments")
        if name and isinstance(args, dict):
            out.append((name, args))
    return out


def _parse_prose_calls(text: str) -> list[tuple[str, dict]]:
    """Fallback for weak models that describe the call in prose:
    ``Calling tool: name ... Arguments: k=v, k2=v2`` or ``name(k=v)``."""
    out = []
    for m in re.finditer(
            r"(?:call(?:ing)?\s+(?:the\s+)?tool\s*:?\s*|tool\s+)([a-z_][a-z0-9_]*)",
            text, flags=re.I):
        name = m.group(1)
        if name.lower() in STOP_NAMES or not name:
            continue
        kwargs = {}
        seg = text[m.end():m.end() + 300]
        am = re.search(r"(?:arguments?\s*[:=]\s*)([^\n]+)", seg, flags=re.I)
        if am:
            for kv in re.finditer(r"([a-z_][a-z0-9_]*)\s*=\s*\"?([^,\"\n]+)\"?",
                                  am.group(1)):
                kwargs[kv.group(1)] = _value(kv.group(2))
        if not kwargs:
            am2 = re.search(r"\(([^)]*)\)", seg)
            if am2:
                for kv in re.finditer(r"([a-z_][a-z0-9_]*)\s*=\s*\"?([^,\")]*)\"?",
                                      am2.group(1)):
                    kwargs[kv.group(1)] = _value(kv.group(2))
        if kwargs:
            out.append((name, kwargs))
            break
    return out


AGENT_HINT = (
    "Reply with either (a) a short message to the user, or (b) a tool call "
    "using EXACTLY one of the tools listed above, with EXACT argument names "
    "and EXACT values from the conversation. Never invent IDs, order numbers, "
    "or emails. To count how many options exist for a product type, call "
    "list_all_product_types() and count the matching entries. If the user "
    "did NOT give an order number, call get_user_details(user_id=...) to "
    "list the user's orders, then use the real order id from that result. "
    "Look up order details before any exchange/return/cancel/modify. If a "
    "tool returns an error, never repeat the same call with the same "
    "arguments."
)


def _tools_block(tools: list) -> str:
    if not tools:
        return ""
    lines = ["<tools>"]
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        if desc:
            desc = desc.split(". ")[0] + ("." if desc else "")
        lines.append(f"tool {name}")
        if desc:
            lines.append(f"  description: {desc}")
        params = fn.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        req = set(params.get("required", []) or [])
        if props:
            lines.append("  arguments:")
            for pname, pinfo in props.items():
                pdesc = (pinfo.get("description", "") or "").split(". ")[0]
                opt = "" if pname in req else " (optional)"
                lines.append(f"    {pname}{opt}: {pinfo.get('type', '')} — {pdesc}")
        else:
            lines.append("  arguments: none")
    lines.append("</tools>")
    return "\n".join(lines) + "\n" + AGENT_HINT


def _render_messages(messages: list[dict], tools_text: str) -> list[dict]:
    """Convert OpenAI messages (with tool_calls / role=tool) to chat template
    messages. Tool results are paired to their calls by id, like the agent."""
    msgs = []
    pending: dict[str, str] = {}
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            if tools_text:
                content = content + "\n" + tools_text
            else:
                # Tool-less requests are the USER simulator: weak models drift
                # into meta-analysis unless told to speak in-character.
                content = (content +
                           "\nYou are role-playing a customer in this "
                           "conversation. Respond ONLY as that customer would, "
                           "in the first person, briefly. Never analyze the "
                           "conversation, never plan actions for the agent.")
            msgs.append({"role": "system", "content": content})
        elif role == "user":
            msgs.append({"role": "user", "content": content})
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    pending[tc.get("id", "")] = fn.get("name", "?")
                call_lines = [f"{tc['function']['name']}({tc['function'].get('arguments', '')})"
                              for tc in tcs]
                content = (content + "\n" + "\n".join(call_lines)).strip()
            msgs.append({"role": "assistant", "content": content})
        elif role == "tool":
            name = pending.pop(m.get("tool_call_id", ""), "?")
            msgs.append({"role": "user",
                         "content": f"[TOOL {name}]\n{content}"})
    # merge consecutive same-role; drop leading assistant (template requires
    # the conversation to open with a user turn)
    merged = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] = (merged[-1]["content"] + "\n" + m["content"]).strip()
        else:
            merged.append(dict(m))
    while len(merged) > 1 and merged[1]["role"] == "assistant":
        del merged[1]
    return merged


def _debug_log(entry: dict) -> None:
    try:
        with open("/root/autodl-tmp/agent-ttrl/logs_local_reqs.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    _debug_log({"ALL": len(messages), "roles": [m.get("role") for m in messages],
                "has_tools": bool(tools),
                "sys0": (messages[0].get("content", "")[:80] if messages else "")})
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", 256))
    seed = _next_seed(body.get("seed"))
    tools_text = _tools_block(tools)
    msgs = _render_messages(messages, tools_text)
    rs = RequestSeed("tau2-official", seed, "req", 0,
                     "production_first_attempt",
                     policy_version=POLICY.policy_version)
    if tools:
        # Native function calling (Qwen2.5/Qwen3 templates): render tools into
        # the chat template and parse <|tool_call|> blocks from the output.
        cid, text = POLICY.generate_chat(rs, msgs, max_tokens=max_tokens,
                                         temperature=temperature, tools=tools,
                                         keep_special_tokens=True,
                                         template_kwargs={
                                             "enable_thinking": False})
        calls = _parse_qwen_tool_calls(text)
        if not calls:
            calls = _parse_calls(text)
        if not calls:
            calls = _parse_bare_json_calls(text)
        if not calls:
            calls = _parse_prose_calls(text)
        last_exec = _last_executed_call(messages)
        _debug_log({"n_msgs": len(messages), "last_exec": last_exec,
                    "parsed": calls[:1], "raw": text[:200]})
        if calls and last_exec == calls[0]:
            # Anti-loop valve: the model repeated the exact last call.
            calls = []
            text = ("I need a moment to review what we know so far. "
                    "Could you confirm your order number and what you'd "
                    "like to do?")
        text = re.sub(r"<\|tool_call\|>.*", "", text, flags=re.S).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"^<think>[^\n]*", "", text, flags=re.S).strip()
    else:
        _debug_log({"n_msgs": len(messages), "sys_prompt": (msgs[0]["content"] if msgs else "")[:120]})
        cid, text = POLICY.generate_chat(rs, msgs, max_tokens=max_tokens,
                                         temperature=temperature)
        text = text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"^<think>[^\n]*", "", text, flags=re.S)
        text = text.strip()
        calls = _parse_calls(text)
    if calls:
        tool_calls = [{
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        } for i, (name, args) in enumerate(calls[:1])]
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None,
                            "tool_calls": tool_calls},
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
        }
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": text or "I'm sorry, could you repeat that?"},
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model-name", default="local-model")
    args = ap.parse_args()

    global POLICY, MODEL_NAME
    POLICY = ColocatedPolicy(args.model, lora_rank=8, lora_alpha=16,
                             device=args.device)
    MODEL_NAME = args.model_name
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

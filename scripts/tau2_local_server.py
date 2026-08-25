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
    global _counter
    with _lock:
        if isinstance(seed_arg, int):
            return seed_arg
        _counter += 1
        return _counter


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
        parsed = _parse_args(args)
        if parsed:
            out.append((name, parsed))
    return out


def _parse_args(args: str) -> dict:
    """Parse ``k1="v1", k2=[...], k3=123`` into typed JSON-ish dict."""
    out = {}
    for am in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)=\"?([^,\")]*)\"?", args):
        out[am.group(1)] = _value(am.group(2))
    return out


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
    return "\n".join(lines)


def _render_messages(messages: list[dict], tools_text: str) -> list[dict]:
    """Convert OpenAI messages (with tool_calls / role=tool) to chat template
    messages. Tool results are paired to their calls by id, like the agent."""
    msgs = []
    pending: dict[str, str] = {}
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            content = content + ("\n" + tools_text if tools_text else "")
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", 256))
    seed = _next_seed(body.get("seed"))
    tools_text = _tools_block(tools)
    msgs = _render_messages(messages, tools_text)
    rs = RequestSeed("tau2-official", seed, "req", 0,
                     "production_first_attempt",
                     policy_version=POLICY.policy_version)
    cid, text = POLICY.generate_chat(rs, msgs, max_tokens=max_tokens,
                                     temperature=temperature)
    text = text.strip()
    # Qwen3 emits <think>...</think> reasoning blocks; strip them (including
    # an unclosed leading block) so tool calls and final content are clean.
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

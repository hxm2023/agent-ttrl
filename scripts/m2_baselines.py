"""M2: CTS baseline reproduction (design doc §10).

Family A/B baselines on the CTS dev stream, same prequential protocol as M3:
  frozen         : no adaptation (floor; = M3 frozen)
  best_of_n      : 8 samples, verifier(utility)-selected answer, no update
  reflexion      : memory-based — failed-task errors appended to next prompt
  hard_verifier  : terminal reward uses hard evidence only (no soft verifier)

All baselines share the identical task stream and first-attempt hidden scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_ROOT = Path(os.environ.get("ATTRL_M2_OUT", "/root/autodl-tmp/agent-ttrl/artifacts/m2"))
ATTRL_DIR = Path(os.environ.get("ATTRL_DIR", "/root/autodl-tmp/agent-ttrl"))
sys.path.insert(0, str(ATTRL_DIR / "src"))

MAX_COMPLETION = 128
N_TASKS = 8
BO_N = 8

TASK_SPECS = [
    {"sku": "sku:a", "addr": "addr-1", "order": "o1", "user": "u1"},
    {"sku": "sku:b", "addr": "addr-2", "order": "o2", "user": "u2"},
    {"sku": "sku:a", "addr": "addr-3", "order": "o3", "user": "u1"},
    {"sku": "sku:c", "addr": "addr-1", "order": "o4", "user": "u3"},
    {"sku": "sku:b", "addr": "addr-1", "order": "o5", "user": "u2"},
    {"sku": "sku:a", "addr": "addr-2", "order": "o6", "user": "u3"},
    {"sku": "sku:c", "addr": "addr-2", "order": "o7", "user": "u1"},
    {"sku": "sku:b", "addr": "addr-3", "order": "o8", "user": "u2"},
]

BASE_PROMPT = """You are an order assistant. Available tools (JSON list of {{"tool": ..., "call": {{...}}}}):
- reserve_item {{item_key, order_id}}
- create_order {{order_id}}
- charge {{order_id, user_id, amount_cents}}
- ship {{order_id, user_id, address}}
- complete_task {{}}
Task: user {user} wants item {sku} shipped to address {addr}. Call the tools in order.
Return ONLY the JSON list of tool calls."""

VARIANT = "frozen"


def log(msg: str) -> None:
    print(f"[m2:{VARIANT}] {msg}", flush=True)


def parse_tool_calls(text: str) -> list[dict]:
    import re
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        calls = json.loads(m.group(0))
        return [c for c in calls if isinstance(c, dict) and isinstance(c.get("tool"), str)
                and isinstance(c.get("call"), dict)] if isinstance(calls, list) else []
    except Exception:
        return []


def make_world(spec: dict):
    from agent_ttrl.environments.cts_world import WorldState
    return WorldState(inventory={"sku:a": 5, "sku:b": 3, "sku:c": 1},
                      balance={spec["user"]: 100_000},
                      address={spec["user"]: spec["addr"]},
                      permission_scope=["payment", "shipping"])


def run_episode(state, calls, config, spec, use_soft: bool = True) -> tuple[float, dict]:
    from agent_ttrl.environments.cts_evidence import AccessibleEvidence, evidence_utility
    from agent_ttrl.environments.cts_oracle import GoalSpec, hidden_score
    from agent_ttrl.environments.cts_world import advance_turn, transition

    errors = []
    for call in calls:
        try:
            st2, _ = transition(state, call.get("tool", ""), call.get("call", {}), config)
            state = st2
            state = advance_turn(state)
        except ValueError as e:
            errors.append(str(e))
    goal = GoalSpec(user_id=spec["user"], want_item=spec["sku"], want_address=spec["addr"])
    bundle = AccessibleEvidence(state, goal).collect()
    if not use_soft:
        bundle.soft_evidence = []
    return evidence_utility(bundle, goal), {
        "hidden": hidden_score(state, goal).success, "errors": errors[:4]}


def patch_device_normalization() -> None:
    import torch
    import trl
    import vllm
    from trl.generation.vllm_client import VLLMClient
    assert trl.__version__ == "1.10.0" and vllm.__version__ == "0.26.0"
    _orig = VLLMClient.init_communicator

    def _normalized(self, device, *a, **kw):
        if isinstance(device, torch.device) and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        return _orig(self, device, *a, **kw)

    VLLMClient.init_communicator = _normalized


def start_server(server_log: Path, port: int, server_gpu: int = 1) -> subprocess.Popen:
    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [trl_bin, "vllm-serve", "--model", MODEL_PATH, "--port", str(port),
         "--gpu-memory-utilization", "0.4", "--max-model-len", "2048"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(server_gpu)},
        stdout=open(server_log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    for _ in range(180):
        time.sleep(2)
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"server died: {Path(server_log).read_text()[-2000:]}")
    raise RuntimeError("server not healthy in 360s")


def stop_server(proc: subprocess.Popen, port: int) -> None:
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    subprocess.run(
        ["bash", "-c",
         f"for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do "
         f"[ \"$p\" = \"$$\" ] && continue; "
         f"cmd=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null); "
         f"if echo \"$cmd\" | grep -qE 'vllm-serve|VLLM::EngineCore'; then kill -9 $p 2>/dev/null; fi; done"],
        capture_output=True)
    time.sleep(5)


def _unpack_gen(res: dict):
    return (res["prompt_ids"], res["completion_ids"], res["logprobs"], res.get("logprob_token_ids"))


def main() -> int:
    global VARIANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["frozen", "best_of_n", "reflexion", "hard_verifier"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--group-port", type=int, default=51700)
    ap.add_argument("--server-gpu", type=int, default=1)
    args = ap.parse_args()
    VARIANT = args.variant

    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient
    OUT_DIR = OUT_ROOT / f"{VARIANT}_s{args.seed}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_device_normalization()

    server = start_server(OUT_DIR / "vllm_server.log", args.port, args.server_gpu)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{args.port}", group_port=args.group_port,
                            connection_timeout=300)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        from agent_ttrl.environments.cts_world import ShiftConfig
        config = ShiftConfig()

        memory_notes: list[str] = []
        stream_log = []
        for t_idx in range(N_TASKS):
            spec = TASK_SPECS[t_idx % len(TASK_SPECS)]
            prompt = BASE_PROMPT.format(**spec)
            if VARIANT == "reflexion" and memory_notes:
                prompt += "\n\nLessons from earlier tasks:\n" + "\n".join(memory_notes)

            if VARIANT == "best_of_n":
                best_u, best_hidden, best_text = -1.0, False, ""
                for _ in range(BO_N):
                    res = client.generate([prompt], n=1, temperature=1.0, top_p=1.0,
                                          top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
                    pid, cid, _, _ = _unpack_gen(res)
                    text = tokenizer.decode(cid[0], skip_special_tokens=True)
                    u, info = run_episode(make_world(spec), parse_tool_calls(text), config, spec)
                    if u > best_u:
                        best_u, best_hidden, best_text = u, info["hidden"], text
                y_pre = 1.0 if best_hidden else 0.0
                stream_log.append({"task": t_idx, "y_pre": y_pre, "u_pre": round(best_u, 3),
                                   "samples": BO_N, "updated": False})
            else:
                res = client.generate([prompt], n=1, temperature=1.0, top_p=1.0,
                                      top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
                pid, cid, _, _ = _unpack_gen(res)
                text = tokenizer.decode(cid[0], skip_special_tokens=True)
                use_soft = VARIANT != "hard_verifier"
                u, info = run_episode(make_world(spec), parse_tool_calls(text), config, spec,
                                      use_soft=use_soft)
                y_pre = 1.0 if info["hidden"] else 0.0
                stream_log.append({"task": t_idx, "y_pre": y_pre, "u_pre": round(u, 3),
                                   "updated": False, "errors": info["errors"]})
                if VARIANT == "reflexion" and info["errors"]:
                    # memory gated on E_hard exec errors ONLY — never on the
                    # hidden evaluator verdict (protocol red line 1)
                    memory_notes.append(f"- Task {t_idx}: avoid errors: " + ", ".join(info["errors"]))

        aupc = float(np.mean([s["y_pre"] for s in stream_log]))
        report = {"run_id": f"m2-{VARIANT}-s{args.seed}", "variant": VARIANT, "seed": args.seed,
                  "aupc_prequential": round(aupc, 4), "tasks": stream_log,
                  "parallel_with": "GRPO-Guard-idle"}
        (OUT_DIR / "run_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"AUPC_prequential={aupc:.4f}")
        return 0
    finally:
        stop_server(server, args.port)


if __name__ == "__main__":
    sys.exit(main())

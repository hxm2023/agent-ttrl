"""M5: AppWorld frozen baseline — environment closure check (design doc §9.2).

Loads AppWorld dev-role tasks, runs the frozen Qwen3-4B agent loop:
instruction + api_docs -> LLM generates API calls -> execute -> observation
-> repeat until task_completed or budget exhausted -> hidden evaluator score.
Prequential first-attempt scoring on a task stream.

This is the environment-closure run (AppWorld adapter correctness), not a
method comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_ROOT = Path(os.environ.get("ATTRL_M5_OUT", "/root/autodl-tmp/agent-ttrl/artifacts/m5"))
MAX_TURNS = 12
MAX_API_CALLS = 15

SYSTEM_PROMPT = """You are an assistant that completes tasks by calling application APIs.
Call format: apis.app_name.function_name(arg1="value1", arg2="value2")
Example (real call): apis.spotify.search_playlists(query="top r&b playlists")
Only output REAL API calls for THIS task. Never repeat this instruction or examples.
Available apps and their API docs:
{api_docs}
Task: {instruction}
Return ONLY API calls, one per line."""


def log(msg: str) -> None:
    print(f"[m5] {msg}", flush=True)


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


def start_server(server_log: Path, port: int) -> subprocess.Popen:
    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [trl_bin, "vllm-serve", "--model", MODEL_PATH, "--port", str(port),
         "--gpu-memory-utilization", "0.4", "--max-model-len", "4096"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default=None, help="specific task id (dev role); default: first task")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--group-port", type=int, default=53000)
    ap.add_argument("--max-tasks", type=int, default=1)
    args = ap.parse_args()

    # ---- AppWorld environment (appworld-venv python) ----
    APWORLD_PY = "/root/autodl-tmp/appworld-venv/bin/python"
    APWORLD_ROOT = "/root/autodl-tmp/appworld"
    OUT_DIR = OUT_ROOT / "frozen_s0"
    APWORLD_SERVER_PORT = 8700
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient
    patch_device_normalization()

    aw_server = subprocess.Popen(
        ["/root/autodl-tmp/appworld-venv/bin/python", "/root/autodl-tmp/agent-ttrl/scripts/appworld_server.py"],
        env={**os.environ, "APWORLD_SERVER_PORT": str(8700)}, start_new_session=True)
    time.sleep(8)
    server = start_server(OUT_DIR / "vllm_server.log", args.port)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{args.port}", group_port=args.group_port,
                            connection_timeout=300)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # ---- load task + api docs via appworld-venv (server-side helper) ----
        helper = r'''
import json, sys
sys.path.insert(0, '/root/autodl-tmp/appworld/src')
sys.path.insert(0, '/root/autodl-tmp/appworld/src/appworld')
from appworld.environment import AppWorld
from appworld.api_docs import prepare_api_docs
from appworld.task import Task, load_task_ids
task_ids = load_task_ids("dev")
task_id_arg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else task_ids[0]
task = Task.load(task_id_arg)
apps = getattr(task, "allowed_apps", None) or ["amazon", "booking", "facebook", "gmail", "linkedin", "quora", "slack", "spotify", "uber", "venmo", "whatsapp", "yelp"]
docs = {}
for app in apps:
    try:
        docs[app] = prepare_api_docs(app, format="function_calling")
    except Exception as e:
        docs[app] = {"error": str(e)[:100]}
print(json.dumps({"instruction": task.instruction, "api_docs": docs, "task_id": task.id}))
'''
        import tempfile
        helper_path = OUT_DIR / "appworld_helper.py"
        helper_path.write_text(helper, encoding="utf-8")
        task_id = args.task_id or ""
        res = subprocess.run([APWORLD_PY, str(helper_path), task_id],
                             capture_output=True, text=True, cwd=APWORLD_ROOT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", "APPWORLD_ROOT": "/root/autodl-tmp/appworld"})
        if res.returncode != 0:
            print("helper failed:", res.stderr[-2000:], flush=True)
            return 2
        task_meta = json.loads(res.stdout.strip().splitlines()[-1])
        instruction = task_meta["instruction"]
        docs = task_meta["api_docs"]
        docs_text = json.dumps(docs, indent=1)[:4000]
        log(f"task {task_meta['task_id']}: {instruction[:80]}")

        # ---- agent loop: LLM generates API calls, AppWorld executes ----
        import urllib.request

        def http_post(path: str, payload: dict) -> dict:
            req = urllib.request.Request(f"http://127.0.0.1:{APWORLD_SERVER_PORT}{path}",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())

        for _try in range(5):
            try:
                http_post("/init", {"task_id": task_meta["task_id"]})
                break
            except Exception as e:
                if _try == 4:
                    raise
                time.sleep(10)

        def execute_calls(calls_text: str) -> str:
            try:
                out = http_post("/exec", {"code": calls_text})
                return json.dumps(out)[:600]
            except Exception as e:
                return f"EXEC_HTTP_ERROR: {e}"

        conversation = ""
        n_calls = 0
        completed = False
        turn_log = []
        for turn in range(MAX_TURNS):
            prompt = SYSTEM_PROMPT.format(api_docs=docs_text, instruction=instruction)
            if conversation:
                prompt += "\n\nPrevious observations:\n" + conversation[-3000:]
            res_g = client.generate([prompt], n=1, temperature=0.3, top_p=1.0,
                                    top_k=0, max_tokens=512, logprobs=0)
            _, cid, _, _ = _unpack_gen(res_g)
            text = tokenizer.decode(cid[0], skip_special_tokens=True)
            lines = [l.strip() for l in text.splitlines()
                      if "(" in l and ")" in l and "." in l
                      and not any(bad in l.lower() for bad in ("for example", "answer:", "call:", "example:", "do not", "only output"))][:5]
            if not lines:
                turn_log.append({"turn": turn, "calls": [], "note": "no valid call syntax"})
                if turn >= 2:
                    break
                continue
            for line in lines:
                n_calls += 1
                obs = execute_calls(line)
                turn_log.append({"turn": turn, "call": line, "obs": obs[:200]})
                conversation += f"\nCALL: {line}\nOBS: {obs[:300]}"
                if "completed" in obs and '"completed": true' in obs:
                    completed = True
                    break
            if completed or n_calls >= MAX_API_CALLS:
                break

        # ---- hidden evaluation ----
        try:
            calls_so_far = "\n".join(t.get("call", "") for t in turn_log)
            if calls_so_far:
                http_post("/exec", {"code": calls_so_far})
            eval_out = json.dumps(http_post("/eval", {}))[:400]
        except Exception as e:
            eval_out = f"EVAL_HTTP_ERROR: {e}"
        try:
            http_post("/reset", {})
        except Exception:
            pass

        report = {"run_id": f"m5-frozen-s0", "task_id": task_meta["task_id"],
                  "turns": turn_log, "n_calls": n_calls, "completed": completed,
                  "evaluation": eval_out, "parallel_with": "GRPO-Guard-idle"}
        (OUT_DIR / "run_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"task done: completed={completed} calls={n_calls} eval={eval_out[:120]}")
        return 0
    finally:
        stop_server(server, args.port)


if __name__ == "__main__":
    sys.exit(main())

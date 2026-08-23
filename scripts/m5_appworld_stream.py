"""M5: AppWorld prequential stream (design doc Block 1).

Runs a task stream from the dev role (first K tasks) through the persistent
AppWorld exec server. Variants:
  frozen : no update (floor)
Both record first-attempt hidden scores prequentially (Y_pre before any
update); frozen never updates. Extends the closure run to a stream.

Method-level note: the 4B model emits prose on complex prompts; call parsing
keeps only lines matching the apis.* call grammar, with a re-prompt fallback.
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
APWORLD_SERVER_PORT = 8700
MAX_TURNS = 8
MAX_API_CALLS = 20

SYSTEM_PROMPT = """You are an assistant that completes tasks by calling application APIs.
Call format: apis.app_name.function_name(arg1="value1", arg2="value2")
Example (real call): apis.spotify.search_playlists(query="top r&b playlists")
Only output REAL API calls for THIS task. Never repeat this instruction or examples.
Available apps and their API docs:
{api_docs}
Task: {instruction}
Return ONLY API calls, one per line."""


def log(msg: str) -> None:
    print(f"[m5s] {msg}", flush=True)


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
    ap.add_argument("--variant", choices=["frozen", "naive"], default="frozen")
    ap.add_argument("--n-tasks", type=int, default=3)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--port", type=int, default=8230)
    ap.add_argument("--group-port", type=int, default=53200)
    args = ap.parse_args()

    import urllib.request
    import torch
    from transformers import AutoModelForCausalLM as _T_LLM
    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient
    patch_device_normalization()

    OUT_DIR = OUT_ROOT / f"{args.variant}_stream"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aw_server = subprocess.Popen(
        ["/root/autodl-tmp/appworld-venv/bin/python",
         "/root/autodl-tmp/agent-ttrl/scripts/appworld_server.py"],
        env={**os.environ, "APWORLD_SERVER_PORT": str(APWORLD_SERVER_PORT)},
        start_new_session=True)
    time.sleep(10)

    def http_post(path: str, payload: dict) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{APWORLD_SERVER_PORT}{path}",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    # load task list + docs via the appworld-venv helper (one-shot)
    helper = r'''
import json, sys, os
sys.path.insert(0, "/root/autodl-tmp/appworld/src")
os.environ["APPWORLD_ROOT"] = "/root/autodl-tmp/appworld"
from appworld.task import Task, load_task_ids
from appworld.api_docs import prepare_api_docs
ids = load_task_ids("dev")
out = []
for tid in ids[:6]:
    try:
        t = Task.load(tid)
        apps = getattr(t, "allowed_apps", None) or []
        docs = {}
        for a in apps[:4]:
            try:
                docs[a] = prepare_api_docs(a, format="function_calling")
            except Exception as e:
                docs[a] = {"error": str(e)[:80]}
        out.append({"id": tid, "instruction": t.instruction, "allowed_apps": apps, "docs": docs})
    except Exception as e:
        out.append({"id": tid, "error": str(e)[:200]})
print(json.dumps(out))
'''
    hp = OUT_DIR / "tasks_helper.py"
    hp.write_text(helper, encoding="utf-8")
    r = subprocess.run(["/root/autodl-tmp/appworld-venv/bin/python", str(hp)],
                       capture_output=True, text=True, cwd="/root/autodl-tmp/appworld",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8",
                            "APPWORLD_ROOT": "/root/autodl-tmp/appworld"})
    if r.returncode != 0:
        print("helper failed:", r.stderr[-1500:], flush=True)
        return 2
    tasks = json.loads(r.stdout.strip().splitlines()[-1])
    log(f"loaded {len(tasks)} dev tasks")

    server = start_server(OUT_DIR / "vllm_server.log", args.port)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{args.port}", group_port=args.group_port,
                            connection_timeout=300)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        stream_log = []
        for t_idx in range(args.start_idx, min(args.start_idx + args.n_tasks, len(tasks))):
            task = tasks[t_idx]
            if "error" in task:
                stream_log.append({"task": t_idx, "task_id": task["id"], "y_pre": 0.0,
                                   "error": task["error"]})
                continue
            docs_text = json.dumps(task["docs"], indent=1)[:4000]
            for _try in range(5):
                try:
                    http_post("/init", {"task_id": task["id"]})
                    break
                except Exception:
                    if _try == 4:
                        raise
                    time.sleep(10)

            # ---- first attempt (prequential) ----
            prompt = SYSTEM_PROMPT.format(api_docs=docs_text, instruction=task["instruction"])
            conversation = ""
            n_calls = 0
            completed = False
            turn_log = []
            for turn in range(MAX_TURNS):
                p = prompt
                if conversation:
                    p += "\n\nPrevious observations:\n" + conversation[-2500:]
                res_g = client.generate([p], n=1, temperature=0.3, top_p=1.0,
                                        top_k=0, max_tokens=512, logprobs=0)
                _, cid, _, _ = _unpack_gen(res_g)
                text = tokenizer.decode(cid[0], skip_special_tokens=True)
                lines = [l.strip() for l in text.splitlines()
                         if "apis." in l and "(" in l and ")" in l
                         and not any(b in l.lower() for b in ("example", "call:", "only output"))][:4]
                if not lines:
                    if turn >= 3:
                        break
                    continue
                for line in lines:
                    n_calls += 1
                    try:
                        out = http_post("/exec", {"code": line})
                        obs = json.dumps(out)[:300]
                        if out.get("completed"):
                            completed = True
                    except Exception as e:
                        obs = f"EXEC_HTTP_ERROR: {e}"
                    turn_log.append({"turn": turn, "call": line, "obs": obs})
                    conversation += f"\nCALL: {line}\nOBS: {obs[:200]}"
                if completed or n_calls >= MAX_API_CALLS:
                    break

            try:
                if turn_log:
                    calls_so_far = "\n".join(t.get("call", "") for t in turn_log)
                    try:
                        http_post("/exec", {"code": calls_so_far})
                    except Exception as e:
                        log(f"  replay exec failed (non-fatal): {e}")
                eval_out = http_post("/eval", {})
                y_pre = 1.0 if eval_out.get("success") else 0.0
            except Exception as e:
                eval_out = {"error": str(e)[:200]}
                y_pre = 0.0
            try:
                http_post("/reset", {})
            except Exception:
                pass

            # ---- naive variant: LoRA-GRPO update from accessible call evidence ----
            update_info = {"updated": False}
            if args.variant == "naive":
                import torch as _T
                from peft import LoraConfig, get_peft_model
                if "model" not in dir():
                    _base = _T_LLM.from_pretrained(MODEL_PATH, torch_dtype=_T.bfloat16,
                                                   device_map="cuda:0")
                    _base.eval()
                    _lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "gate_proj", "up_proj", "down_proj"],
                                           task_type="CAUSAL_LM")
                    model = get_peft_model(_base, _lora_cfg)
                    model.train()
                    optimizer = _T.optim.AdamW(model.parameters(), lr=5e-6)
                    _T_LLM_REF = _T
                # sample 4 rollouts; reward = fraction of calls that executed OK
                gens = []
                for _g in range(4):
                    res_g = client.generate([prompt], n=1, temperature=1.0, top_p=1.0,
                                            top_k=0, max_tokens=512, logprobs=0)
                    _, cid_g, _, _ = _unpack_gen(res_g)
                    text_g = tokenizer.decode(cid_g[0], skip_special_tokens=True)
                    calls_g = [l.strip() for l in text_g.splitlines()
                               if "apis." in l and "(" in l and ")" in l][:4]
                    ok = 0
                    for c in calls_g:
                        try:
                            out = http_post("/exec", {"code": c})
                            if out.get("ok") and "failed" not in str(out.get("output", ""))[:60]:
                                ok += 1
                        except Exception:
                            pass
                    util = ok / max(1, len(calls_g))
                    gens.append({"cid": cid_g[0], "u": util})
                utils = [g["u"] for g in gens]
                import numpy as _np
                advs = ( _np.array(utils) - _np.mean(utils)) / (_np.std(utils) + 1e-3)
                # update: one clipped-GRPO step over the sampled completions
                model.train()
                optimizer.zero_grad()
                total = 0.0
                for g, a in zip(gens, advs):
                    if a == 0.0:
                        continue
                    ids = _T.tensor([prompt_ids + g["cid"]], device="cuda:0")
                    mask = _T.zeros_like(ids)
                    mask[0, len(prompt_ids):] = 1.0
                    out = model(input_ids=ids)
                    logp = _T.log_softmax(out.logits, dim=-1)
                    shift_logp = logp[:, :-1, :]
                    shift_ids = ids[:, 1:]
                    shift_mask = mask[:, 1:]
                    tok_logp = shift_logp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
                    masked = (tok_logp * shift_mask).sum()
                    n = shift_mask.sum().clamp(min=1.0)
                    total += -float(a) * masked / n
                if total != 0.0:
                    total.backward()
                    _T.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    update_info = {"updated": True, "n_rows": 4, "adv": [round(float(a), 3) for a in advs]}
            stream_log.append({"task": t_idx, "task_id": task["id"], "y_pre": y_pre,
                               "completed": completed, "calls": n_calls,
                               "eval": eval_out, "turns": turn_log, **update_info})
            log(f"task {task['id']}: y_pre={y_pre} calls={n_calls} {update_info}")

        aupc = sum(s["y_pre"] for s in stream_log) / max(1, len(stream_log))
        report = {"run_id": f"m5-{args.variant}-stream", "variant": args.variant,
                  "aupc_prequential": round(aupc, 4), "tasks": stream_log,
                  "parallel_with": "GRPO-Guard-idle"}
        (OUT_DIR / "run_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"AUPC_prequential={aupc:.4f} over {len(stream_log)} tasks")
        return 0
    finally:
        stop_server(server, args.port)
        try:
            http_post("/reset", {})
        except Exception:
            pass
        aw_server.kill()


if __name__ == "__main__":
    sys.exit(main())

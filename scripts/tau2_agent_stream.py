"""M6: tau2 retail prequential stream (frozen baseline; design doc §9.3).

LLM agent calls tau2 retail tools directly (func(arg="v")) via the persistent
tau2 exec server; hidden scoring = evaluation_criteria match (E_hard-visible
in our adapter; R_hidden for the paper's protocol it is the official
evaluator proxy). Prequential first-attempt scores per task.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT_ROOT = Path(os.environ.get("ATTRL_M6_OUT", "/root/autodl-tmp/agent-ttrl/artifacts/m6"))
TAU2_SERVER_PORT = 8800
MAX_TURNS = 6

SYSTEM_PROMPT = """You are a customer service assistant for a retail company.
Available tools (call them directly, one per line):
{tools}
Policy:
{policy}
Task: {instruction}
Examples of correct calls:
find_user_id_by_name_zip(first_name="Yusuf", last_name="Rossi", zip="19122")
search_for_product(keywords="mechanical keyboard", limit=10)
Return ONLY tool calls, one per line, keyword arguments in double quotes."""


def log(msg: str) -> None:
    print(f"[m6] {msg}", flush=True)


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


def start_server(server_log: Path, port: int, model_path: str) -> subprocess.Popen:
    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [trl_bin, "vllm-serve", "--model", model_path, "--port", str(port),
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
    ap.add_argument("--model", default=os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B"))
    ap.add_argument("--variant", choices=["frozen", "naive", "egc"], default="frozen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--n-tasks", type=int, default=4)
    ap.add_argument("--port", type=int, default=8240)
    ap.add_argument("--group-port", type=int, default=53300)
    args = ap.parse_args()
    model_path = args.model

    import urllib.request
    import torch
    from transformers import AutoModelForCausalLM as _T_LLM
    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient
    patch_device_normalization()

    OUT_DIR = OUT_ROOT / f"{args.variant}_stream_{Path(model_path).name}_s{args.seed}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t2_server = subprocess.Popen(
        ["/root/autodl-tmp/appworld-venv/bin/python",
         "/root/autodl-tmp/agent-ttrl/scripts/tau2_server.py"],
        env={**os.environ, "TAU2_SERVER_PORT": str(TAU2_SERVER_PORT)},
        start_new_session=True)
    time.sleep(10)

    def http_post(path: str, payload: dict) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{TAU2_SERVER_PORT}{path}",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    # task list + tools via helper
    helper = r'''
import json, sys, os
sys.path.insert(0, "/root/autodl-tmp/tau2-bench/src")
from tau2.domains.retail.environment import get_environment, get_tasks
env = get_environment()
tools = env.get_tools_description("assistant")
tasks = get_tasks("base")[:20]
out = []
for t in tasks:
    out.append({"id": t.id, "instruction": t.user_scenario.instructions.task_instructions,
                "known": t.user_scenario.instructions.known_info,
                "policy": None})
print(json.dumps({"tools": str(tools)[:4000], "tasks": out}))
'''
    hp = OUT_DIR / "tau2_helper.py"
    hp.write_text(helper, encoding="utf-8")
    r = subprocess.run(["/root/autodl-tmp/appworld-venv/bin/python", str(hp)],
                       capture_output=True, text=True, cwd="/root/autodl-tmp/tau2-bench",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        print("helper failed:", r.stderr[-1500:], flush=True)
        return 2
    meta = json.loads(r.stdout.strip().splitlines()[-1])
    tools_text = meta["tools"]
    tasks = meta["tasks"]
    log(f"loaded {len(tasks)} tau2 retail tasks; tools desc {len(tools_text)} chars")

    server = start_server(OUT_DIR / "vllm_server.log", args.port, model_path)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{args.port}", group_port=args.group_port,
                            connection_timeout=300)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        import random as _rnd
        _rnd.seed(args.seed)
        task_order = list(range(min(args.n_tasks, len(tasks))))
        _rnd.shuffle(task_order)
        stream_log = []
        for t_idx in task_order:
            task = tasks[t_idx]
            for _try in range(5):
                try:
                    http_post("/init", {"task_id": task["id"]})
                    break
                except Exception:
                    if _try == 4:
                        raise
                    time.sleep(8)
            prompt = SYSTEM_PROMPT.format(tools=tools_text, policy="Follow store policy.",
                                          instruction=task["instruction"])
            conversation = ""
            turn_log = []
            for turn in range(MAX_TURNS):
                p = prompt
                if conversation:
                    p += "\n\nPrevious observations:\n" + conversation[-2500:]
                res_g = client.generate([p], n=1, temperature=0.3, top_p=1.0,
                                        top_k=0, max_tokens=256, logprobs=0)
                pids_first, cid, _, _ = _unpack_gen(res_g)
                prompt_ids = pids_first[0] if turn == 0 else prompt_ids
                text = tokenizer.decode(cid[0], skip_special_tokens=True)
                import re as _re
                lines = []
                for m in _re.finditer(r"[a-zA-Z_][a-zA-Z0-9_]*\([^\n]{0,160}\)", text):
                    frag = m.group(0).strip()
                    if frag not in lines:
                        lines.append(frag)
                lines = lines[:4]
                if not lines:
                    if turn >= 3:
                        break
                    continue
                for line in lines:
                    try:
                        out = http_post("/exec", {"code": line})
                        obs = json.dumps(out)[:250]
                    except Exception as e:
                        obs = f"EXEC_HTTP_ERROR: {e}"
                    turn_log.append({"turn": turn, "call": line, "obs": obs})
                    conversation += f"\nCALL: {line}\nOBS: {obs[:200]}"
            try:
                eval_out = http_post("/eval", {})
                y_pre = eval_out.get("pass_pct", 0.0) / 100.0  # partial-match score (more informative than binary)
            except Exception as e:
                eval_out = {"error": str(e)[:150]}
                y_pre = 0.0
            update_info = {"updated": False}
            if args.variant in ("naive", "egc") and "model" not in dir():
                import torch as _T
                from peft import LoraConfig, get_peft_model
                _base = _T_LLM.from_pretrained(model_path, torch_dtype=_T.bfloat16, device_map="cuda:0")
                _base.eval()
                _lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                       "gate_proj", "up_proj", "down_proj"],
                                       task_type="CAUSAL_LM")
                model = get_peft_model(_base, _lora_cfg)
                model.train()
                optimizer = _T.optim.AdamW(model.parameters(), lr=args.lr)
                import numpy as _np
            if args.variant in ("naive", "egc"):
                # sample N rollouts; utility = fraction of calls that executed OK
                gens = []
                for _g in range(args.n_rollouts):
                    res_g = client.generate([prompt], n=1, temperature=0.9, top_p=1.0,
                                            top_k=0, max_tokens=256, logprobs=0)
                    _, cid_g, _, _ = _unpack_gen(res_g)
                    text_g = tokenizer.decode(cid_g[0], skip_special_tokens=True)
                    import re as _re
                    call_frags = [m.group(0) for m in _re.finditer(
                        r"[a-zA-Z_][a-zA-Z0-9_]*\([^\n]{0,160}\)", text_g)][:4]
                    ok = 0
                    for frag in call_frags:
                        try:
                            out = http_post("/exec", {"code": frag})
                            if out.get("ok") and out.get("n_ok", 0) > 0:
                                ok += 1
                        except Exception:
                            pass
                    util = ok / max(1, len(call_frags))
                    gens.append({"cid": cid_g[0], "u": util})
                if gens and max(g["u"] for g in gens) > min(g["u"] for g in gens):
                    utils = _np.array([g["u"] for g in gens])
                    advs = (utils - utils.mean()) / (utils.std() + 1e-3)
                    if args.variant == "egc":
                        # reliability gate: zero out non-significant credits (|z| < 0.5)
                        advs = _np.where(_np.abs(advs) >= 0.5, advs, 0.0)
                    model.train()
                    # per-sequence forward/backward (gradient accumulation) to
                    # bound peak memory for 7B-scale batches
                    for _s in range(args.steps):
                        optimizer.zero_grad()
                        n_used = 0
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
                            loss = -float(a) * masked / n
                            loss.backward()
                            n_used += 1
                            del ids, mask, out, logp, shift_logp, shift_ids, shift_mask, tok_logp, masked
                        if n_used > 0:
                            _T.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        _T.cuda.empty_cache()
                    update_info = {"updated": n_used > 0, "n_rollouts": args.n_rollouts,
                                   "steps": args.steps, "lr": args.lr,
                                   "utils": [round(float(u), 3) for u in utils]}
            try:
                http_post("/reset", {})
            except Exception:
                pass
            stream_log.append({"task": t_idx, "task_id": task["id"], "y_pre": y_pre,
                               "eval": eval_out, "turns": turn_log, **update_info})
            log(f"task {task['id']}: y_pre={y_pre} eval={str(eval_out)[:110]}")

        aupc = sum(s["y_pre"] for s in stream_log) / max(1, len(stream_log))
        report = {"run_id": f"m6-{args.variant}-stream", "variant": args.variant,
                  "seed": args.seed, "n_tasks": len(stream_log),
                  "aupc_prequential": round(aupc, 4), "metric": "pass_pct_partial", "tasks": stream_log,
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
        t2_server.kill()


if __name__ == "__main__":
    sys.exit(main())

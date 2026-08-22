"""Minimal egc reproduction: step-by-step, print+flush each step, find the kill point."""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
REPO_DIR = Path("/root/autodl-tmp/grpo-guard-src")
ATTRL_DIR = Path("/root/autodl-tmp/agent-ttrl")
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(ATTRL_DIR / "src"))
PORT = 8060
GPORT = 51600
TRAINER_GPU = 0
SERVER_GPU = 1

BASE_PROMPT = """You are an order assistant. Available tools (JSON list of {"tool": ..., "call": {...}}):
- reserve_item {item_key, order_id}
- create_order {order_id}
- charge {order_id, user_id, amount_cents}
- ship {order_id, user_id, address}
- complete_task {}
Task: user u1 wants item sku:a shipped to address addr-1. Call the tools in order.
Return ONLY the JSON list of tool calls."""

BRANCH_PROMPT = """You are an order assistant. Available tools (JSON list of {"tool": ..., "call": {...}}):
- reserve_item {item_key, order_id}
- create_order {order_id}
- charge {order_id, user_id, amount_cents}
- ship {order_id, user_id, address}
- cancel_order {order_id}
- complete_task {}
State: order o1 for item sku:a is CREATED but NOT paid. User u1 (address addr-1) waits.
Complete the order. Return ONLY the JSON list of tool calls."""


def step(tag: str) -> None:
    print(f"[diag2] STEP {tag}", flush=True)
    t0 = time.time()
    return t0


def main() -> int:
    t = step("import torch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient
    print(f"[diag2] imports ok in {time.time()-t:.1f}s", flush=True)

    t = step("patch+server")
    import trl
    import vllm
    from trl.generation.vllm_client import VLLMClient as VC
    _orig = VC.init_communicator
    def _norm(self, device, *a, **kw):
        if isinstance(device, torch.device) and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        return _orig(self, device, *a, **kw)
    VC.init_communicator = _norm
    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [trl_bin, "vllm-serve", "--model", MODEL_PATH, "--port", str(PORT),
         "--gpu-memory-utilization", "0.4", "--max-model-len", "2048"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(SERVER_GPU)},
        stdout=open("/root/autodl-tmp/agent-ttrl/artifacts/m3/diag2_vllm.log", "w"),
        stderr=subprocess.STDOUT, start_new_session=True)
    healthy = False
    for _ in range(180):
        time.sleep(2)
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    healthy = True
                    break
        except Exception:
            pass
        if proc.poll() is not None:
            print("[diag2] SERVER DIED", flush=True)
            return 2
    print(f"[diag2] server healthy in {time.time()-t:.1f}s", flush=True)

    t = step("client")
    client = VLLMClient(base_url=f"http://127.0.0.1:{PORT}", group_port=GPORT, connection_timeout=300)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[diag2] client+tokenizer ok in {time.time()-t:.1f}s", flush=True)

    t = step("generate BASE")
    res = client.generate([BASE_PROMPT], n=1, temperature=1.0, top_p=1.0,
                          top_k=0, max_tokens=128, logprobs=0)
    pid = res["prompt_ids"][0]
    cid = res["completion_ids"][0]
    text = tokenizer.decode(cid, skip_special_tokens=True)
    print(f"[diag2] BASE generate ok in {time.time()-t:.1f}s, completion={len(cid)} tok, text[:40]={text[:40]!r}", flush=True)

    t = step("generate BRANCH x1")
    res2 = client.generate([BRANCH_PROMPT], n=1, temperature=1.0, top_p=1.0,
                           top_k=0, max_tokens=128, logprobs=0)
    cid2 = res2["completion_ids"][0]
    print(f"[diag2] BRANCH generate ok in {time.time()-t:.1f}s, completion={len(cid2)} tok", flush=True)

    t = step("load model")
    torch.cuda.set_device(TRAINER_GPU)
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
                                                device_map=f"cuda:{TRAINER_GPU}")
    base.eval()
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    print(f"[diag2] model+peft ok in {time.time()-t:.1f}s", flush=True)

    t = step("paired_credit")
    from agent_ttrl.credit.paired_credit import paired_credit
    U = np.array([[0.9, 0.9], [0.6, 0.6], [0.1, 0.1], [0.0, 0.0]], dtype=float)
    verdict = paired_credit(U)
    print(f"[diag2] paired_credit ok: {verdict.status} in {time.time()-t:.1f}s", flush=True)

    t = step("native step")
    seqs = [{"cid": cid}, {"cid": cid2}, {"cid": cid}, {"cid": cid2}]
    advs = [0.5, 0.2, -0.3, -0.4]
    model.train()
    optimizer.zero_grad()
    total = 0.0
    for seq, adv in zip(seqs, advs):
        ids = torch.tensor([pid + seq["cid"]], device=f"cuda:{TRAINER_GPU}")
        mask = torch.zeros_like(ids)
        mask[0, len(pid):] = 1.0
        out = model(input_ids=ids)
        logp = torch.log_softmax(out.logits, dim=-1)
        shift_logp = logp[:, :-1, :]
        shift_ids = ids[:, 1:]
        shift_mask = mask[:, 1:]
        tok_logp = shift_logp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        masked = (tok_logp * shift_mask).sum()
        n = shift_mask.sum().clamp(min=1.0)
        total += -float(adv) * masked / n
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"[diag2] native step ok in {time.time()-t:.1f}s", flush=True)

    print("[diag2] ALL STEPS OK — no kill in egc path", flush=True)
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

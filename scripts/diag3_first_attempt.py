"""Diagnose why egc first attempts all fail: print raw completions."""
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
PORT = 8080
GPORT = 51800

BASE_PROMPT = """You are an order assistant. Available tools (JSON list of {"tool": ..., "call": {...}}):
- reserve_item {item_key, order_id}
- create_order {order_id}
- charge {order_id, user_id, amount_cents}
- ship {order_id, user_id, address}
- complete_task {}
Task: user u1 wants item sku:a shipped to address addr-1. Call the tools in order.
Return ONLY the JSON list of tool calls."""


def main() -> int:
    import torch
    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    _orig = VLLMClient.init_communicator
    def _norm(self, device, *a, **kw):
        if isinstance(device, torch.device) and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        return _orig(self, device, *a, **kw)
    VLLMClient.init_communicator = _norm

    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [trl_bin, "vllm-serve", "--model", MODEL_PATH, "--port", str(PORT),
         "--gpu-memory-utilization", "0.4", "--max-model-len", "2048"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
        stdout=open("/root/autodl-tmp/agent-ttrl/artifacts/m3/diag3_vllm.log", "w"),
        stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(180):
            time.sleep(2)
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
        else:
            print("SERVER NOT HEALTHY", flush=True)
            return 2
        print("server healthy", flush=True)
        client = VLLMClient(base_url=f"http://127.0.0.1:{PORT}", group_port=GPORT, connection_timeout=300)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        for i in range(5):
            res = client.generate([BASE_PROMPT], n=1, temperature=1.0, top_p=1.0,
                                  top_k=0, max_tokens=128, logprobs=0)
            cid = res["completion_ids"][0]
            text = tokenizer.decode(cid, skip_special_tokens=True)
            print(f"[diag3] sample {i}: len={len(cid)} text={text[:150]!r}", flush=True)
        print("[diag3] done", flush=True)
        return 0
    finally:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

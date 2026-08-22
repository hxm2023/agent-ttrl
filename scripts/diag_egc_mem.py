"""Diagnostic: locate the GPU0 memory allocation that precedes the SIGKILL in egc runs."""
import os
import sys
import time
from pathlib import Path

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
REPO_DIR = Path("/root/autodl-tmp/grpo-guard-src")
sys.path.insert(0, str(REPO_DIR / "src"))


def mem(tag: str) -> None:
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().replace("\n", " | ")
    print(f"[diag] {tag}: {out}", flush=True)


def main() -> int:
    print("[diag] start", flush=True)
    mem("start")
    import torch
    print(f"[diag] torch {torch.__version__} cuda {torch.version.cuda}", flush=True)
    mem("after torch import")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mem("after tokenizer")

    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    base.eval()
    mem("after model load")

    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    mem("after peft+optimizer")
    print(f"[diag] trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)

    # forward pass on a small batch (like native_grpo_step)
    import torch as T
    ids = tokenizer("reserve sku:a", return_tensors="pt").input_ids.to("cuda:0")
    out = model(input_ids=ids)
    print(f"[diag] forward logits {tuple(out.logits.shape)}", flush=True)
    mem("after forward")

    # 2 more forwards to simulate multi-row batches
    for i in range(2):
        ids = tokenizer(f"charge order o{i}", return_tensors="pt").input_ids.to("cuda:0")
        out = model(input_ids=ids)
        mem(f"after forward {i+2}")
    print("[diag] all steps ok — no kill here", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

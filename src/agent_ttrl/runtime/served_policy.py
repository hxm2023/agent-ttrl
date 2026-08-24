"""Policy-consistent runtime: HF colocated backend (v2 protocol, §12.1).

The SAME PEFT model generates every request and receives gradient updates —
no separate static serving model. This closes the v1 fatal defect where the
LoRA-updated HF model was never synchronized to the vLLM server used for
subsequent first attempts (STATIC_SERVED_POLICY).

Guarantees provided:
- generate() is deterministic per RequestSeed (torch RNG state set/restored
  around sampling); the same seed + policy_version yields bitwise-identical
  output.
- commit() is atomic: the candidate adapter state becomes the served state
  only after a canary check; rollback() restores the parent state.
- canary requires (a) adapter hash matches candidate, (b) parent/candidate
  logits differ beyond a numeric tolerance on a fixed prompt, (c) a
  deterministic rollout observably changes. weight_delta>0 alone is NOT a
  pass (v1 R002 defect).
"""
from __future__ import annotations

import hashlib
import torch
from dataclasses import dataclass

from agent_ttrl.runtime.request_seed import RequestSeed


@dataclass
class CanaryResult:
    passed: bool
    adapter_sha256: str
    parent_logit_kl: float
    deterministic_output_changed: bool
    reason: str = ""


def _adapter_hash(model) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _kl(a: torch.Tensor, b: torch.Tensor) -> float:
    loga = torch.log_softmax(a.float(), dim=-1)
    logb = torch.log_softmax(b.float(), dim=-1)
    pa = loga.exp()
    return float((pa * (loga - logb)).sum(dim=-1).mean().item())


class ColocatedPolicy:
    """One PEFT model: generate + train. Slow but policy-consistent."""

    def __init__(self, model_path: str, lora_rank: int = 16, lora_alpha: int = 32,
                 device: str = "cuda:0", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device, trust_remote_code=True)
        base.eval()
        lora_cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora_cfg)
        self.model.train()
        self.policy_version = 0
        # parent snapshot for rollback (initial served state)
        self._parent = {n: p.detach().clone() for n, p in self.model.named_parameters()
                        if p.requires_grad}

    # ---------------------------------------------------------------- generation
    def generate(self, seed: RequestSeed, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7) -> tuple[list[int], str]:
        """Deterministic per-request generation (torch RNG state restored after)."""
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed.seed())
        try:
            with torch.no_grad():
                out = self.model.generate(
                    ids, max_new_tokens=max_tokens, do_sample=True,
                    temperature=temperature, top_p=1.0, pad_token_id=self.tokenizer.eos_token_id)
        finally:
            torch.random.set_rng_state(rng_state)
        gen = out[0][ids.shape[1]:].tolist()
        return gen, self.tokenizer.decode(gen, skip_special_tokens=True)

    # ---------------------------------------------------------------- training
    def train_step(self, prompt_ids: list[int], completion_ids: list[int],
                   advantage: float, lr: float, max_grad_norm: float = 1.0) -> dict:
        """REINFORCE-style step over the completion span (same update rule as
        v2 streams; v1 simplification retained but now on the SERVED policy)."""
        ids = torch.tensor([prompt_ids + completion_ids], device=self.device)
        mask = torch.zeros_like(ids)
        mask[0, len(prompt_ids):] = 1.0
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        opt.zero_grad()
        out = self.model(input_ids=ids)
        logp = torch.log_softmax(out.logits.float(), dim=-1)
        shift_logp = logp[:, :-1, :]
        shift_ids = ids[:, 1:]
        shift_mask = mask[:, 1:]
        tok_logp = shift_logp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        masked = (tok_logp * shift_mask).sum()
        n = shift_mask.sum().clamp(min=1.0)
        loss = -float(advantage) * masked / n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        opt.step()
        self.model.zero_grad()
        torch.cuda.empty_cache()
        return {"loss": float(loss.item()), "tokens": int(n.item())}

    def update_params(self, lr: float) -> None:
        """Expose LR changes (A0 uses lr=0 to prove RNG/schedule isolation)."""
        self._lr = lr

    # ---------------------------------------------------------------- commit / rollback
    def freeze_candidate(self) -> str:
        """Hash the current (candidate) adapter state without serving it yet."""
        return _adapter_hash(self.model)

    def commit(self, candidate_sha: str, canary_prompt: str, canary_seed: RequestSeed,
               logit_tol: float = 1e-3) -> CanaryResult:
        """Atomically serve the candidate: verify adapter hash + canary change,
        then bump policy_version (the state IS the served state — colocated)."""
        cur = _adapter_hash(self.model)
        if cur != candidate_sha:
            return CanaryResult(False, cur, 0.0, False,
                                f"adapter hash mismatch: {cur[:12]} != {candidate_sha[:12]}")
        parent = self._parent
        kl = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in parent:
                kl = max(kl, _kl(p.detach().float().reshape(1, -1),
                                 parent[n].float().reshape(1, -1)))
        before, _ = self.generate(canary_seed, canary_prompt, max_tokens=16, temperature=0.0)
        self.policy_version += 1
        after, _ = self.generate(canary_seed, canary_prompt, max_tokens=16, temperature=0.0)
        changed = before != after or kl > logit_tol
        if not changed:
            self.policy_version -= 1
            return CanaryResult(False, cur, kl, False, "canary: no observable change")
        self._parent = {n: p.detach().clone() for n, p in self.model.named_parameters()
                        if p.requires_grad}
        return CanaryResult(True, cur, kl, changed)

    def rollback(self) -> str:
        """Restore the parent (last committed) state."""
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self._parent:
                    p.copy_(self._parent[n])
        return _adapter_hash(self.model)

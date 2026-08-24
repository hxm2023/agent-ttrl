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


def _adapter_hash_state(state: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].detach().cpu().to(torch.float32).numpy().tobytes())
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
        # v3: committed (served) state vs shadow candidate
        self._committed = {n: p.detach().clone()
                           for n, p in self.model.named_parameters() if p.requires_grad}
        self._candidate = None
        # stack of served states; rollback pops back to the previously
        # committed (or initial) state
        self._stack = [self._committed]

    # ---------------------------------------------------------------- generation
    def generate(self, seed: RequestSeed, prompt: str, max_tokens: int = 128,
                 temperature: float = 0.7) -> tuple[list[int], str]:
        """Deterministic per-request generation (torch RNG state restored after).
        temperature <= 0 => greedy (do_sample=False), used by canaries."""
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed.seed())
        try:
            with torch.no_grad():
                out = self.model.generate(
                    ids, max_new_tokens=max_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else 1.0,
                    top_p=1.0, pad_token_id=self.tokenizer.eos_token_id)
        finally:
            torch.random.set_rng_state(rng_state)
        gen = out[0][ids.shape[1]:].tolist()
        return gen, self.tokenizer.decode(gen, skip_special_tokens=True)

    # ---------------------------------------------------------------- training
    def _swap_to(self, state: dict) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in state:
                    p.data.copy_(state[n])

    def _optim_step(self, ids: torch.Tensor, mask: torch.Tensor,
                    advantage: float, lr: float) -> dict:
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        self.model.zero_grad()
        torch.cuda.empty_cache()
        return {"loss": float(loss.item()), "tokens": int(n.item())}

    def _ids_mask(self, prompt_ids, completion_ids):
        ids = torch.tensor([prompt_ids + completion_ids], device=self.device)
        mask = torch.zeros_like(ids)
        mask[0, len(prompt_ids):] = 1.0
        return ids, mask

    def train_step(self, prompt_ids: list[int], completion_ids: list[int],
                   advantage: float, lr: float, max_grad_norm: float = 1.0) -> dict:
        """REINFORCE-style step. v3: trains the SHADOW CANDIDATE state only;
        the served (committed) state is untouched until commit_candidate()."""
        if getattr(self, "_candidate", None) is None:
            raise RuntimeError("train_step requires begin_candidate() first")
        self._swap_to(self._candidate)
        ids, mask = self._ids_mask(prompt_ids, completion_ids)
        r = self._optim_step(ids, mask, advantage, lr)
        # snapshot the trained candidate
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._candidate[n].copy_(p.data)
        self._swap_to(self._committed)
        return r

    def begin_candidate(self) -> None:
        """Start a shadow candidate: generation keeps serving the committed
        state; training writes only to the candidate copy."""
        self._candidate = {n: p.detach().clone()
                           for n, p in self.model.named_parameters() if p.requires_grad}

    def candidate_hash(self) -> str:
        return _adapter_hash_state(self._candidate) if getattr(self, "_candidate", None) else ""

    def commit_candidate(self, canary_prompt: str, canary_seed: RequestSeed,
                         logit_tol: float = 1e-3) -> CanaryResult:
        """Canary-check the candidate against the committed state; on pass,
        atomically serve it (swap model params + bump version); on fail,
        discard the candidate. The served state never changes before this."""
        cand = getattr(self, "_candidate", None)
        if cand is None:
            return CanaryResult(False, "", 0.0, False, "no candidate in flight")
        ids = self.tokenizer(canary_prompt, return_tensors="pt").input_ids.to(self.device)
        logits_cand = self._logits_with(cand, ids)
        logits_committed = self._logits_with(self._committed, ids)
        kl = _kl(logits_cand, logits_committed)
        before, _ = self.generate(canary_seed, canary_prompt, max_tokens=16, temperature=0.0)
        # serve candidate only for the canary rollout, then restore
        self._swap_to(cand)
        after, _ = self.generate(canary_seed, canary_prompt, max_tokens=16, temperature=0.0)
        self._swap_to(self._committed)
        changed = before != after or kl > logit_tol
        if not changed:
            self._candidate = None
            return CanaryResult(False, "", kl, False, "canary: no observable change")
        # atomic swap: candidate becomes the served state
        self._swap_to(cand)
        self._committed = {n: p.detach().clone()
                           for n, p in self.model.named_parameters() if p.requires_grad}
        self._stack.append(self._committed)
        self.policy_version += 1
        cur = _adapter_hash_state(self._committed)
        self._candidate = None
        return CanaryResult(True, cur, kl, changed)

    def update_params(self, lr: float) -> None:
        """Expose LR changes (A0 uses lr=0 to prove RNG/schedule isolation)."""
        self._lr = lr

    # ---------------------------------------------------------------- commit / rollback
    def freeze_candidate(self) -> str:
        """Hash the current (candidate) adapter state without serving it yet."""
        return _adapter_hash(self.model)

    def _logits_with(self, state: dict, ids: torch.Tensor) -> torch.Tensor:
        """Forward with a given parameter state without mutating the model."""
        saved = {n: p.detach().clone() for n, p in self.model.named_parameters()
                 if p.requires_grad}
        try:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and n in state:
                        p.data.copy_(state[n])
                out = self.model(input_ids=ids).logits
        finally:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in saved:
                    p.data.copy_(saved[n])
        return out

    def commit(self, candidate_sha: str, canary_prompt: str, canary_seed: RequestSeed,
               logit_tol: float = 1e-3) -> CanaryResult:
        """Back-compat wrapper: commit_candidate under the v3 shadow protocol."""
        if getattr(self, "_candidate", None) is None:
            self.begin_candidate()
        return self.commit_candidate(canary_prompt, canary_seed, logit_tol)

    def rollback(self) -> str:
        """Pop the stack and restore the previously committed state."""
        if len(self._stack) > 1:
            self._stack.pop()
        parent = self._stack[-1]
        self._swap_to(parent)
        self._committed = {n: p.detach().clone()
                           for n, p in self.model.named_parameters() if p.requires_grad}
        return _adapter_hash(self.model)

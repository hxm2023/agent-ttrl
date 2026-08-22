"""Canonical adapter manifest hash (design doc §18.1, §2.1).

Every candidate/parent adapter is identified by a canonical hash over:
base model revision, adapter hyperparameters, target modules, optimizer state
ref, training input event refs, and policy version. Used for identity binding,
canary tests, and atomic commit manifests.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class AdapterSpec:
    base_sha256: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str] = field(default_factory=list)
    optimizer: str = "adamw"
    learning_rate: float = 5.0e-6
    policy_version: str = "v1"
    training_input_event_refs: list[str] = field(default_factory=list)

    def canonical(self) -> str:
        return json.dumps({
            "base_sha256": self.base_sha256,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": sorted(self.target_modules),
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "policy_version": self.policy_version,
            "training_input_event_refs": sorted(self.training_input_event_refs),
        }, sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()


def same_identity(a: AdapterSpec, b: AdapterSpec) -> bool:
    """Identity equality (used by canary tests to distinguish parent/candidate)."""
    return a.sha256() == b.sha256()

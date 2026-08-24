"""Request-scoped RNG (v2 protocol, §12.2).

Every generation request gets a seed derived from the full request identity:
    seed = H(protocol_hash, stream_seed, task_id, turn_id,
             policy_version, purpose, branch_group, action_id, continuation_id)

Purpose namespace: production_first_attempt / within_task_recovery /
credit_action_proposal / credit_continuation / shadow_gain / shadow_anchor /
canary.

Guarantee: extra branch/rollout generations in an update arm MUST NOT change
the seed of any future production request. This removes the v1
sampling-stream-displacement confound (frozen vs update arms differed only
because update arms consumed extra RNG draws).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

PURPOSES = {
    "production_first_attempt",
    "within_task_recovery",
    "credit_action_proposal",
    "credit_continuation",
    "shadow_gain",
    "shadow_anchor",
    "canary",
}


@dataclass(frozen=True)
class RequestSeed:
    protocol_hash: str
    stream_seed: int
    task_id: str
    turn_id: int
    policy_version: int
    purpose: str
    branch_group: int = 0
    action_id: int = 0
    continuation_id: int = 0

    def __post_init__(self) -> None:
        if self.purpose not in PURPOSES:
            raise ValueError(f"unknown purpose: {self.purpose}")

    def seed(self) -> int:
        payload = "|".join([
            self.protocol_hash, str(self.stream_seed), self.task_id,
            str(self.turn_id), str(self.policy_version), self.purpose,
            str(self.branch_group), str(self.action_id), str(self.continuation_id),
        ])
        return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)

    def namespace(self) -> str:
        return f"p{self.protocol_hash[:8]}_s{self.stream_seed}_t{self.task_id}_u{self.turn_id}_v{self.policy_version}_{self.purpose}"

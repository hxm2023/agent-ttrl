"""Request-scoped RNG (v3 protocol).

Every generation request gets a seed derived from EXOGENOUS identity only:
    seed = H(protocol_hash, stream_seed, task_id, turn_id,
             purpose, branch_group, action_id, continuation_id)

policy_version is recorded in manifests but NEVER enters the seed — this
keeps common random numbers across arms (frozen vs update arms draw the
same production sequence regardless of treatment), fixing the v2
treatment-dependent-seed confound.

Purpose namespace: production_first_attempt / within_task_recovery /
credit_action_proposal / credit_continuation / shadow_gain / shadow_anchor /
canary.

Guarantee: extra branch/rollout generations in an update arm MUST NOT change
the seed of any future production request.
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
    purpose: str
    branch_group: int = 0
    action_id: int = 0
    continuation_id: int = 0
    # policy_version is manifest metadata only (recorded per request, not in
    # the seed) — kept as an explicit field so call sites can log it
    policy_version: int = 0

    def __post_init__(self) -> None:
        if self.purpose not in PURPOSES:
            raise ValueError(f"unknown purpose: {self.purpose}")

    def seed(self) -> int:
        payload = "|".join([
            self.protocol_hash, str(self.stream_seed), self.task_id,
            str(self.turn_id), self.purpose,
            str(self.branch_group), str(self.action_id), str(self.continuation_id),
        ])
        return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)

    def namespace(self) -> str:
        return (f"p{self.protocol_hash[:8]}_s{self.stream_seed}_t{self.task_id}"
                f"_u{self.turn_id}_{self.purpose}_b{self.branch_group}")

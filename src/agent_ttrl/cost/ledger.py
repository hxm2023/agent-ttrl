"""CostLedger: 3-channel hard-cap budget accounting (design doc §10.5/§19.1).

One canonical ledger; each operation is billed exactly once into exactly one
channel; caps are hard; no cross-channel exchange. A ledger-conservation check
detects missed or double billing (fault F12: LEDGER_CONSERVATION).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum


class Channel(str, Enum):
    ENV = "B_env"          # environment transitions / tool calls (production, branch, restore-continuation, shadow/sentinel)
    MODEL = "B_model"      # non-padding generated/scored tokens
    UPDATE = "B_update"    # action tokens entering forward/backward x optimizer steps


@dataclass
class LedgerEvent:
    op_id: str
    channel: Channel
    amount: float          # transitions, tokens, or action-tokens*steps
    scope: str             # production | branch | shadow | sentinel | update
    artifact_ref: str | None = None

    def sha256(self) -> str:
        payload = json.dumps({
            "op_id": self.op_id, "channel": self.channel.value, "amount": self.amount,
            "scope": self.scope, "artifact_ref": self.artifact_ref,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class CostLedger:
    caps: dict[Channel, float]
    events: list[LedgerEvent] = field(default_factory=list)
    _seen_ops: set[str] = field(default_factory=set)

    def bill(self, op_id: str, channel: Channel, amount: float, scope: str,
             artifact_ref: str | None = None) -> None:
        if op_id in self._seen_ops:
            raise ValueError(f"DUPLICATE_LEDGER_EVENT:{op_id}")
        if amount < 0:
            raise ValueError(f"NEGATIVE_BILL:{op_id}")
        self._seen_ops.add(op_id)
        self.events.append(LedgerEvent(op_id, channel, amount, scope, artifact_ref))

    def totals(self) -> dict[str, float]:
        t = {c.value: 0.0 for c in Channel}
        for e in self.events:
            t[e.channel.value] += e.amount
        return t

    def consumption(self) -> dict[str, float]:
        return {c.value: self.totals()[c.value] / self.caps[c] for c in Channel}

    def within_caps(self) -> bool:
        return all(self.totals()[c.value] <= self.caps[c] + 1e-9 for c in Channel)

    def conservation_ok(self, external_tally: dict[str, float]) -> bool:
        """Ledger totals must equal an independent external tally (no missed/double billing)."""
        for c in Channel:
            if abs(self.totals()[c.value] - external_tally.get(c.value, 0.0)) > 1e-9:
                return False
        return True

    def event_log_sha256(self) -> str:
        payload = json.dumps([e.sha256() for e in self.events], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

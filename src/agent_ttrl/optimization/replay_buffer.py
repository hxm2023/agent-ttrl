"""Cross-task evidence replay buffer (v2 §11.2, §13.3).

v1 updated with 1-8 rows per single task at tiny LR — too weak and noisy to
produce measurable transfer. v2 aggregates reliable signed action rows ACROSS
tasks (session-scoped) and applies one small LoRA step per update epoch:

  - rows are bucketed by tool intent (family/workflow), de-duplicated
  - positive and negative signed rows both kept
  - recency weighting + anchor/rehearsal rows (10-20%) against forgetting
  - every row keeps its producer prompt_ids + completion_ids + hash
  - all arms share the same buffer capacity and update-token cap (matched)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class EvidenceRow:
    task_id: str
    tool_intent: str          # bucket key (e.g., "F1_refund")
    prompt_ids: list[int]
    completion_ids: list[int]
    advantage: float          # signed, group-relative
    policy_version: int
    weight: float = 1.0       # recency/anchor weighting
    hash: str = ""

    def __post_init__(self):
        h = hashlib.sha256()
        h.update(self.tool_intent.encode())
        h.update(str(self.prompt_ids[:64]).encode())
        h.update(str(self.completion_ids[:64]).encode())
        h.update(f"{self.advantage:.6f}".encode())
        self.hash = h.hexdigest()[:16]


class ReplayBuffer:
    def __init__(self, capacity: int = 256, anchor_fraction: float = 0.15,
                 recency_gamma: float = 0.95):
        self.capacity = capacity
        self.anchor_fraction = anchor_fraction
        self.recency_gamma = recency_gamma
        self.rows: list[EvidenceRow] = []
        self.anchors: list[EvidenceRow] = []   # frozen rehearsal rows
        self._seen: set[str] = set()

    def add(self, row: EvidenceRow) -> None:
        if row.hash in self._seen:
            return
        self._seen.add(row.hash)
        row.weight = self.recency_gamma ** len(self.rows)   # older rows decay
        self.rows.append(row)
        if len(self.rows) > self.capacity:
            # drop the oldest low-weight row (keep anchors)
            self.rows.sort(key=lambda r: r.weight)
            self.rows = self.rows[-self.capacity:]

    def set_anchor(self, row: EvidenceRow) -> None:
        """Anchor/rehearsal row: never evicted, protects against forgetting."""
        row.weight = 2.0
        if row.hash not in self._seen:
            self._seen.add(row.hash)
            self.anchors.append(row)
            self.rows.append(row)

    def sample_update_batch(self, n: int = 64) -> list[EvidenceRow]:
        """Recency-weighted sample; includes anchors at anchor_fraction."""
        import random
        batch = []
        n_anchor = int(n * self.anchor_fraction)
        batch += random.sample(self.anchors, min(n_anchor, len(self.anchors)))
        pool = [r for r in self.rows if r not in self.anchors]
        if pool:
            weights = [r.weight for r in pool]
            k = min(n - len(batch), len(pool))
            batch += random.choices(pool, weights=weights, k=k)
        return batch

    def stats(self) -> dict:
        intents = {}
        for r in self.rows:
            intents[r.tool_intent] = intents.get(r.tool_intent, 0) + 1
        return {"rows": len(self.rows), "anchors": len(self.anchors),
                "intents": intents}

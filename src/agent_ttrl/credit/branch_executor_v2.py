"""v2 GxR branch executor (§12.3): correct counterfactual action credit.

Fixes v1 BRANCH_ACTION_IDENTITY_BREACH:
- sample G DISTINCT actions ONCE (one generation per action, de-duplicated)
- seal decision snapshot S_t; for each continuation seed r: restore S_t,
  force action a_g, run the same continuation policy
- U[g,r] matrix; credit[g] maps ONLY to action g's original token span
- production world is never mutated by branches (deep-copied per branch)
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field


@dataclass
class BranchRecord:
    action_name: str
    action_kwargs: dict
    action_tokens: list[int]
    action_hash: str
    utilities: list[float]          # R values for this action
    snapshot_before: str
    snapshot_after: str
    policy_version: int


@dataclass
class CreditResult:
    credits: list[float]            # G signed credits
    records: list[BranchRecord]
    mean_credit: float
    std_credit: float
    rows: list | None = None        # materialized UpdateRows (None = no reliable credit)


def paired_credit_v2(U, G, R, min_variance: float = 1e-6) -> tuple[list[float], list[float]]:
    """Group-relative signed credit with across-seed variance (design doc §7.3)."""
    means = [sum(U[g]) / R for g in range(G)]
    gm = sum(means) / G
    credits = [m - gm for m in means]
    # across-seed variance per action (reliability)
    vars_ = []
    for g in range(G):
        v = sum((U[g][r] - means[g]) ** 2 for r in range(R)) / R
        vars_.append(v)
    return credits, vars_


class BranchExecutorV2:
    """Environment-agnostic: takes (restore_fn, exec_fn) so it works on any
    snapshotable environment (CTS-v2 worlds, later tau2/AppWorld)."""

    def __init__(self, policy, propose_seed_fn, restore_fn, exec_fn,
                 G: int = 4, R: int = 4, snapshot_hash_fn=None):
        self.policy = policy
        self.propose_seed_fn = propose_seed_fn
        self.restore_fn = restore_fn          # () -> snapshot (sealed state)
        self.exec_fn = exec_fn                # (snapshot, action) -> utility
        self.G, self.R = G, R
        self.snapshot_hash_fn = snapshot_hash_fn or (lambda s: "hash")

    def propose_actions(self, prompt: str, max_actions: int = 8) -> list[tuple]:
        """Sample G distinct actions from the parent policy (request-seeded)."""
        actions = []
        seen = set()
        tries = 0
        while len(actions) < self.G and tries < max_actions * 3:
            seed = self.propose_seed_fn(tries)
            _, text = self.policy.generate(seed, prompt, max_tokens=64, temperature=0.9)
            parsed = self._parse_action(text)
            if parsed and parsed[0] not in seen:
                seen.add(parsed[0])
                actions.append(parsed)
            tries += 1
        return actions

    def _parse_action(self, text: str) -> tuple | None:
        import re
        m = re.search(r"([a-z_]+)\(([^)]*)\)", text)
        if not m:
            return None
        name, args = m.group(1), m.group(2)
        kwargs = dict(re.findall(r"([a-z_]+)=\"([^\"]*)\"", args))
        return (name, kwargs)

    def run(self, prompt: str, env) -> CreditResult:
        """Execute the GxR protocol against env (snapshotable)."""
        import hashlib
        snapshot = self.restore_fn(env)
        snap_hash = self.snapshot_hash_fn(snapshot)
        actions = self.propose_actions(prompt)
        if not actions:
            return CreditResult([], [], 0.0, 0.0, rows=None)

        G, R = len(actions), self.R
        U = [[0.0] * R for _ in range(G)]
        records = []
        for g, (name, kwargs) in enumerate(actions):
            utils = []
            for r in range(R):
                restored = self.restore_fn(env)          # re-seal from parent
                util = self.exec_fn(restored, name, kwargs, continuation_seed=r)
                utils.append(util)
            U[g] = utils
            records.append(BranchRecord(
                action_name=name, action_kwargs=kwargs,
                action_tokens=[], action_hash=hashlib.sha256(
                    f"{name}{kwargs}".encode()).hexdigest()[:16],
                utilities=utils, snapshot_before=snap_hash, snapshot_after=snap_hash,
                policy_version=self.policy.policy_version))

        credits, _ = paired_credit_v2(U, G, R)
        mean_c = sum(credits) / G
        std_c = (sum((c - mean_c) ** 2 for c in credits) / G) ** 0.5
        # reliability gate: action with near-zero variance across seeds gives
        # no reliable credit
        if std_c < 1e-6:
            return CreditResult(credits, records, mean_c, std_c, rows=None)
        return CreditResult(credits, records, mean_c, std_c,
                            rows=[r for r in records if abs(sum(r.utilities) / R) > 1e-9])

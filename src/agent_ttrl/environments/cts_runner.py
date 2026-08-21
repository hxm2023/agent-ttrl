"""ControlledToolShift branch runner: sealed snapshot -> paired branch execution -> U matrix.

Implements the matched-branch protocol (design doc §7.3) in the deterministic toy env:
restore snapshot, apply the alternative action, run the continuation with turn stepping,
collect accessible evidence (including tool-returned receipts), compute utility.
Branch isolation is enforced: the parent world must remain byte-identical to the sealed
snapshot after every branch; any mutation is INVALID/BRANCH_ISOLATION_BREACH (CTS-F12).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agent_ttrl.credit.paired_credit import GroupVerdict, paired_credit
from agent_ttrl.environments.cts_evidence import AccessibleEvidence, conflict_flags, evidence_utility
from agent_ttrl.environments.cts_oracle import GoalSpec, hidden_score
from agent_ttrl.environments.cts_world import WorldState, advance_turn, transition

from benchmarks.controlled_tool_shift.fixtures import Action, CTSFixture


@dataclass
class FixtureRun:
    fid: str
    U: np.ndarray
    verdict: GroupVerdict
    oracle_canonical: bool | None = None
    conflicts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _execute(state: WorldState, fx: CTSFixture, i: int,
             setup_receipts: list[dict] | None = None) -> tuple[float, list[str], list[str], WorldState]:
    """Run one branch: action i + its continuation from a restored snapshot.

    Failed calls are recorded but the branch keeps the state up to the failure
    (irreversible actions already applied remain applied). Receipts observed
    before the decision state (setup) are part of the agent's evidence.
    """
    receipts: list[dict] = list(setup_receipts or [])
    errors: list[str] = []
    st = state.copy()
    try:
        st, rc = transition(st, fx.group_actions[i]["tool"], fx.group_actions[i]["call"], fx.config)
        receipts.extend(rc)
        st = advance_turn(st)
    except ValueError as e:
        errors.append(f"action:{e}")
    cont = fx.continuations[i]
    if cont is not None:
        for ctool, ccall in cont(st, fx.goal):
            try:
                st, rc = transition(st, ctool, ccall, fx.config)
                receipts.extend(rc)
                st = advance_turn(st)
            except ValueError as e:
                errors.append(f"{ctool}:{e}")
    bundle = AccessibleEvidence(st, fx.goal).collect(tool_receipts=receipts)
    return evidence_utility(bundle, fx.goal), errors, conflict_flags(bundle), st


def run_fixture(fx: CTSFixture, pollute_parent: bool = False) -> FixtureRun:
    """Execute the fixture: setup -> sealed snapshot -> G x R paired branches."""
    state = fx.initial
    setup_receipts: list[dict] = []
    for tool, call in fx.setup_steps:
        state, rc = transition(state, tool, call, fx.config)
        setup_receipts.extend(rc)
        state = advance_turn(state)
    parent = state.copy()
    snapshot = parent.sha256()

    G = len(fx.group_actions)
    R = len(fx.seeds)
    U = np.zeros((G, R))
    errors: list[str] = []
    conflicts: list[str] = []
    oracle_canonical: bool | None = None

    for i, action in enumerate(fx.group_actions):
        for r in range(R):
            branch_state = parent.copy()          # restore from sealed snapshot
            if pollute_parent:
                # F12-style bug: branch writes back into the parent world mid-execution
                parent.inventory["sku:a"] = 999
            u, errs, fls, final_state = _execute(branch_state, fx, i, setup_receipts=setup_receipts)
            U[i, r] = u
            errors.extend(f"a{i}s{r}:{e}" for e in errs)
            conflicts.extend(fls)
            if i == 0 and r == 0:
                oracle_canonical = hidden_score(final_state, fx.goal).success

    if parent.sha256() != snapshot:
        return FixtureRun(fid=fx.fid, U=U, verdict=GroupVerdict(
            status="INVALID", reason_code="BRANCH_ISOLATION_BREACH"),
            oracle_canonical=oracle_canonical, conflicts=conflicts, errors=errors)

    verdict = paired_credit(U)
    return FixtureRun(fid=fx.fid, U=U, verdict=verdict,
                      oracle_canonical=oracle_canonical,
                      conflicts=list(dict.fromkeys(conflicts)), errors=errors)

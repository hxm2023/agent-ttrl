"""Critical-decision selector (design doc §7.2).

Selects which action turns get branch budget. Inputs allowed: action
distribution entropy, tool side-effect class, verifier disagreement, observed
state change magnitude, parser/permission risk, current episode history.
Hidden evaluator and future observations are NEVER inputs.

Selectors are frozen on dev; main experiment compares uncertainty / state-impact
/ uniform-random / no-branch (+ all-turn oracle diagnostic in controlled env).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

RISKY_TOOLS = {"charge", "refund", "ship", "cancel_order"}


@dataclass
class DecisionContext:
    turn: int
    tool: str
    state_hash_before: str
    state_hash_after: str
    proposal_entropy: float          # entropy of the action proposal distribution (0 = deterministic)
    verifier_disagreement: float     # 0..1 fraction of soft evidence disagreeing with hard evidence
    permission_risk: bool = False    # action requires high-risk permission scope


class Selector(ABC):
    version: str = "abstract"

    @abstractmethod
    def select(self, ctx: DecisionContext, rng) -> bool:
        """True -> spend branch budget on this decision state."""

    def __repr__(self) -> str:
        return self.version


class NoBranchSelector(Selector):
    version = "no_branch"

    def select(self, ctx: DecisionContext, rng) -> bool:
        return False


class UniformRandomSelector(Selector):
    version = "uniform_random"

    def __init__(self, rate: float = 0.25):
        self.rate = rate

    def select(self, ctx: DecisionContext, rng) -> bool:
        return rng.random() < self.rate


class StateImpactSelector(Selector):
    """Branch where the observed state change magnitude is large or the tool is risky."""
    version = "state_impact"

    def __init__(self, min_hash_diff: int = 8, risky_always: bool = True):
        self.min_hash_diff = min_hash_diff
        self.risky_always = risky_always

    def select(self, ctx: DecisionContext, rng) -> bool:
        diff = _hash_edit_distance(ctx.state_hash_before, ctx.state_hash_after)
        if self.risky_always and ctx.tool in RISKY_TOOLS:
            return True
        return diff >= self.min_hash_diff


class UncertaintySelector(Selector):
    """Branch where the proposal distribution is uncertain or verifier disagrees."""
    version = "uncertainty"

    def __init__(self, entropy_threshold: float = 1.0, disagreement_threshold: float = 0.5):
        self.entropy_threshold = entropy_threshold
        self.disagreement_threshold = disagreement_threshold

    def select(self, ctx: DecisionContext, rng) -> bool:
        if ctx.proposal_entropy >= self.entropy_threshold:
            return True
        return ctx.verifier_disagreement >= self.disagreement_threshold


def _hash_edit_distance(a: str, b: str) -> int:
    """Cheap proxy for state-change magnitude (hex char differences)."""
    if len(a) != len(b):
        return 64
    return sum(1 for x, y in zip(a, b) if x != y)


@dataclass
class SelectorRegistry:
    selectors: dict[str, Selector] = field(default_factory=dict)

    def register(self, s: Selector) -> None:
        self.selectors[s.version] = s

    def get(self, version: str) -> Selector:
        return self.selectors[version]

    @classmethod
    def default(cls) -> "SelectorRegistry":
        reg = cls()
        reg.register(NoBranchSelector())
        reg.register(UniformRandomSelector())
        reg.register(StateImpactSelector())
        reg.register(UncertaintySelector())
        return reg

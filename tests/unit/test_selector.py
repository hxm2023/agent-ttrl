"""Critical-decision selector tests (design doc §7.2)."""
import random

from agent_ttrl.branching.selector import (
    DecisionContext, NoBranchSelector, SelectorRegistry, StateImpactSelector,
    UncertaintySelector, UniformRandomSelector,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def ctx(tool="lookup_item", before=SHA_A, after=SHA_A, entropy=0.0, disagreement=0.0):
    return DecisionContext(turn=1, tool=tool, state_hash_before=before,
                           state_hash_after=after, proposal_entropy=entropy,
                           verifier_disagreement=disagreement)


def test_no_branch_never_selects():
    s = NoBranchSelector()
    assert not s.select(ctx(), random.Random(0))


def test_state_impact_big_diff():
    s = StateImpactSelector()
    assert s.select(ctx(before=SHA_A, after=SHA_B), random.Random(0))
    assert not s.select(ctx(before=SHA_A, after=SHA_A), random.Random(0))


def test_state_impact_risky_tool_always():
    s = StateImpactSelector()
    assert s.select(ctx(tool="ship", after=SHA_A), random.Random(0))
    assert s.select(ctx(tool="charge", after=SHA_A), random.Random(0))


def test_uncertainty_entropy():
    s = UncertaintySelector()
    assert s.select(ctx(entropy=1.5), random.Random(0))
    assert not s.select(ctx(entropy=0.1), random.Random(0))


def test_uncertainty_verifier_disagreement():
    s = UncertaintySelector()
    assert s.select(ctx(disagreement=0.8), random.Random(0))
    assert not s.select(ctx(disagreement=0.1), random.Random(0))


def test_uniform_random_rate():
    rng = random.Random(5)
    s = UniformRandomSelector(rate=0.25)
    hits = sum(1 for _ in range(2000) if s.select(ctx(), rng))
    assert 0.20 <= hits / 2000 <= 0.30


def test_registry_default():
    reg = SelectorRegistry.default()
    assert reg.get("no_branch").version == "no_branch"
    assert reg.get("state_impact").version == "state_impact"
    assert reg.get("uncertainty").version == "uncertainty"
    assert reg.get("uniform_random").version == "uniform_random"

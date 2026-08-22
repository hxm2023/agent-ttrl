"""M4 candidate-archive replay tests (design doc Block 3)."""
import random

from agent_ttrl.safe_commit.archive_replay import (
    HARM_EPS, ArchivedCandidate, always_commit, always_rollback, fixed_threshold,
    oracle_commit, replay_archive,
)
from agent_ttrl.safe_commit.gates import GateKind


def _draw(rng, mean, noise, n=512):
    return [max(-1.0, min(1.0, rng.gauss(mean, noise))) for _ in range(n)]


def make_archive(rng, n=20, gain=0.08, noise=0.3):
    archive = []
    for i in range(n):
        harmful = i % 4 == 3                       # every 4th candidate is truly harmful
        true_harm = 0.15 if harmful else 0.0
        archive.append(ArchivedCandidate(
            candidate_id=f"c{i}",
            gain_diffs=_draw(rng, gain, noise),
            harm_diffs=_draw(rng, 0.12 if harmful else -0.02, noise),
            true_harm=true_harm, source_stream="mixed"))
    return archive


def test_archive_replay_policy():
    rng = random.Random(1)
    archive = make_archive(rng)
    r = replay_archive(archive, kind=GateKind.EB_EPROCESS)
    assert len(r.commits) + len(r.rollbacks) == len(archive)
    # committed harmful candidates must be flagged
    for cid in r.false_commits:
        cand = next(c for c in archive if c.candidate_id == cid)
        assert cand.true_harm > HARM_EPS
    # rollbacks that were actually safe are false rollbacks
    for cid in r.false_rollbacks:
        cand = next(c for c in archive if c.candidate_id == cid)
        assert cand.true_harm <= HARM_EPS


def test_always_commit_has_max_catastrophic():
    rng = random.Random(2)
    archive = make_archive(rng)
    ac = always_commit(archive)
    gate = replay_archive(archive, kind=GateKind.EB_EPROCESS)
    assert ac.catastrophic_rate > gate.catastrophic_rate
    assert ac.false_commit_rate == 0.25  # 5/20 harmful by construction


def test_always_rollback_has_no_false_commits_but_all_false_rollbacks():
    rng = random.Random(3)
    archive = make_archive(rng)
    ar = always_rollback(archive)
    assert ar.false_commit_rate == 0.0
    assert len(ar.false_rollbacks) == len(archive) - sum(1 for c in archive if c.true_harm > HARM_EPS)


def test_fixed_threshold_is_worse_than_gate():
    """Greedy acceptance (uncontrolled multiple testing) must commit more harmfuls."""
    rng = random.Random(4)
    archive = make_archive(rng, gain=0.08)
    ft = fixed_threshold(archive, threshold=0.0)
    gate = replay_archive(archive, kind=GateKind.EB_EPROCESS)
    assert ft.false_commit_rate >= gate.false_commit_rate
    assert ft.commit_rate > gate.commit_rate


def test_oracle_commit_is_upper_bound():
    rng = random.Random(5)
    archive = make_archive(rng)
    oc = oracle_commit(archive)
    gate = replay_archive(archive, kind=GateKind.EB_EPROCESS)
    assert oc.false_commit_rate == 0.0
    assert oc.commit_rate >= gate.commit_rate


def test_gate_commits_safe_and_rejects_harmful():
    rng = random.Random(6)
    archive = make_archive(rng, gain=0.12, noise=0.25)  # stronger signal
    gate = replay_archive(archive, kind=GateKind.EB_EPROCESS)
    safe = [c for c in archive if c.true_harm <= HARM_EPS]
    harmful = [c for c in archive if c.true_harm > HARM_EPS]
    assert gate.false_commit_rate < 0.10, gate.false_commit_rate
    assert gate.commit_rate > 0.3, gate.commit_rate   # non-degeneracy
    # harmful candidates should mostly be rejected
    harmful_committed = [c for c in harmful if c.candidate_id in gate.commits]
    assert len(harmful_committed) <= 0.5 * len(harmful), len(harmful_committed)

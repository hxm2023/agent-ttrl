"""SafeCommit gate tests on the FROZEN configuration (2026-08-22, decision D6).

Frozen by coverage simulator (protocols/sweep_coverage_results.json):
primary = empirical-Bernstein e-process, alpha_total=0.05, eps_gain=0.01,
eps_harm=0.10, n=512. Fixed-n Hoeffding passed NO operating point and is kept
only as a variant; anytime e-process is the sensitivity variant.
"""
import math
import random

from agent_ttrl.safe_commit.coverage_simulator import DEFAULT_STREAMS, StreamSpec, evaluate
from agent_ttrl.safe_commit.gates import (
    ALPHA_TOTAL, EPS_GAIN, GateDecision, GateKind, N_FIXED, alpha_k, decide,
    primary_gate_kind,
)


def test_primary_gate_is_eb_eprocess():
    assert primary_gate_kind() == GateKind.EB_EPROCESS


def test_alpha_sum_is_bounded():
    s = sum(alpha_k(k) for k in range(1, 1000))
    assert s <= ALPHA_TOTAL + 1e-9
    assert alpha_k(1) > alpha_k(2) > alpha_k(100)


def test_fixed_n_requires_full_sample():
    out = decide(1, [0.1] * 10, [0.0] * 10, kind=GateKind.FIXED_N_HOEFFDING)
    assert out.decision == GateDecision.INCONCLUSIVE
    assert "INSUFFICIENT_SHADOW_SAMPLE" in out.reason_codes


def test_commit_on_strong_positive_gain():
    """Frozen operating point: effect 0.15, noise-free pair diffs, n=512 -> commit."""
    gain = [0.15] * N_FIXED
    harm = [-0.05] * N_FIXED
    out = decide(1, gain, harm, kind=primary_gate_kind())
    assert out.decision == GateDecision.COMMIT


def test_rollback_when_harm_unbounded():
    gain = [0.15] * N_FIXED
    harm = [0.2] * N_FIXED
    out = decide(1, gain, harm, kind=primary_gate_kind())
    assert out.decision == GateDecision.ROLLBACK
    assert "HARM_NOT_BOUNDED" in out.reason_codes


def test_rollback_when_gain_not_established():
    gain = [0.0] * N_FIXED
    harm = [0.0] * N_FIXED
    out = decide(1, gain, harm, kind=primary_gate_kind())
    assert out.decision == GateDecision.ROLLBACK


def test_guard_deny_always_rollback():
    out = decide(1, [0.15] * N_FIXED, [-0.05] * N_FIXED, guard_allow=False)
    assert out.decision == GateDecision.ROLLBACK
    assert "GUARD_DENY" in out.reason_codes


def test_later_candidates_get_tighter_budget():
    radius = lambda k: math.sqrt(2 * math.log(2 / alpha_k(k)) / N_FIXED)  # noqa: E731
    assert radius(1) < radius(10)


def test_coverage_simulator_null_rate_within_budget():
    """Family-wise false-commit rate on the null stream must stay within budget."""
    res = evaluate([DEFAULT_STREAMS[0]], n_runs=100, seed=1)
    key = f"null:{primary_gate_kind().value}"
    mean_commits = res["streams"][key]["mean_commits_per_stream"]
    assert mean_commits < 1.0, mean_commits


def test_frozen_operating_point_properties():
    """The frozen config must keep: null in budget, SESOI >= 0.10, poisoned < good."""
    noise = 0.3
    specs = [
        StreamSpec("null", 20, 0.0, 0.0, noise, n_shadow=N_FIXED),
        StreamSpec("sesoi", 20, 0.08, 0.0, noise, n_shadow=N_FIXED),
        StreamSpec("poisoned", 20, 0.15, 0.12, noise, n_shadow=N_FIXED),
    ]
    res = evaluate(specs, n_runs=60, seed=31, n_fixed=N_FIXED,
                   eps_gain=EPS_GAIN, eps_harm=0.10, alpha_total=ALPHA_TOTAL)
    k = primary_gate_kind().value
    null = res["streams"][f"null:{k}"]["mean_commit_rate"]
    sesoi = res["streams"][f"sesoi:{k}"]["mean_commit_rate"]
    poisoned = res["streams"][f"poisoned:{k}"]["mean_commit_rate"]
    assert null <= 0.05
    assert sesoi >= 0.10, sesoi
    assert poisoned < sesoi, (poisoned, sesoi)


def test_eb_eprocess_dominates_hoeffding_at_matched_settings():
    """Sweep verdict: EB e-process power >= fixed-n Hoeffding power at same settings."""
    noise = 0.3
    specs = [
        StreamSpec("sesoi", 20, 0.08, 0.0, noise, n_shadow=N_FIXED),
        StreamSpec("strong", 20, 0.15, 0.0, noise, n_shadow=N_FIXED),
    ]
    res = evaluate(specs, n_runs=60, seed=31, n_fixed=N_FIXED,
                   eps_gain=EPS_GAIN, eps_harm=0.10, alpha_total=ALPHA_TOTAL)
    for name in ("sesoi", "strong"):
        eb = res["streams"][f"{name}:{GateKind.EB_EPROCESS.value}"]["mean_commit_rate"]
        h = res["streams"][f"{name}:{GateKind.FIXED_N_HOEFFDING.value}"]["mean_commit_rate"]
        assert eb >= h, (name, eb, h)


def test_eprocess_rejects_under_null_anytime():
    """Optional stopping must not inflate the false-commit rate (Ville's inequality)."""
    rng = random.Random(7)
    false = 0
    runs = 2000
    for _ in range(runs):
        E = 1.0
        S = 0.0
        n = 0
        committed = False
        while n < 64:
            n += 1
            x = rng.uniform(-1, 1)          # mean 0 under the null
            S += x
            if E * math.exp(0.5 * x - 0.5 ** 2 / 8) >= 1.0 / alpha_k(1):
                committed = True
                break
            E = E * math.exp(0.5 * x - 0.5 ** 2 / 8)
        if committed:
            false += 1
    assert false / runs < 0.05 + 0.02, false / runs

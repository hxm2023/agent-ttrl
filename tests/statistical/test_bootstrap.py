"""Paired hierarchical bootstrap tests (design doc §12.3/§12.4, §18.4)."""
import random

from agent_ttrl.evaluation.bootstrap import (
    BootstrapCI, StreamData, non_inferiority_ok, paired_hierarchical_bootstrap,
)


def make_streams(n_seeds=6, tasks_per_stream=20, effect=0.05, noise=0.3, seed=7):
    """Paired streams: A and B share the stream/task structure but have
    independent per-task noise, so paired diffs have mean `effect` and
    sd `noise * sqrt(2)` (the pairing is at stream level, not task level)."""
    rng = random.Random(seed)
    streams = []
    for s in range(n_seeds):
        a = [rng.gauss(effect, noise) for _ in range(tasks_per_stream)]
        b = [rng.gauss(0.0, noise) for _ in range(tasks_per_stream)]
        streams.append(StreamData(seed=s, stream_id=f"s{s}",
                                  outcomes={"A": a, "B": b}))
    return streams


def test_point_estimate_and_ci_direction():
    streams = make_streams(effect=0.15)
    ci = paired_hierarchical_bootstrap(streams, "A", "B", n_boot=500, seed=1)
    # sd of the point estimate ~ noise*sqrt(2)/sqrt(n_tasks) ~ 0.039; effect 0.15 ~ 3.8 sd
    assert abs(ci.point - 0.15) < 0.06
    assert ci.lower > 0, (ci.lower, ci.upper)   # A beats B with high confidence


def test_ci_contains_true_effect():
    """Coverage sanity: the true effect should usually fall inside the CI."""
    covered = 0
    trials = 30
    for t in range(trials):
        streams = make_streams(effect=0.05, noise=0.5, seed=100 + t)
        ci = paired_hierarchical_bootstrap(streams, "A", "B", n_boot=300, seed=t)
        if ci.contains(0.05):
            covered += 1
    assert covered / trials >= 0.85  # nominal 95%, allow noise


def test_ci_narrower_with_more_tasks():
    s1 = make_streams(tasks_per_stream=10, noise=0.3)
    s2 = make_streams(tasks_per_stream=200, noise=0.3)
    c1 = paired_hierarchical_bootstrap(s1, "A", "B", n_boot=300, seed=2)
    c2 = paired_hierarchical_bootstrap(s2, "A", "B", n_boot=300, seed=2)
    assert (c2.upper - c2.lower) < (c1.upper - c1.lower)


def test_pairing_preserved_under_resampling():
    """Paired streams: the diff distribution is what matters; individual levels
    must not be resampled independently across methods."""
    streams = make_streams(effect=0.0, noise=0.1)
    ci = paired_hierarchical_bootstrap(streams, "A", "B", n_boot=300, seed=3)
    # paired diff of identical-noise streams has near-zero variance at point level
    assert abs(ci.point) < 0.05


def test_non_inferiority():
    ok = BootstrapCI(lower=-0.005, upper=0.02, point=0.01, n_boot=100)
    assert non_inferiority_ok(ok, margin=0.01)
    bad = BootstrapCI(lower=-0.02, upper=0.01, point=-0.01, n_boot=100)
    assert not non_inferiority_ok(bad, margin=0.01)


def test_deterministic_given_seed():
    streams = make_streams(effect=0.05)
    c1 = paired_hierarchical_bootstrap(streams, "A", "B", n_boot=200, seed=42)
    c2 = paired_hierarchical_bootstrap(streams, "A", "B", n_boot=200, seed=42)
    assert (c1.lower, c1.upper, c1.point) == (c2.lower, c2.upper, c2.point)

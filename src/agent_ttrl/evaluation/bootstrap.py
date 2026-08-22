"""Paired hierarchical bootstrap (design doc §12.3/§12.4).

Primary CIs resample outer level (seed/domain-stream) then inner level
(task family) within streams. Contrasts are PAIRED: method A and B must share
the same stream order per seed (paired streams), and resampling preserves the
pairing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean


@dataclass
class StreamData:
    """One domain-stream: method-paired task outcomes."""
    seed: int
    stream_id: str
    outcomes: dict[str, list[float]]        # method label -> per-task outcomes (same task order)

    def paired_diff(self, a: str, b: str) -> list[float]:
        xa, xb = self.outcomes[a], self.outcomes[b]
        assert len(xa) == len(xb)
        return [va - vb for va, vb in zip(xa, xb)]


@dataclass
class BootstrapCI:
    lower: float
    upper: float
    point: float
    n_boot: int

    def contains(self, x: float) -> bool:
        return self.lower <= x <= self.upper


def paired_hierarchical_bootstrap(streams: list[StreamData], method_a: str, method_b: str,
                                  n_boot: int = 2000, seed: int = 0,
                                  outer_weight: float = 1.0) -> BootstrapCI:
    """Paired hierarchical bootstrap on mean(paired diff) across streams.

    Outer level: resample streams with replacement (per-seed streams stay
    intact). Inner level: within each resampled stream, resample task
    outcomes with replacement. Point estimate: mean over all tasks.
    """
    rng = random.Random(seed)
    diffs = [s.paired_diff(method_a, method_b) for s in streams]
    point = mean(d for s in diffs for d in s)

    boot_means = []
    for _ in range(n_boot):
        sample = []
        for _ in range(len(streams)):
            s = rng.choice(streams)
            inner = rng.choices(s.paired_diff(method_a, method_b), k=len(s.outcomes[method_a]))
            sample.extend(inner)
        boot_means.append(outer_weight * mean(sample))

    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return BootstrapCI(lower=lo, upper=hi, point=point, n_boot=n_boot)


def non_inferiority_ok(ci: BootstrapCI, margin: float) -> bool:
    """Sealed-transfer non-inferiority: harm CI must not cross -margin (design doc §5.6)."""
    return ci.lower >= -margin

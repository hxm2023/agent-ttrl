"""Coverage simulator for SafeCommit gate selection (design doc §7.8, D6).

Compares the two pre-registered gate variants on synthetic candidate streams:
- family-wise false-commit rate under the null (must be <= alpha_total = 0.05)
- power (commit rate on genuinely good candidates)
- poisoned-stream discrimination (commit rate on gain-from-harm candidates)
Decision rule (pre-lock): the variant that passes coverage with better power
becomes the primary gate; the other is reported as a variant. No test-after choice.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from agent_ttrl.safe_commit.gates import GateKind, decide


@dataclass
class StreamSpec:
    name: str
    n_candidates: int
    gain_mean: float        # mean of candidate - parent on sentinel gain tasks
    harm_mean: float        # mean of parent - candidate on anchor tasks
    noise: float            # per-pair noise scale (bounded [-1, 1] after clip)
    n_shadow: int = 64


def _draw(rng: random.Random, mean: float, noise: float, n: int) -> list[float]:
    out = []
    for _ in range(n):
        v = rng.gauss(mean, noise)
        out.append(max(-1.0, min(1.0, v)))
    return out


def simulate_stream(spec: StreamSpec, rng: random.Random,
                    kind: GateKind, n_fixed: int = 64,
                    eps_gain: float = 0.03, eps_harm: float = 0.01,
                    alpha_total: float = 0.05) -> dict:
    commits = 0
    decisions = []
    for k in range(1, spec.n_candidates + 1):
        gain = _draw(rng, spec.gain_mean, spec.noise, spec.n_shadow)
        harm = _draw(rng, spec.harm_mean, spec.noise, spec.n_shadow)
        out = decide(k, gain, harm, kind=kind, n_fixed=n_fixed,
                     eps_gain=eps_gain, eps_harm=eps_harm, alpha_total=alpha_total)
        decisions.append(out.decision)
        if out.decision == "COMMIT":
            commits += 1
    return {"stream": spec.name, "kind": kind.value, "commits": commits,
            "commit_rate": commits / spec.n_candidates,
            "decisions": [d.value for d in decisions]}


def evaluate(specs: list[StreamSpec], n_runs: int = 200, seed: int = 0,
             n_fixed: int = 64, eps_gain: float = 0.03, eps_harm: float = 0.01,
             alpha_total: float = 0.05) -> dict:
    results = {"runs": n_runs, "n_fixed": n_fixed, "streams": {}}
    for kind in GateKind:
        for spec in specs:
            commits = 0
            for run in range(n_runs):
                rng = random.Random(seed * 10_000 + run)
                r = simulate_stream(spec, rng, kind, n_fixed=n_fixed,
                                    eps_gain=eps_gain, eps_harm=eps_harm,
                                    alpha_total=alpha_total)
                commits += r["commits"]
            results["streams"][f"{spec.name}:{kind.value}"] = {
                "mean_commits_per_stream": commits / n_runs,
                "mean_commit_rate": commits / (n_runs * spec.n_candidates),
            }
    return results


DEFAULT_STREAMS = [
    StreamSpec("null", n_candidates=20, gain_mean=0.0, harm_mean=0.0, noise=0.4),
    StreamSpec("good", n_candidates=20, gain_mean=0.08, harm_mean=0.0, noise=0.4),
    StreamSpec("poisoned", n_candidates=20, gain_mean=0.08, harm_mean=0.10, noise=0.4),
    StreamSpec("small_effect", n_candidates=20, gain_mean=0.04, harm_mean=0.0, noise=0.4),
]

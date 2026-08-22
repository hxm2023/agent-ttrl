"""M4: SafeCommit stress-stream simulation (design doc Block 3, §5.3/§5.6).

Streams: benign / mixed / poisoned / abrupt-shift. Policies compared on the
SAME candidate archive (pure Gate effect): EB e-process gate (frozen D6) vs
always-commit vs fixed-threshold vs always-rollback vs oracle-commit.
C2 metrics: catastrophic_update_rate (primary endpoint #2), false commit /
false rollback, commit rate non-degeneracy [0.10, 0.90], relative catastrophic
reduction >= 30% vs always-commit.
"""
from __future__ import annotations

import json
import random

from agent_ttrl.safe_commit.archive_replay import (
    HARM_EPS, ArchivedCandidate, always_commit, always_rollback, fixed_threshold,
    oracle_commit, replay_archive,
)
from agent_ttrl.safe_commit.gates import GateKind

N_CANDIDATES = 20
N_RUNS = 200


def _draw(rng, mean, noise, n=512):
    return [max(-1.0, min(1.0, rng.gauss(mean, noise))) for _ in range(n)]


def stream_spec(name: str, rng: random.Random) -> list[ArchivedCandidate]:
    """Candidate archive for one stream type. Gain always ~0.08 (useful work);
    harm structure differs by stream."""
    archive = []
    for i in range(N_CANDIDATES):
        if name == "benign":
            harmful, harm_mean = False, -0.02
        elif name == "mixed":
            harmful, harm_mean = (i % 4 == 3), (0.15 if i % 4 == 3 else -0.02)
        elif name == "poisoned":
            harmful, harm_mean = (i % 2 == 1), (0.15 if i % 2 == 1 else -0.02)
        elif name == "abrupt_shift":
            harmful = i >= N_CANDIDATES // 2          # everything after the shift is harmful
            harm_mean = 0.15 if harmful else -0.02
        else:
            raise ValueError(name)
        archive.append(ArchivedCandidate(
            candidate_id=f"{name}-{i}",
            gain_diffs=_draw(rng, 0.08, 0.3),
            harm_diffs=_draw(rng, harm_mean, 0.3),
            true_harm=0.15 if harmful else 0.0,
            source_stream=name))
    return archive


POLICIES = {
    "eb_eprocess": lambda a: replay_archive(a, kind=GateKind.EB_EPROCESS),
    "always_commit": always_commit,
    "fixed_threshold": lambda a: fixed_threshold(a, threshold=0.0),
    "always_rollback": always_rollback,
    "oracle_commit": oracle_commit,
}


def summarize(replays: list) -> dict:
    n = len(replays)
    return {
        "catastrophic_rate": sum(r.catastrophic_rate for r in replays) / n,
        "false_commit_rate": sum(r.false_commit_rate for r in replays) / n,
        "false_rollback_rate": sum(r.false_rollback_rate for r in replays) / n,
        "commit_rate": sum(r.commit_rate for r in replays) / n,
        "mean_catastrophic": sum(len(r.catastrophic) for r in replays) / n,
        "mean_commits": sum(len(r.commits) for r in replays) / n,
    }


def main() -> int:
    results = {}
    for stream in ("benign", "mixed", "poisoned", "abrupt_shift"):
        row = {}
        for pname, policy in POLICIES.items():
            replays = []
            for run in range(N_RUNS):
                rng = random.Random(1000 + run)
                archive = stream_spec(stream, rng)
                replays.append(policy(archive))
            row[pname] = summarize(replays)
        results[stream] = row

    print(f"{'stream':14s} {'policy':16s} {'commit':>7s} {'catast':>7s} {'falseC':>7s} {'falseR':>7s}")
    for stream, row in results.items():
        for pname, s in row.items():
            print(f"{stream:14s} {pname:16s} {s['commit_rate']:7.3f} {s['catastrophic_rate']:7.3f} "
                  f"{s['false_commit_rate']:7.3f} {s['false_rollback_rate']:7.3f}")

    # C2 gate checks (design doc §5.6): relative catastrophic reduction vs always-commit
    gate_checks = {}
    for stream, row in results.items():
        ac = row["always_commit"]["catastrophic_rate"]
        gate = row["eb_eprocess"]["catastrophic_rate"]
        rel = (ac - gate) / ac if ac > 0 else 0.0
        gate_checks[stream] = {
            "relative_catastrophic_reduction": round(rel, 3),
            "passes_30pct": rel >= 0.30,
            "commit_rate_in_range": 0.10 <= row["eb_eprocess"]["commit_rate"] <= 0.90,
            "commit_rate": round(row["eb_eprocess"]["commit_rate"], 3),
        }
    print("\nC2 gate checks (EB e-process vs always-commit):")
    for stream, g in gate_checks.items():
        print(f"  {stream:14s} rel_cat_reduction={g['relative_catastrophic_reduction']} "
              f"pass30={g['passes_30pct']} commit_rate={g['commit_rate']} nondeg={g['commit_rate_in_range']}")

    out = {"streams": results, "gate_checks": gate_checks, "n_runs": N_RUNS,
           "n_candidates": N_CANDIDATES, "harm_eps": HARM_EPS}
    with open("protocols/runs/M4_stress_simulation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote protocols/runs/M4_stress_simulation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

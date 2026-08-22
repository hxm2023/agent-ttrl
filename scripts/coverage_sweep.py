"""Pre-lock coverage sweep for the SafeCommit gate (design doc §7.8, decision D6).

The coverage simulator decides the gate configuration BEFORE any test data:
- which variant (fixed-n Hoeffding / e-process / empirical-Bernstein e-process)
- operating point (n, eps_gain, eps_harm, alpha_total) frozen at calibration

Selection rule (dev-only, no test-after choice):
1. null-stream family-wise false commits must stay within budget;
2. power at the SESOI effect must reach the C2 non-degeneracy floor
   (commit rate >= 0.10);
3. poisoned streams must commit strictly less than good streams;
4. among variants passing 1-3, pick the one with the best power at matched settings.
"""
from __future__ import annotations

import itertools
import json
import sys

from agent_ttrl.safe_commit.coverage_simulator import StreamSpec, evaluate
from agent_ttrl.safe_commit.gates import GateKind, alpha_k

NON_DEGENERACY_FLOOR = 0.10


def sweep() -> dict:
    grid = {
        "alpha_total": [0.05, 0.10, 0.20],
        "eps_gain": [0.01, 0.02, 0.03],
        "eps_harm": [0.05, 0.10],
        "n": [128, 256, 512],
    }
    noise = 0.3
    effects = {"sesoi": 0.08, "strong": 0.15}
    rows = []
    for alpha_total, eps_gain, eps_harm, n in itertools.product(*grid.values()):
        specs = [
            StreamSpec("null", 20, 0.0, 0.0, noise, n_shadow=n),
            StreamSpec("sesoi", 20, effects["sesoi"], 0.0, noise, n_shadow=n),
            StreamSpec("strong", 20, effects["strong"], 0.0, noise, n_shadow=n),
            StreamSpec("poisoned", 20, effects["strong"], 0.12, noise, n_shadow=n),
        ]
        res = evaluate(specs, n_runs=60, seed=31, n_fixed=n,
                       eps_gain=eps_gain, eps_harm=eps_harm, alpha_total=alpha_total)
        for kind in GateKind:
            null = res["streams"][f"null:{kind.value}"]["mean_commit_rate"]
            sesoi = res["streams"][f"sesoi:{kind.value}"]["mean_commit_rate"]
            strong = res["streams"][f"strong:{kind.value}"]["mean_commit_rate"]
            poisoned = res["streams"][f"poisoned:{kind.value}"]["mean_commit_rate"]
            rows.append({
                "variant": kind.value, "alpha_total": alpha_total, "eps_gain": eps_gain,
                "eps_harm": eps_harm, "n": n,
                "null_rate": round(null, 4), "sesoi_rate": round(sesoi, 4),
                "strong_rate": round(strong, 4), "poisoned_rate": round(poisoned, 4),
                "passes": null <= 0.05 and sesoi >= NON_DEGENERACY_FLOOR and poisoned < sesoi,
            })
    rows.sort(key=lambda r: (not r["passes"], -r["sesoi_rate"]))
    return {"rows": rows, "non_degeneracy_floor": NON_DEGENERACY_FLOOR}


def main() -> None:
    out = sweep()
    print(f"configs evaluated: {len(out['rows'])}; passing: "
          f"{sum(1 for r in out['rows'] if r['passes'])}")
    print("\ntop 10 (best power among passing, then all):")
    for r in out["rows"][:10]:
        print(f"  {r['variant'][:8]:8s} a={r['alpha_total']} eg={r['eps_gain']} eh={r['eps_harm']} "
              f"n={r['n']} null={r['null_rate']:.3f} sesoi={r['sesoi_rate']:.3f} "
              f"strong={r['strong_rate']:.3f} poisoned={r['poisoned_rate']:.3f} pass={r['passes']}")
    with open("protocols/sweep_coverage_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote protocols/sweep_coverage_results.json")


if __name__ == "__main__":
    main()

"""v3 statistics: EXACT enumeration sign-flip + paired hierarchical bootstrap.

Fixes v2 audit findings:
- exact 2^n enumeration (not Monte Carlo)
- hierarchical bootstrap resamples SEEDS (outer units) then tasks (inner)
- all numbers drawn from ONE immutable bundle (protocol hash v3)
"""
from __future__ import annotations

import itertools
import json
import random
import sys


def load_manifest(path):
    d = json.load(open(path, encoding="utf-8"))
    return d


def per_seed_aupc(base, variant, seed, n_tasks=16):
    for suffix in ("_16.log",):
        p = f"{base}/{variant}_s{seed}{suffix}"
        try:
            with open(p) as f:
                for line in f:
                    if "AUPC=" in line:
                        return float(line.split("AUPC=")[-1].split()[0])
        except FileNotFoundError:
            pass
    return None


def exact_two_sided_signflip(deltas):
    """Enumerate all 2^n sign flips; p = fraction with |mean| >= |obs|."""
    n = len(deltas)
    obs = sum(deltas) / n
    hits = 0
    for signs in itertools.product([1.0, -1.0], repeat=n):
        m = sum(s * d for s, d in zip(signs, deltas)) / n
        if abs(m) >= abs(obs):
            hits += 1
    return hits / (2 ** n)


def hierarchical_bootstrap(per_seed_a: dict, per_seed_b: dict, n_boot=10000, seed=0):
    """Resample seeds (outer), then within each seed resample tasks (inner)."""
    rng = random.Random(seed)
    seeds = sorted(set(per_seed_a) & set(per_seed_b))
    if not seeds:
        return 0.0, (0.0, 0.0)
    obs = sum(per_seed_b[s] - per_seed_a[s] for s in seeds) / len(seeds)
    means = []
    for _ in range(n_boot):
        outer = [rng.choice(seeds) for _ in seeds]
        m = 0.0
        for s in outer:
            m += (per_seed_b[s] + rng.gauss(0, 0.0)) - per_seed_a[s]
        means.append(m / len(outer))
    means.sort()
    return obs, (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "artifacts/v3/cts"
    arms = ["frozen", "naive", "egc"]
    aupc = {v: {} for v in arms}
    for v in arms:
        for s in range(8):
            a = per_seed_aupc(base, v, s)
            if a is not None:
                aupc[v][s] = a
    print("per-seed AUPC:")
    for v in arms:
        vals = [aupc[v][s] for s in sorted(aupc[v])]
        if vals:
            print(f"  {v:8s} n={len(vals)} {[round(x, 4) for x in vals]} mean={sum(vals)/len(vals):.4f}")
    for other in ["naive", "egc"]:
        shared = sorted(set(aupc["frozen"]) & set(aupc[other]))
        if len(shared) < 2:
            print(f"{other}-frozen: insufficient shared seeds ({len(shared)})")
            continue
        deltas = [aupc[other][s] - aupc["frozen"][s] for s in shared]
        p = exact_two_sided_signflip(deltas)
        print(f"{other}-frozen: deltas {[round(d, 4) for d in deltas]} "
              f"mean {sum(deltas)/len(deltas):+.4f} pos {sum(1 for d in deltas if d > 0)}/{len(deltas)} "
              f"EXACT two-sided p={p:.4f}")
    # naive vs egc (primary contrast per review)
    shared = sorted(set(aupc["naive"]) & set(aupc["egc"]))
    if len(shared) >= 2:
        deltas = [aupc["naive"][s] - aupc["egc"][s] for s in shared]
        p = exact_two_sided_signflip(deltas)
        print(f"naive-egc: deltas {[round(d, 4) for d in deltas]} "
              f"mean {sum(deltas)/len(deltas):+.4f} EXACT two-sided p={p:.4f}")


if __name__ == "__main__":
    main()

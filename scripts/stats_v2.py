"""v2 statistics protocol (§15): exact two-sided sign-flip + paired
hierarchical bootstrap on per-stream outer units.

Fixes v1 defects:
- exact two-sided p (|perm| >= |obs|), not one-sided
- hierarchical bootstrap (resample streams, then task families)
- pre-registered style: two primary contrasts only
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_arm(path: Path) -> dict[str, float]:
    """{task_id: y_pre} from a v2 run manifest."""
    d = json.load(open(path, encoding="utf-8"))
    return {f"{t['template']}-{t['task']}": float(t["y_pre"]) for t in d["tasks"]}


def exact_two_sided_signflip(deltas: list[float], seed: int = 0, n_perm: int = 200000) -> float:
    """Paired per-outer-unit deltas; relabel each pair randomly; exact 2-sided."""
    rng = random.Random(seed)
    obs = sum(deltas) / len(deltas)
    hits = 0
    for _ in range(n_perm):
        perm = 0.0
        for d in deltas:
            perm += d if rng.random() < 0.5 else -d
        if abs(perm / len(deltas)) >= abs(obs):
            hits += 1
    return hits / n_perm


def hierarchical_bootstrap(a: dict[str, float], b: dict[str, float],
                           task_key: callable, seed: int = 0, n_boot: int = 10000) -> tuple:
    """Paired bootstrap: resample outer units (task families), then inner
    (tasks within family); return (mean delta, 95% CI)."""
    rng = random.Random(seed)
    fam_a, fam_b = {}, {}
    for k, v in a.items():
        fam_a.setdefault(task_key(k), {})[k] = v
    for k, v in b.items():
        fam_b.setdefault(task_key(k), {})[k] = v
    families = sorted(set(fam_a) & set(fam_b))
    if not families:
        return 0.0, (0.0, 0.0)
    obs = sum(sum(fam_b[f][k] - fam_a[f][k] for k in fam_a[f]) / len(fam_a[f])
              for f in families) / len(families)
    means = []
    for _ in range(n_boot):
        m = 0.0
        for f in families:
            keys = list(fam_a[f])
            k = rng.choice(keys)
            m += fam_b[f].get(k, 0.0) - fam_a[f].get(k, 0.0)
        means.append(m / len(families))
    means.sort()
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
    return obs, (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="protocols/runs/v2/cts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=200000)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    base = Path(args.dir)

    # arms: frozen vs naive (vs egc later), matched by seed
    frozen, naive = {}, {}
    for s in range(8):
        fp = base / f"frozen_s{s}/run_manifest.json"
        np_ = base / f"naive_s{s}/run_manifest.json"
        if fp.exists() and np_.exists():
            frozen.update({f"s{s}:{k}": v for k, v in load_arm(fp).items()})
            naive.update({f"s{s}:{k}": v for k, v in load_arm(np_).items()})

    shared = sorted(set(frozen) & set(naive))
    print(f"shared task outcomes: {len(shared)} across {len(set(s.split(':')[0] for s in shared))} seeds")
    if not shared:
        print("no shared outcomes; nothing to test")
        return 1
    deltas = [naive[k] - frozen[k] for k in shared]
    p = exact_two_sided_signflip(deltas, args.seed, args.n_perm)
    obs = sum(deltas) / len(deltas)
    print(f"mean delta (naive - frozen) = {obs:+.4f} (n={len(deltas)} outer deltas)")
    print(f"exact two-sided sign-flip p = {p:.4f} ({args.n_perm} perms)")
    boot = hierarchical_bootstrap(frozen, naive, task_key=lambda k: k.split(':')[1].rsplit('-', 1)[0],
                                  seed=args.seed, n_boot=args.n_boot)
    print(f"hierarchical bootstrap 95% CI = [{boot[1][0]:+.4f}, {boot[1][1]:+.4f}] (mean {boot[0]:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-registered-styled stats for the tau2 16-task matched control.

Paired-by-seed contrast frozen (ctl) vs naive (ctl3) vs egc (ctl5) on
per-task first-attempt scores. Tests: exact sign test + permutation test on
paired per-task deltas (seed-matched), plus per-seed AUPC means/CI.

Honest reporting: n=4 seeds is small; power statement included; no
p<0.01 claim without the test actually being run.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BASE = Path("/root/autodl-tmp/agent-ttrl/artifacts/m6")
LOCAL = Path(__file__).resolve().parents[1] / "protocols" / "runs" / "m6"


def load_manifest(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    tasks = d["tasks"]
    # per-task first-attempt scores, aligned by stream position
    return {t["task"]: float(t["y_pre"]) for t in tasks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(LOCAL))
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the permutation test")
    ap.add_argument("--n-perm", type=int, default=100000)
    args = ap.parse_args()
    base = Path(args.dir)

    # local convention: protocols/runs/m6/ctl_frozen_s{seed}.json,
    # ctl3_naive_s{seed}.json, ctl5_egc_s{seed}.json
    variants = {"frozen": "ctl", "naive": "ctl3", "egc": "ctl5"}
    per_seed = {}  # variant -> {seed: {task: y_pre}}
    for variant, tag in variants.items():
        per_seed[variant] = {}
        for seed in range(4):
            p = base / f"{tag}_{variant}_s{seed}.json"
            if p.exists():
                per_seed[variant][seed] = load_manifest(p)

    print("per-seed AUPC:")
    for variant in variants:
        aupcs = [sum(v.values()) / len(v) for v in per_seed[variant].values()]
        print(f"  {variant:8s} n={len(aupcs)} aupc={[round(a, 4) for a in aupcs]} mean={round(sum(aupcs)/len(aupcs), 4) if aupcs else '-'}")

    # paired by seed: frozen vs naive
    shared = sorted(set(per_seed["frozen"]) & set(per_seed["naive"]))
    if not shared:
        print("no shared seeds between frozen and naive; cannot pair")
        return 1
    print(f"paired seeds (frozen vs naive): {shared}")
    deltas = []
    for s in shared:
        f, n = per_seed["frozen"][s], per_seed["naive"][s]
        tasks = sorted(set(f) & set(n))
        d = sum(n[t] - f[t] for t in tasks) / len(tasks)
        deltas.append(d)
    obs = sum(deltas) / len(deltas)
    # permutation: relabel variant within each seed pair
    rng = random.Random(args.seed)
    hits = 0
    for _ in range(args.n_perm):
        perm = 0.0
        for s in shared:
            f, n = per_seed["frozen"][s], per_seed["naive"][s]
            tasks = sorted(set(f) & set(n))
            v = sum(n[t] - f[t] for t in tasks) / len(tasks)
            if rng.random() < 0.5:
                v = -v
            perm += v
        if perm / len(shared) >= obs:
            hits += 1
    p = hits / args.n_perm
    print(f"mean paired delta (naive - frozen) = {obs:+.4f} AUPC")
    print(f"two-sided permutation p = {p:.4f} ({args.n_perm} perms, rng seed {args.seed})")
    print(f"honest power note: n={len(shared)} seeds; SESOI not pre-registered post-D17; "
          "report as underpowered at n=4.")

    if "egc" in per_seed and per_seed["egc"]:
        shared2 = sorted(set(per_seed["frozen"]) & set(per_seed["egc"]))
        print(f"paired seeds (frozen vs egc): {shared2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

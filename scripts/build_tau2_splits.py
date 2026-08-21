"""Build tau2 role manifests by grouped hash partition (design doc §9.3).

Partition selected official splits into roles:
dev / calibration / adaptation / sentinel / candidate_audit / future = 20/15/30/10/10/15.
Tasks are first grouped by template/goal family (action-signature of evaluation
criteria), then partitioned with a public salt by grouped hash. The manifest is
machine-readable and stable across runs; roles are opaque to the online manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

PUBLIC_SALT = "agent-ttrl.tau2.role-partition.v1"

ROLE_PERCENT = {"dev": 0.20, "calibration": 0.15, "adaptation": 0.30,
                "sentinel": 0.10, "candidate_audit": 0.10, "future": 0.15}


def goal_family(task: dict) -> str:
    """Proxy for template/goal family: canonical set of action names."""
    actions = [a.get("name", "") for a in task.get("evaluation_criteria", {}).get("actions", [])]
    return hashlib.sha256(",".join(sorted(actions)).encode()).hexdigest()[:16]


def grouped_hash(task_id: str, family: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{family}:{task_id}".encode()).hexdigest(), 16)


def build(domain_dir: Path, split_name: str, out_dir: Path) -> dict:
    split_file = domain_dir / "split_tasks.json"
    tasks_file = domain_dir / "tasks.json"
    with open(split_file, encoding="utf-8") as f:
        splits = json.load(f)
    with open(tasks_file, encoding="utf-8") as f:
        tasks = {t["id"]: t for t in json.load(f)}

    ids = [str(i) for i in splits[split_name]]
    families: dict[str, list[str]] = defaultdict(list)
    for tid in ids:
        families[goal_family(tasks[tid])].append(tid)

    # whole-family assignment against global role targets (family integrity kept;
    # grouped-hash ordering decides family order and intra-family member order)
    total = len(ids)
    targets = _target_counts(total)
    roles: dict[str, list[str]] = {r: [] for r in ROLE_PERCENT}
    capacity = dict(targets)
    for family in sorted(families, key=lambda f: int(hashlib.sha256(f.encode()).hexdigest(), 16)):
        members = sorted(families[family], key=lambda t: grouped_hash(t, family, PUBLIC_SALT))
        target = max(capacity, key=lambda r: capacity[r])  # role with most remaining capacity
        roles[target].extend(members)
        capacity[target] -= len(members)

    manifest = {
        "schema_version": "agent-ttrl.tau2-role-manifest.v1",
        "tau2_commit": "a2c024725189473d2d7cea3a5cfdbcc67478e41f",
        "domain": domain_dir.name,
        "source_split": split_name,
        "salt": PUBLIC_SALT,
        "grouping": "action-signature goal family",
        "counts": {r: len(v) for r, v in roles.items()},
        "roles": roles,
        "content_hashes": {r: hashlib.sha256(",".join(sorted(v)).encode()).hexdigest()
                           for r, v in roles.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tau2_{domain_dir.name}_roles.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _target_counts(total: int) -> dict[str, int]:
    """Target role counts from global proportions; last role absorbs rounding."""
    targets = {r: int(total * p) for r, p in ROLE_PERCENT.items()}
    targets["future"] += total - sum(targets.values())
    return targets


def overlap_report(manifests: list[dict]) -> dict:
    """Template-family overlap across roles within each domain (design doc §8.3).

    Sentinel/audit/future must share no goal family with adaptation; the report
    is a gate input, not a decision.
    """
    report = {}
    for m in manifests:
        domain = m["domain"]
        role_families = {}
        for role, ids in m["roles"].items():
            role_families[role] = set(ids)
        protected = {r: set(role_families.get(r, [])) for r in ("sentinel", "candidate_audit", "future")}
        adapt = set(role_families.get("adaptation", []))
        overlaps = {r: sorted(protected[r] & adapt) for r in protected}
        report[domain] = {
            "counts": m["counts"],
            "overlap_with_adaptation": {r: len(v) for r, v in overlaps.items()},
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau2-root", default=r"C:\Users\w1828\AppData\Local\Temp\tau2-bench")
    ap.add_argument("--out", default=r"C:\Users\w1828\repos\agent-ttrl\protocols\splits")
    ap.add_argument("--domains", nargs="+", default=["retail", "telecom"])
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    root = Path(args.tau2_root) / "data" / "tau2" / "domains"
    manifests = []
    for dom in args.domains:
        m = build(root / dom, args.split, Path(args.out))
        manifests.append(m)
        print(f"{dom}[{args.split}]: " + " ".join(f"{r}={v}" for r, v in m["counts"].items()))
    print(overlap_report(manifests))


if __name__ == "__main__":
    main()

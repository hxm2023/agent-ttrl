"""Build AppWorld role manifests (design doc §9.2).

Role mapping (frozen): official train -> dev, official dev -> calibration,
official test_normal -> adaptation:sentinel:candidate_audit = 70:15:15
(grouped-hash partition by template family), official test_challenge ->
sealed future holdout.

Template family = the hex prefix of the task dir (AppWorld task IDs are
<family>_<variant>, e.g. 024c982_1/2/3 share the family 024c982). Partition
keeps whole families in one role (family integrity) against global targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PUBLIC_SALT = "agent-ttrl.appworld.role-partition.v1"
ROLE_PERCENT = {"adaptation": 0.70, "sentinel": 0.15, "candidate_audit": 0.15}
APPWORLD_COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"


def family_of(task_id: str) -> str:
    return task_id.split("_")[0]


def grouped_hash(task_id: str, family: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{family}:{task_id}".encode()).hexdigest(), 16)


def partition(test_normal_ids: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for tid in test_normal_ids:
        families.setdefault(family_of(tid), []).append(tid)
    total = len(test_normal_ids)
    targets = {r: int(total * p) for r, p in ROLE_PERCENT.items()}
    targets["candidate_audit"] += total - sum(targets.values())
    capacity = dict(targets)
    roles: dict[str, list[str]] = {r: [] for r in ROLE_PERCENT}
    for fam in sorted(families, key=lambda f: int(hashlib.sha256(f.encode()).hexdigest(), 16)):
        members = sorted(families[fam], key=lambda t: grouped_hash(t, fam, PUBLIC_SALT))
        target = max(capacity, key=lambda r: capacity[r])
        roles[target].extend(members)
        capacity[target] -= len(members)
    return roles


def data_manifest_sha256(data_dir: Path) -> str:
    """Hash of the pinned data release: split files + version + task dirs."""
    h = hashlib.sha256()
    for f in sorted((data_dir / "datasets").glob("*.txt")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    h.update((data_dir / "version.txt").read_bytes())
    task_ids = sorted(p.name for p in (data_dir / "tasks").iterdir() if p.is_dir())
    h.update(json.dumps(task_ids, separators=(",", ":")).encode())
    return h.hexdigest()


def build(data_dir: Path, out_dir: Path) -> dict:
    datasets = {s.name.replace(".txt", ""): s.read_text().splitlines()
                for s in (data_dir / "datasets").glob("*.txt")}
    roles = partition(datasets["test_normal"])
    manifest = {
        "schema_version": "agent-ttrl.appworld-role-manifest.v1",
        "source_repo": "stonybrooknlp/appworld",
        "source_commit": APPWORLD_COMMIT,
        "data_manifest_sha256": data_manifest_sha256(data_dir),
        "role_mapping": {"train": "dev", "dev": "calibration",
                         "test_normal": "adaptation:sentinel:candidate_audit=70:15:15",
                         "test_challenge": "sealed_future_holdout"},
        "counts": {
            "train_as_dev": len(datasets["train"]),
            "dev_as_calibration": len(datasets["dev"]),
            **{r: len(v) for r, v in roles.items()},
            "test_challenge_as_sealed_holdout": len(datasets["test_challenge"]),
        },
        "roles": roles,
        "test_challenge_ids": datasets["test_challenge"],
        "content_hashes": {r: hashlib.sha256(",".join(sorted(v)).encode()).hexdigest()
                           for r, v in roles.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "appworld_test_normal_roles.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"C:\Users\w1828\AppData\Local\Temp\appworld\data")
    ap.add_argument("--out", default=r"C:\Users\w1828\repos\agent-ttrl\protocols\splits")
    args = ap.parse_args()
    m = build(Path(args.data_dir), Path(args.out))
    print(json.dumps(m["counts"]))
    print("data_manifest_sha256:", m["data_manifest_sha256"])


if __name__ == "__main__":
    main()

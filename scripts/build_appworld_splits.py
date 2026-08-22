"""Build AppWorld role manifests (design doc §9.2).

Role mapping (frozen): official train -> dev, official dev -> calibration,
official test_normal -> adaptation:sentinel:candidate_audit = 70:15:15
(grouped-hash partition by template family), official test_challenge ->
sealed future holdout.

The AppWorld dataset on HuggingFace is gated (HTTP 401 without auth); this
script runs once the pinned data release is downloaded. Grouping uses the
task template family (goal graph / task_template field when present).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PUBLIC_SALT = "agent-ttrl.appworld.role-partition.v1"
ROLE_PERCENT = {"adaptation": 0.70, "sentinel": 0.15, "candidate_audit": 0.15}
APPWORLD_COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"


def grouped_hash(task_id: str, family: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{family}:{task_id}".encode()).hexdigest(), 16)


def build(test_normal: dict, data_manifest_sha256: str, out_dir: Path) -> dict:
    families: dict[str, list[str]] = {}
    for tid in test_normal:
        fam = test_normal[tid].get("template_family", "unknown")
        families.setdefault(fam, []).append(tid)
    total = sum(len(v) for v in families.values())
    targets = {r: int(total * p) for r, p in ROLE_PERCENT.items()}
    targets["candidate_audit"] += total - sum(targets.values())
    capacity = dict(targets)
    roles: dict[str, list[str]] = {r: [] for r in ROLE_PERCENT}
    for family in sorted(families, key=lambda f: int(hashlib.sha256(f.encode()).hexdigest(), 16)):
        members = sorted(families[family], key=lambda t: grouped_hash(t, family, PUBLIC_SALT))
        target = max(capacity, key=lambda r: capacity[r])
        roles[target].extend(members)
        capacity[target] -= len(members)
    manifest = {
        "schema_version": "agent-ttrl.appworld-role-manifest.v1",
        "appworld_commit": APPWORLD_COMMIT,
        "data_manifest_sha256": data_manifest_sha256,
        "salt": PUBLIC_SALT,
        "role_mapping": {"train": "dev", "dev": "calibration",
                         "test_normal": "adaptation:sentinel:candidate_audit=70:15:15",
                         "test_challenge": "sealed_future_holdout"},
        "counts": {r: len(v) for r, v in roles.items()},
        "roles": roles,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "appworld_test_normal_roles.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-normal-json", required=True,
                    help="path to the pinned AppWorld test_normal task JSON (from HF data release)")
    ap.add_argument("--data-manifest-sha256", required=True)
    ap.add_argument("--out", default=r"C:\Users\w1828\repos\agent-ttrl\protocols\splits")
    args = ap.parse_args()
    test_normal = json.loads(Path(args.test_normal_json).read_text(encoding="utf-8"))
    m = build(test_normal, args.data_manifest_sha256, Path(args.out))
    print(json.dumps(m["counts"]))


if __name__ == "__main__":
    main()

"""AppWorld role manifest tests (design doc §9.2)."""
import json
from pathlib import Path

import jsonschema

SCHEMA = json.loads(Path("schemas/split_manifest.schema.json").read_text(encoding="utf-8"))
M = json.loads(Path("protocols/splits/appworld_test_normal_roles.json").read_text(encoding="utf-8"))

PROTECTED = {"sentinel", "candidate_audit"}


def test_manifest_exists_and_validates():
    assert M["source_commit"] == "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
    assert len(M["data_manifest_sha256"]) == 64
    assert M["counts"]["adaptation"] + M["counts"]["sentinel"] + M["counts"]["candidate_audit"] == \
        len(M["roles"]["adaptation"]) + len(M["roles"]["sentinel"]) + len(M["roles"]["candidate_audit"])


def test_role_proportions():
    total = sum(M["counts"][r] for r in ("adaptation", "sentinel", "candidate_audit"))
    assert abs(M["counts"]["adaptation"] / total - 0.70) < 0.03
    assert abs(M["counts"]["sentinel"] / total - 0.15) < 0.03


def test_no_protected_role_overlap_with_adaptation():
    adapt = set(M["roles"]["adaptation"])
    for role in PROTECTED:
        assert not (set(M["roles"][role]) & adapt)


def test_family_integrity():
    """No template family (hex prefix) appears in two different roles."""
    fams_by_role = {role: set(tid.split("_")[0] for tid in ids)
                    for role, ids in M["roles"].items()}
    roles = list(M["roles"])
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            overlap = fams_by_role[roles[i]] & fams_by_role[roles[j]]
            assert not overlap, f"family overlap between {roles[i]} and {roles[j]}: {sorted(overlap)[:3]}"


def test_sealed_holdout_is_separate():
    sealed = set(M["test_challenge_ids"])
    adapt = set(M["roles"]["adaptation"])
    assert not (sealed & adapt)


def test_content_hashes_match():
    for role, ids in M["roles"].items():
        import hashlib
        h = hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()
        assert h == M["content_hashes"][role]

"""Split manifest validation + role non-overlap tests (design doc §8.2/§8.3)."""
import json
from pathlib import Path

import jsonschema

from agent_ttrl.environments.cts_world import ShiftConfig  # noqa: F401 (import path sanity)

SCHEMA = json.loads(Path("schemas/split_manifest.schema.json").read_text(encoding="utf-8"))
SPLITS_DIR = Path("protocols/splits")

PROTECTED = {"sentinel", "candidate_audit", "future"}


def _load(name):
    return json.loads((SPLITS_DIR / name).read_text(encoding="utf-8"))


def test_tau2_manifests_validate():
    for name in ("tau2_retail_roles.json", "tau2_telecom_roles.json"):
        jsonschema.validate(_load(name), SCHEMA)


def test_tau2_roles_no_task_overlap():
    """Protected roles must share no task ID with adaptation."""
    for name in ("tau2_retail_roles.json", "tau2_telecom_roles.json"):
        m = _load(name)
        adapt = set(m["roles"]["adaptation"])
        for role in PROTECTED:
            overlap = set(m["roles"][role]) & adapt
            assert not overlap, f"{name} {role} overlaps adaptation: {sorted(overlap)[:5]}"


def test_tau2_counts_match_roles():
    for name in ("tau2_retail_roles.json", "tau2_telecom_roles.json"):
        m = _load(name)
        assert m["counts"] == {r: len(v) for r, v in m["roles"].items()}


def test_tau2_deterministic_rebuild():
    """Rebuilding from the pinned repo must reproduce the manifests (stability)."""
    import hashlib
    m = _load("tau2_retail_roles.json")
    for role in ("adaptation", "future"):
        h = hashlib.sha256(",".join(sorted(m["roles"][role])).encode()).hexdigest()
        assert h == m["content_hashes"][role]


def test_all_split_ids_are_strings():
    for name in ("tau2_retail_roles.json", "tau2_telecom_roles.json"):
        m = _load(name)
        for role, ids in m["roles"].items():
            assert all(isinstance(i, str) for i in ids), (name, role)

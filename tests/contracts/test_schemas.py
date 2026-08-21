"""Contract schema validation: positive + negative fixtures (design doc §16.2/§17.3)."""
import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
SHA = "a" * 64
SHA2 = "b" * 64


def load(name):
    with open(SCHEMA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def validate(name, instance):
    jsonschema.validate(instance, load(name))


# ---------- positive fixtures ----------

def make_stream(extra=None):
    d = {
        "schema_version": "agent-ttrl.online-stream.v1",
        "stream_id": "s1",
        "domain_id": "appworld",
        "seed": 0,
        "reset_scope": "domain_seed",
        "adaptation_refs": [SHA],
        "split_hash": SHA2,
        "allowed_signal_classes": ["hard_evidence", "calibrated_soft_evidence"],
        "forbidden_signal_classes": ["hidden_outcome"],
    }
    if extra:
        d.update(extra)
    return d


def make_gate():
    return {
        "schema_version": "agent-ttrl.gate-manifest.v1",
        "candidate_stream_id": "s1",
        "opaque_sentinel_capability": "cap:sentinel:opaque",
        "opaque_anchor_capability": "cap:anchor:opaque",
        "max_candidates": 20,
        "max_exposure_per_item": 1,
        "n_gain": 64,
        "n_anchor": 64,
        "alpha_total": 0.05,
        "protocol_sha256": SHA,
    }


def make_sealed():
    return {
        "schema_version": "agent-ttrl.sealed-audit.v1",
        "opaque_candidate_audit_capability": "cap:audit:opaque",
        "opaque_future_holdout_capability": "cap:holdout:opaque",
        "unlock_condition": "FINAL_ADAPTER_AND_RUN_MANIFEST_SEALED",
        "recipient_public_key": "pubkey-0123456789abcdef",
    }


def make_evidence(hidden=None):
    d = {
        "schema_version": "agent-ttrl.evidence.v1",
        "trajectory_envelope_ref": SHA,
        "environment_state_before_sha256": SHA,
        "environment_state_after_sha256": SHA2,
        "hard_evidence": [
            {"producer": "receipt", "value": "reservation-ok", "unit": None, "allowed_in_gradient": True}
        ],
        "soft_evidence": [
            {"producer": "verifier-v1", "score": 0.8, "calibration_version": SHA, "allowed_in_gradient": True}
        ],
        "hidden_evaluator_ref": None,
        "calibration_profile_sha256": SHA,
        "missingness": {"soft_evidence": "present"},
        "cost_ledger_ref": SHA2,
    }
    if hidden is not None:
        d["hidden_evaluator_ref"] = hidden
    return d


def make_branch():
    return {
        "schema_version": "agent-ttrl.branch-record.v1",
        "branch_id": SHA,
        "parent_trajectory_ref": SHA,
        "decision_span_ref": SHA,
        "selection_probability": 0.25,
        "restore_snapshot_sha256": SHA,
        "behavior_policy_ref": SHA,
        "alternative_action_tokens_ref": SHA,
        "continuation_protocol_sha256": SHA,
        "coupling_seed": 42,
        "evidence_bundle_ref": SHA,
        "action_idx": 0,
        "group_id": "g1",
    }


def make_update_row():
    return {
        "schema_version": "agent-ttrl.update-row.v1",
        "row_id": SHA,
        "state_prefix_tokens_ref": SHA,
        "action_tokens_ref": SHA,
        "action_loss_mask_ref": SHA,
        "behavior_logprobs_ref": SHA,
        "advantage": 0.4,
        "decision_state_ref": SHA,
        "branch_group_ref": "g1",
        "policy_identity": {"base_sha256": SHA, "adapter_sha256": SHA2, "policy_version": "v1"},
        "evidence_and_cost_refs": [SHA, SHA2],
        "local_gate_kind": "reliability_t",
        "gate_passed": True,
    }


def make_decision(decision="COMMIT", commit_extra=True):
    d = {
        "schema_version": "agent-ttrl.candidate-adapter-decision.v1",
        "candidate_id": "c1",
        "parent_adapter_ref": SHA,
        "candidate_adapter_sha256": SHA2,
        "update_input_event_ref": SHA,
        "shadow_protocol_sha256": SHA,
        "gain_bound": {"lower": 0.03, "upper": 0.2, "alpha": 0.025, "n": 64},
        "risk_bound": {"lower": -0.1, "upper": 0.01, "alpha": 0.025, "n": 64},
        "guard_decision_ref": SHA,
        "decision": decision,
        "reason_codes": [],
    }
    if decision == "COMMIT" and commit_extra:
        d["fencing_epoch"] = 3
        d["commit_event_ref"] = SHA2
    return d


class TestPositiveFixtures:
    def test_stream(self):
        validate("online_stream.schema.json", make_stream())

    def test_gate(self):
        validate("gate_manifest.schema.json", make_gate())

    def test_sealed(self):
        validate("sealed_audit.schema.json", make_sealed())

    def test_evidence(self):
        validate("evidence_bundle.schema.json", make_evidence())

    def test_branch(self):
        validate("branch_record.schema.json", make_branch())

    def test_update_row(self):
        validate("update_row.schema.json", make_update_row())

    def test_decision_commit(self):
        validate("candidate_adapter_decision.schema.json", make_decision("COMMIT"))

    def test_decision_rollback(self):
        validate("candidate_adapter_decision.schema.json", make_decision("ROLLBACK"))


class TestNegativeFixtures:
    @pytest.mark.parametrize("mutate", [
        lambda d: d.update({"sentinel_ids": [1, 2, 3]}),          # capability leak: field must not exist
        lambda d: d.update({"split_hash": "short"}),              # bad sha
        lambda d: d.update({"allowed_signal_classes": ["hidden_outcome"]}),  # forbidden signal allowed
        lambda d: d.update({"schema_version": "agent-ttrl.online-stream.v2"}),  # unknown major version
        lambda d: d.pop("seed"),
    ])
    def test_stream_invalid(self, mutate):
        d = make_stream()
        mutate(d)
        with pytest.raises(jsonschema.ValidationError):
            validate("online_stream.schema.json", d)

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update({"max_exposure_per_item": 2}),         # reuse limit violated
        lambda d: d.update({"n_gain": 0}),
        lambda d: d.update({"alpha_total": 1.5}),
        lambda d: d.update({"protocol_sha256": "x" * 63}),
    ])
    def test_gate_invalid(self, mutate):
        d = make_gate()
        mutate(d)
        with pytest.raises(jsonschema.ValidationError):
            validate("gate_manifest.schema.json", d)

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update({"unlock_condition": "ANYTIME"}),      # early unlock forbidden
        lambda d: d.pop("recipient_public_key"),
    ])
    def test_sealed_invalid(self, mutate):
        d = make_sealed()
        mutate(d)
        with pytest.raises(jsonschema.ValidationError):
            validate("sealed_audit.schema.json", d)

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update({"hidden_evaluator_ref": "https://eval.example/score"}),  # hidden ref must be null
        lambda d: d.update({"hidden_evaluator_ref": SHA}),
        lambda d: d["soft_evidence"].append({"producer": "v", "score": 1.1, "calibration_version": SHA, "allowed_in_gradient": True}),
        lambda d: d.update({"environment_state_before_sha256": "zz"}),
    ])
    def test_evidence_invalid(self, mutate):
        d = make_evidence()
        mutate(d)
        with pytest.raises(jsonschema.ValidationError):
            validate("evidence_bundle.schema.json", d)

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update({"selection_probability": 0.0}),       # must be in (0,1]
        lambda d: d.update({"selection_probability": 1.5}),
        lambda d: d.update({"coupling_seed": 2 ** 70}),           # must be 64-bit
        lambda d: d.update({"coupling_seed": -1}),
    ])
    def test_branch_invalid(self, mutate):
        d = make_branch()
        mutate(d)
        with pytest.raises(jsonschema.ValidationError):
            validate("branch_record.schema.json", d)

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update({"action_tokens_ref": "raw-text-fallback"}),  # producer artifact only
        lambda d: d.update({"behavior_logprobs_ref": None}),
        lambda d: d["policy_identity"].pop("adapter_sha256"),
        lambda d: d.update({"local_gate_kind": "unknown_gate"}),
        lambda d: d.update({"advantage": 11.0}),
    ])
    def test_update_row_invalid(self, mutate):
        d = make_update_row()
        mutate(d)
        with pytest.raises(jsonschema.ValidationError):
            validate("update_row.schema.json", d)

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update({"decision": "COMMIT"}),                # COMMIT without fencing/commit event
        lambda d: d.update({"decision": "COMMIT", "fencing_epoch": 1}),  # missing commit_event
        lambda d: d.update({"decision": "ROLLBACK", "commit_event_ref": SHA}),  # rollback with authoritative commit
        lambda d: d.update({"decision": "QUARANTINE", "commit_event_ref": SHA}),
        lambda d: d.update({"decision": "APPROVE"}),              # unknown decision
    ])
    def test_decision_invalid(self, mutate):
        d = make_decision("ROLLBACK")
        mutate(d)
        with pytest.raises(jsonschema.ValidationError):
            validate("candidate_adapter_decision.schema.json", d)

    def test_no_unknown_major_version(self):
        with pytest.raises(jsonschema.ValidationError):
            validate("online_stream.schema.json", make_stream(extra={"schema_version": "agent-ttrl.online-stream.v9"}))


def test_all_schemas_load():
    for f in SCHEMA_DIR.glob("*.schema.json"):
        s = load(f.name)
        assert s["$schema"].endswith("draft/2020-12/schema")
        assert s.get("additionalProperties") is False

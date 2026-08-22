"""UpdateRow materialization (design doc §7.4/§7.6).

Each candidate action produces one immutable UpdateRow binding:
- producer token artifacts (state prefix, action tokens, action loss mask,
  behavior log-probs) — raw-text fallback is FORBIDDEN;
- signed advantage A_i from the local gate;
- policy identity (base_sha256, adapter_sha256, policy_version);
- evidence and cost refs.

Mask scope validation: the action loss mask must be exactly 1 on the producer
action span and 0 everywhere else (observation/prefix/tool-output/system/user
tokens can never be trained). Violations reject the whole batch
(REJECT / ACTION_MASK_SCOPE, fault F04-analog).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from agent_ttrl.credit.paired_credit import CreditRow


@dataclass
class ActionSpan:
    """Producer-authored token span for one action (never derived via str.find)."""
    producer: str
    start: int
    end: int            # exclusive
    token_ids_ref: str  # sha256 of the producer token artifact
    text_hash: str


@dataclass
class PolicyIdentity:
    base_sha256: str
    adapter_sha256: str
    policy_version: str


class UpdateRowMaterializer:
    def __init__(self, identity: PolicyIdentity, local_gate_kind: str):
        self.identity = identity
        self.local_gate_kind = local_gate_kind

    def materialize(self, prefix_tokens_ref: str, span: ActionSpan,
                    behavior_logprobs_ref: str, mask: list[int],
                    credit: CreditRow, evidence_refs: list[str],
                    cost_ledger_ref: str) -> dict:
        """Build one immutable UpdateRow; rejects on mask/identity violations."""
        if span.end <= span.start:
            raise ValueError("REJECT / EMPTY_ACTION_SPAN")
        if len(mask) <= span.end or len(mask) < span.end:
            raise ValueError("REJECT / MASK_LENGTH_MISMATCH")
        if not (all(m == 1 for m in mask[span.start:span.end])
                and all(m == 0 for m in mask[:span.start])
                and all(m == 0 for m in mask[span.end:])):
            raise ValueError("REJECT / ACTION_MASK_SCOPE")
        if not credit.gate_passed and credit.credit != 0.0:
            raise ValueError("REJECT / GATED_ROW_NONZERO_ADVANTAGE")

        row = {
            "schema_version": "agent-ttrl.update-row.v1",
            "row_id": self._row_id(prefix_tokens_ref, span, credit),
            "state_prefix_tokens_ref": prefix_tokens_ref,
            "action_tokens_ref": span.token_ids_ref,
            "action_loss_mask_ref": self._hash_mask(mask),
            "behavior_logprobs_ref": behavior_logprobs_ref,
            "advantage": credit.credit,
            "decision_state_ref": prefix_tokens_ref,
            "branch_group_ref": f"group-{span.producer}",
            "policy_identity": {
                "base_sha256": self.identity.base_sha256,
                "adapter_sha256": self.identity.adapter_sha256,
                "policy_version": self.identity.policy_version,
            },
            "evidence_and_cost_refs": evidence_refs + [cost_ledger_ref],
            "local_gate_kind": self.local_gate_kind,
            "gate_passed": credit.gate_passed,
            "text_fallback": False,
        }
        return row

    @staticmethod
    def _hash_mask(mask: list[int]) -> str:
        return hashlib.sha256("".join(str(m) for m in mask).encode()).hexdigest()

    @staticmethod
    def _row_id(prefix_ref: str, span: ActionSpan, credit: CreditRow) -> str:
        payload = json.dumps([prefix_ref, span.token_ids_ref, credit.credit],
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def validate_batch_identity(rows: list[dict], identity: PolicyIdentity) -> None:
    """All rows in a batch must share the sealed parent policy identity."""
    for r in rows:
        pid = r["policy_identity"]
        if (pid["base_sha256"], pid["adapter_sha256"], pid["policy_version"]) != (
                identity.base_sha256, identity.adapter_sha256, identity.policy_version):
            raise ValueError("REJECT / POLICY_IDENTITY_MISMATCH")

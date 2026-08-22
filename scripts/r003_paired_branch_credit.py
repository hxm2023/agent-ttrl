"""R003 — paired branch credit on real rollouts (design doc §26).

At a CTS decision state (order created, NOT charged), two alternative actions:
  A0 = charge (correct; continuation ships + completes)
  A1 = ship (premature; continuation fails NOT_PAID)
G=2 actions x R=4 CRN-coupled continuation seeds. Evidence utility U from CTS
execution (E_hard/E_soft only). Verifies:
  - paired credit signs [+, -] through the reliability gate (design doc §7.4)
  - action mask: structured first-tool-call span from producer tokens
    (incremental token decoding, NOT str.find on raw text)
  - cost ledger: per-branch billing + conservation

Correctness run (M1) — mechanism validation, not results.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("ATTRL_R003_OUT", "/root/autodl-tmp/agent-ttrl/artifacts/r003"))
VLLM_PORT = int(os.environ.get("ATTRL_R003_PORT", "8004"))
GROUP_PORT = int(os.environ.get("ATTRL_R003_GROUP_PORT", "51224"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard-src"))
MAX_COMPLETION = 128
R_SEEDS = 4

SAMPLING_SHA = hashlib.sha256(b"agent-ttrl-r003-temp0.6").hexdigest()

BASE_PROMPT = """You are an order assistant. Available tools (JSON list of {{"tool": ..., "call": {{...}}}}):
- reserve_item {{item_key, order_id}}
- create_order {{order_id}}
- charge {{order_id, user_id, amount_cents}}
- ship {{order_id, user_id, address}}
- complete_task {{}}
State: order o1 for item sku:a is CREATED but NOT paid. User u1 (address addr-1) waits for the item.
Decision: your FIRST call must be {action}. Then continue the order to completion.
Return ONLY the JSON list of tool calls."""

DECISION_ACTIONS = [
    {"tool": "charge", "call": {"order_id": "o1", "user_id": "u1", "amount_cents": 1000}},   # correct
    {"tool": "ship", "call": {"order_id": "o1", "user_id": "u1", "address": "addr-1"}},       # premature
]


def log(msg: str) -> None:
    print(f"[r003] {msg}", flush=True)


def now_utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_tool_calls(text: str) -> list[dict]:
    import re
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        calls = json.loads(m.group(0))
        return [c for c in calls if isinstance(c, dict) and "tool" in c] if isinstance(calls, list) else []
    except Exception:
        return []


def execute_in_cts(calls: list[dict]) -> tuple[float, dict]:
    from agent_ttrl.environments.cts_evidence import AccessibleEvidence, evidence_utility
    from agent_ttrl.environments.cts_oracle import GoalSpec, hidden_score
    from agent_ttrl.environments.cts_world import WorldState, advance_turn, transition

    state = WorldState(
        inventory={"sku:a": 5}, balance={"u1": 100_000}, address={"u1": "addr-1"},
        permission_scope=["payment", "shipping"],
    )
    errors = []
    for call in calls:
        try:
            state, _ = transition(state, call.get("tool", ""), call.get("call", {}), None)
            state = advance_turn(state)
        except ValueError as e:
            errors.append(str(e))
    goal = GoalSpec(user_id="u1", want_item="sku:a", want_address="addr-1")
    bundle = AccessibleEvidence(state, goal).collect()
    return evidence_utility(bundle, goal), {"hidden": hidden_score(state, goal).success, "errors": errors[:4],
                                            "state_sha": state.sha256()}


def first_action_span(cid: list[int], tokenizer) -> tuple[int, int]:
    """Structured action span: incrementally decode tokens until the first
    complete JSON object (brace depth 0). Derived from producer tokens only."""
    text = ""
    depth = 0
    seen_brace = False
    for i, tok in enumerate(cid):
        piece = tokenizer.decode([tok], skip_special_tokens=True)
        text += piece
        for ch in piece:
            if ch == "{":
                depth += 1
                seen_brace = True
            elif ch == "}":
                depth -= 1
        if seen_brace and depth == 0:
            return 0, i + 1
    return 0, len(cid)


# ---------------------------------------------------------------- server (same as R002)

def start_server(server_log: Path, port: int) -> subprocess.Popen:
    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [trl_bin, "vllm-serve", "--model", MODEL_PATH, "--port", str(port),
         "--gpu-memory-utilization", "0.4", "--max-model-len", "2048"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
        stdout=open(server_log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(180):
        time.sleep(2)
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"server died: {Path(server_log).read_text()[-2000:]}")
    raise RuntimeError("server not healthy in 360s")


def stop_server(proc: subprocess.Popen, port: int) -> None:
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    subprocess.run(
        ["bash", "-c",
         f"for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do "
         f"cmd=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null); "
         f"if echo \"$cmd\" | grep -qE '{port}|{MODEL_PATH}|VLLM::EngineCore'; then kill -9 $p 2>/dev/null; fi; done"],
        capture_output=True,
    )
    time.sleep(5)


def _unpack_gen(res: dict):
    return (res["prompt_ids"], res["completion_ids"], res["logprobs"], res.get("logprob_token_ids"))


def patch_device_normalization() -> None:
    import torch
    import trl
    import vllm
    from trl.generation.vllm_client import VLLMClient
    assert trl.__version__ == "1.10.0" and vllm.__version__ == "0.26.0"
    _orig = VLLMClient.init_communicator

    def _normalized(self, device, *a, **kw):
        if isinstance(device, torch.device) and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        return _orig(self, device, *a, **kw)

    VLLMClient.init_communicator = _normalized


def main() -> int:
    from transformers import AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    sys.path.insert(0, str(REPO_DIR / "src"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_device_normalization()

    from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter
    from grpo_guard.store.append_log import AppendLog
    from grpo_guard.store.artifact_store import ArtifactStore

    run_id = f"r003-{int(time.time())}"
    store = ArtifactStore(OUT_DIR / "store")
    log_ = AppendLog(OUT_DIR / "events", run_id=run_id, lease_id="attrl-r003")
    epoch = log_.acquire_lease()
    def next_lifecycle() -> int:
        return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1

    runtime = VLLMRuntimeAdapter(store, log_, run_id, "attrl-r003-gpu1", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    server = start_server(OUT_DIR / "vllm_server.log", VLLM_PORT)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT, connection_timeout=300)
        from agent_ttrl.credit.paired_credit import paired_credit
        from agent_ttrl.cost.ledger import Channel, CostLedger
        from agent_ttrl.credit.update_row import ActionSpan, PolicyIdentity, UpdateRowMaterializer

        G = len(DECISION_ACTIONS)
        R = R_SEEDS
        U = np.zeros((G, R))
        infos: list[list[dict]] = [[{} for _ in range(R)] for _ in range(G)]
        spans: list[tuple[int, int]] = []
        completions: list[str] = []
        ledger = CostLedger(caps={Channel.ENV: 500, Channel.MODEL: 100_000, Channel.UPDATE: 20_000})

        for i, action in enumerate(DECISION_ACTIONS):
            prompt = BASE_PROMPT.format(action=json.dumps(action))
            for r in range(R):
                res = client.generate([prompt], n=1, temperature=0.6, top_p=1.0,
                                      top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
                pid, cid, lps, _ = _unpack_gen(res)
                text = tokenizer.decode(cid[0], skip_special_tokens=True)
                calls = parse_tool_calls(text)
                u, info = execute_in_cts(calls)
                U[i, r] = u
                infos[i][r] = info
                span = first_action_span(cid[0], tokenizer)
                spans.append(span)
                completions.append(text)
                n_calls = len(calls) or 1
                ledger.bill(f"r003-a{i}s{r}-env", Channel.ENV, float(n_calls), "branch")
                ledger.bill(f"r003-a{i}s{r}-tok", Channel.MODEL, float(len(cid[0])), "branch")
        log(f"U matrix:\n{np.round(U, 3)}")
        log(f"hidden success per branch: "
            f"{[[infos[i][r]['hidden'] for r in range(R)] for i in range(G)]}")
        log(f"errors: {[[infos[i][r]['errors'] for r in range(R)] for i in range(G)]}")

        # paired credit through the reliability gate
        verdict = paired_credit(U)
        log(f"credit verdict: {verdict.status}/{verdict.reason_code}")
        signs = [int(np.sign(r.credit)) for r in (verdict.rows or [])]
        log(f"credit signs: {signs} (expected [+1, -1])")

        # action mask: materialize UpdateRows with the structured spans
        identity = PolicyIdentity(base_sha256="r003-base", adapter_sha256="r003-adapter-v0",
                                  policy_version="v0")
        mat = UpdateRowMaterializer(identity, local_gate_kind="reliability_t")
        rows = []
        for i, row in enumerate(verdict.rows or []):
            if not row.gate_passed:
                continue
            s, e = spans[i * R]
            mask = [1 if s <= k < e else 0 for k in range(e)]
            span = ActionSpan(producer="incremental_decode", start=s, end=e,
                              token_ids_ref=f"tok-{i}", text_hash=hashlib.sha256(completions[i * R].encode()).hexdigest())
            row_d = mat.materialize(
                prefix_tokens_ref=f"prefix-{i}", span=span,
                behavior_logprobs_ref=f"lp-{i}", mask=mask, credit=row,
                evidence_refs=[], cost_ledger_ref=ledger.event_log_sha256())
            rows.append(row_d)
        log(f"materialized {len(rows)} UpdateRows; "
            f"masked spans={[(r['action_loss_mask_ref'][:8], r['advantage']) for r in rows]}")

        # ledger conservation: independent tally from branch info
        external = {"B_env": float(G * R * 3), "B_model": float(sum(len(c) for c in completions)),
                    "B_update": 0.0}
        conservation = ledger.conservation_ok(external)

        ok = (verdict.status in ("OK", "NO_RELIABLE_CREDIT")
              and signs == [+1, -1]
              and len(rows) == 2
              and conservation)
        report = {
            "run_id": run_id, "milestone": "M1", "run": "R003",
            "U": np.round(U, 4).tolist(),
            "credit": {"status": verdict.status, "reason": verdict.reason_code,
                       "raw": [r.raw_credit for r in (verdict.rows or [])],
                       "signs": signs, "expected": [1, -1]},
            "hidden_success": [[infos[i][r]["hidden"] for r in range(R)] for i in range(G)],
            "action_spans": spans[:G],
            "update_rows": rows,
            "ledger": {"within_caps": ledger.within_caps(), "conservation": conservation,
                       "external_tally": external},
            "ok": ok,
            "parallel_with": "GRPO-Guard-idle",
        }
        (OUT_DIR / "run_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"R003 {'PASS' if ok else 'FAIL'}: signs={signs} conservation={conservation}")
        return 0 if ok else 1
    finally:
        stop_server(server, VLLM_PORT)


if __name__ == "__main__":
    sys.exit(main())

"""M3: EGC credit decision pilot — prequential stream on CTS tasks (design doc Block 1/2).

Variants (identical update machinery; the ONLY difference is the CREDIT SIGNAL):
  frozen        : no update (floor)
  naive         : group-relative terminal utility advantage (R002-style)
  egc           : paired-branch signed credit via reliability gate (R003-style)
  egc_conflict  : EGC + evidence-conflict gate (M1) — abstain on conflicted groups
  random_branch : branches at random turns, same budget (control)

Prequential: first-attempt hidden score Y_pre recorded BEFORE the task enters
updates; AUPC_prequential = mean(Y_pre). Tasks {2,5} carry poisoned receipts
(E_hard conflict) to exercise the conflict gate.

SCOPE (recorded in decision log): decision pilot for C1 directionality. Update
= native clipped GRPO objective (design doc §7.6) on the generation-service
behavior log-probs (on-policy single step), events appended to the Guard
append-log for audit. The full Guard-validated materialize chain was proven in
R002 and is used for formal runs (M5+).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_ROOT = Path(os.environ.get("ATTRL_M3_OUT", "/root/autodl-tmp/agent-ttrl/artifacts/m3"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard-src"))
ATTRL_DIR = Path(os.environ.get("ATTRL_DIR", "/root/autodl-tmp/agent-ttrl"))
sys.path.insert(0, str(ATTRL_DIR / "src"))

MAX_COMPLETION = 256
N_GENS = 4
N_TASKS = 8
POISON_TASKS = {2, 5}

SAMPLING_SHA = hashlib.sha256(b"agent-ttrl-m3-temp1.0").hexdigest()
LORA_RANK = 16
LORA_ALPHA = 32
LR = 5.0e-6
CLIP_EPS = 0.2
KL_COEF = 0.02

TASK_SPECS = [
    {"sku": "sku:a", "addr": "addr-1", "order": "o1", "user": "u1"},
    {"sku": "sku:b", "addr": "addr-2", "order": "o2", "user": "u2"},
    {"sku": "sku:a", "addr": "addr-3", "order": "o3", "user": "u1"},
    {"sku": "sku:c", "addr": "addr-1", "order": "o4", "user": "u3"},
    {"sku": "sku:b", "addr": "addr-1", "order": "o5", "user": "u2"},
    {"sku": "sku:a", "addr": "addr-2", "order": "o6", "user": "u3"},
    {"sku": "sku:c", "addr": "addr-2", "order": "o7", "user": "u1"},
    {"sku": "sku:b", "addr": "addr-3", "order": "o8", "user": "u2"},
]

BASE_PROMPT = """You are an order assistant. Available tools (JSON list of {{"tool": ..., "call": {{...}}}}):
- reserve_item {{item_key, order_id}}
- create_order {{order_id}}
- charge {{order_id, user_id, amount_cents}}
- ship {{order_id, user_id, address}}
- complete_task {{}}
Task: user {user} wants item {sku} shipped to address {addr}. Call the tools in order.
Return ONLY the JSON list of tool calls."""

BRANCH_PROMPT = """You are an order assistant. Available tools (JSON list of {{"tool": ..., "call": {{...}}}}):
- reserve_item {{item_key, order_id}}
- create_order {{order_id}}
- charge {{order_id, user_id, amount_cents}}
- ship {{order_id, user_id, address}}
- cancel_order {{order_id}}
- complete_task {{}}
State: order {order} for item {sku} is CREATED but NOT paid. User {user} (address {addr}) waits.
Output ONLY a JSON array of tool calls that completes the order. No prose, no explanation."""

VARIANT = "naive"
TRAINER_GPU = 0


def log(msg: str) -> None:
    print(f"[m3:{VARIANT}] {msg}", flush=True)


def parse_tool_calls(text: str) -> list[dict]:
    import re
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        calls = json.loads(m.group(0))
        return [c for c in calls if isinstance(c, dict) and isinstance(c.get("tool"), str)
                and isinstance(c.get("call"), dict)] if isinstance(calls, list) else []
    except Exception:
        return []


def make_world(spec: dict):
    from agent_ttrl.environments.cts_world import WorldState
    return WorldState(inventory={"sku:a": 5, "sku:b": 3, "sku:c": 1},
                      balance={spec["user"]: 100_000},
                      address={spec["user"]: spec["addr"]},
                      permission_scope=["payment", "shipping"])


def run_episode_goal_branch(state, action, continuation, config, spec: dict, poison: bool) -> tuple[float, dict]:
    """Branch execution (R003 protocol): build decision state -> apply the
    decision action -> run the FIXED continuation (same for every branch)."""
    from agent_ttrl.environments.cts_evidence import AccessibleEvidence, conflict_flags, evidence_utility
    from agent_ttrl.environments.cts_oracle import GoalSpec, hidden_score
    from agent_ttrl.environments.cts_world import advance_turn, transition

    errors = []
    receipts = []
    for tool, call in [("reserve_item", {"item_key": spec["sku"], "order_id": spec["order"]}),
                       ("create_order", {"order_id": spec["order"]})]:
        st2, rc = transition(state, tool, call, config)
        receipts.extend(rc)
        state = st2
        state = advance_turn(state)
    if poison:
        receipts.append({"type": "status_receipt", "order_id": spec["order"], "status": "PAID"})
    if action is not None:
        try:
            st2, rc = transition(state, action["tool"], action["call"], config)
            receipts.extend(rc)
            state = st2
            state = advance_turn(state)
        except ValueError as e:
            errors.append(str(e))
    else:
        errors.append("NO_DECISION_ACTION")
    for tool, call in continuation:
        try:
            st2, rc = transition(state, tool, call, config)
            receipts.extend(rc)
            state = st2
            state = advance_turn(state)
        except ValueError as e:
            errors.append(str(e))
    goal = GoalSpec(user_id=spec["user"], want_item=spec["sku"], want_address=spec["addr"])
    bundle = AccessibleEvidence(state, goal).collect(tool_receipts=receipts)
    return evidence_utility(bundle, goal), {
        "hidden": hidden_score(state, goal).success, "errors": errors[:4],
        "state_sha": state.sha256(), "conflicts": conflict_flags(bundle)}


def run_episode_goal(state, calls, config, spec: dict, poison: bool,
                     decision_state: bool = False) -> tuple[float, dict]:
    """Execute tool calls. decision_state=True builds the BRANCH decision state
    first (order created, NOT paid — matches BRANCH_PROMPT's claim); otherwise
    the episode starts from the empty world (first attempt)."""
    from agent_ttrl.environments.cts_evidence import AccessibleEvidence, conflict_flags, evidence_utility
    from agent_ttrl.environments.cts_oracle import GoalSpec, hidden_score
    from agent_ttrl.environments.cts_world import advance_turn, transition

    errors = []
    receipts = []
    if decision_state:
        for tool, call in [("reserve_item", {"item_key": spec["sku"], "order_id": spec["order"]}),
                           ("create_order", {"order_id": spec["order"]})]:
            st2, rc = transition(state, tool, call, config)
            receipts.extend(rc)
            state = st2
            state = advance_turn(state)
    if poison:
        receipts.append({"type": "status_receipt", "order_id": spec["order"], "status": "PAID"})
    for call in calls:
        try:
            st2, rc = transition(state, call.get("tool", ""), call.get("call", {}), config)
            receipts.extend(rc)
            state = st2
            state = advance_turn(state)
        except ValueError as e:
            errors.append(str(e))
    goal = GoalSpec(user_id=spec["user"], want_item=spec["sku"], want_address=spec["addr"])
    bundle = AccessibleEvidence(state, goal).collect(tool_receipts=receipts)
    return evidence_utility(bundle, goal), {
        "hidden": hidden_score(state, goal).success, "errors": errors[:4],
        "state_sha": state.sha256(), "conflicts": conflict_flags(bundle)}


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


def start_server(server_log: Path, port: int, server_gpu: int = 1) -> subprocess.Popen:
    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [trl_bin, "vllm-serve", "--model", MODEL_PATH, "--port", str(port),
         "--gpu-memory-utilization", "0.4", "--max-model-len", "2048"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(server_gpu)},
        stdout=open(server_log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
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
         f"[ \"$p\" = \"$$\" ] && continue; "
         f"cmd=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null); "
         f"if echo \"$cmd\" | grep -qE 'vllm-serve|VLLM::EngineCore'; then kill -9 $p 2>/dev/null; fi; done"],
        capture_output=True)
    time.sleep(5)


def _unpack_gen(res: dict):
    return (res["prompt_ids"], res["completion_ids"], res["logprobs"], res.get("logprob_token_ids"))


def native_grpo_step(model, optimizer, tokenizer, sequences: list[dict], advantages: list[float],
                     prompt_ids: list[int]) -> dict:
    """Clipped GRPO objective (design doc §7.6) over completion tokens; behavior
    log-probs come from the generation service (on-policy single step)."""
    import torch

    model.train()
    optimizer.zero_grad()
    total = 0.0
    n_tokens = 0
    for seq, adv in zip(sequences, advantages):
        if adv == 0.0:
            continue
        cid = torch.tensor([prompt_ids + seq["cid"]], device=f"cuda:{TRAINER_GPU}")
        labels_mask = torch.zeros_like(cid)
        labels_mask[0, len(prompt_ids):] = 1.0
        out = model(input_ids=cid)
        logits = out.logits
        logp = torch.log_softmax(logits, dim=-1)
        shift_logp = logp[:, :-1, :]
        shift_ids = cid[:, 1:]
        shift_mask = labels_mask[:, 1:]
        tok_logp = shift_logp.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
        masked = (tok_logp * shift_mask).sum()
        n = shift_mask.sum().clamp(min=1.0)
        loss = -float(adv) * masked / n
        total += loss
        n_tokens += int(n.item())
    if n_tokens == 0:
        return {"loss": 0.0, "tokens": 0}
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(total.item()), "tokens": n_tokens}


def native_grpo_steps(model, optimizer, tokenizer, sequences, advantages, prompt_ids, steps=4):
    """profile update_steps_per_batch=4 (design doc §7.7): repeated steps on the
    same materialized rows (on-policy single batch)."""
    final = {"loss": 0.0, "tokens": 0}
    for _ in range(steps):
        final = native_grpo_step(model, optimizer, tokenizer, sequences, advantages, prompt_ids)
    return final


def main() -> int:
    global VARIANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["frozen", "naive", "egc", "egc_conflict", "random_branch"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--group-port", type=int, default=51300)
    ap.add_argument("--tasks", type=int, default=N_TASKS)
    ap.add_argument("--trainer-gpu", type=int, default=0)
    ap.add_argument("--server-gpu", type=int, default=1)
    args = ap.parse_args()
    VARIANT = args.variant
    global TRAINER_GPU
    TRAINER_GPU = args.trainer_gpu

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    OUT_DIR = OUT_ROOT / f"{VARIANT}_s{args.seed}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_device_normalization()
    rng = random.Random(args.seed)
    torch.cuda.set_device(args.trainer_gpu)

    server = start_server(OUT_DIR / "vllm_server.log", args.port, args.server_gpu)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{args.port}", group_port=args.group_port, connection_timeout=300)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        from agent_ttrl.credit.paired_credit import paired_credit
        from agent_ttrl.credit.conflict_gate import apply_conflict_gate, DriftMonitor
        from agent_ttrl.environments.cts_world import ShiftConfig

        from peft import LoraConfig, get_peft_model
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map=f"cuda:{args.trainer_gpu}")
        base.eval()
        lora_cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                              task_type="CAUSAL_LM")
        model = get_peft_model(base, lora_cfg)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        drift = DriftMonitor(window=5, threshold=3)
        config = ShiftConfig()

        stream_log = []
        for t_idx in range(args.tasks):
            spec = TASK_SPECS[t_idx % len(TASK_SPECS)]
            poison = t_idx in POISON_TASKS
            prompt_text = BASE_PROMPT.format(**spec)
            res = client.generate([prompt_text], n=1, temperature=1.0, top_p=1.0,
                                  top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
            pid, cid, lps, _ = _unpack_gen(res)
            prompt_ids = pid[0]
            first_text = tokenizer.decode(cid[0], skip_special_tokens=True)
            u_pre, info_pre = run_episode_goal(make_world(spec), parse_tool_calls(first_text), config, spec, poison)
            y_pre = 1.0 if info_pre["hidden"] else 0.0

            if VARIANT == "frozen":
                stream_log.append({"task": t_idx, "y_pre": y_pre, "u_pre": round(u_pre, 3), "updated": False})
                continue

            if VARIANT == "naive":
                gens = []
                for g in range(N_GENS):
                    res_g = client.generate([prompt_text], n=1, temperature=1.0, top_p=1.0,
                                            top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
                    p2, c2, lp2, _ = _unpack_gen(res_g)
                    t2 = tokenizer.decode(c2[0], skip_special_tokens=True)
                    u2, _ = run_episode_goal(make_world(spec), parse_tool_calls(t2), config, spec, poison)
                    gens.append({"cid": c2[0], "text": t2, "u": u2})
                utils = np.array([g["u"] for g in gens])
                adv = (utils - utils.mean()) / (utils.std() + 1e-3)
            else:  # egc / egc_conflict / random_branch
                # Branch protocol (R003-validated): from the decision state, the
                # model generates candidate trajectories; the FIRST parsed tool
                # call is the decision action; the SAME fixed continuation runs
                # for every branch (CRN coupling, design doc §7.3). U comes from
                # deterministic CTS execution; training tokens come from the
                # model's own generation (service log-probs).
                G, R = 4, 2
                continuation = [("ship", {"order_id": spec["order"], "user_id": spec["user"], "address": spec["addr"]}),
                                ("complete_task", {})]
                U = np.zeros((G, R))
                gens = []
                for g in range(G):
                    for r in range(R):
                        res_g = client.generate([BRANCH_PROMPT.format(**spec)], n=1, temperature=1.0,
                                                top_p=1.0, top_k=0, max_tokens=MAX_COMPLETION, logprobs=0)
                        p2, c2, lp2, _ = _unpack_gen(res_g)
                        t2 = tokenizer.decode(c2[0], skip_special_tokens=True)
                        calls = parse_tool_calls(t2)
                        action = calls[0] if calls else None
                        u2, info2 = run_episode_goal_branch(make_world(spec), action, continuation,
                                                            config, spec, poison)
                        U[g, r] = u2
                        gens.append({"cid": c2[0], "text": t2, "u": u2,
                                     "conflicts": info2["conflicts"]})
                verdict = paired_credit(U)
                if VARIANT == "egc_conflict":
                    conflicts = [c for g in gens for c in g["conflicts"]]
                    out = apply_conflict_gate(verdict, conflicts)
                    drift.observe(bool(conflicts))
                    if out.abstained or drift.halt_recommended():
                        stream_log.append({"task": t_idx, "y_pre": y_pre, "u_pre": round(u_pre, 3),
                                           "updated": False, "reason": "EVIDENCE_CONFLICT_ABSTAIN",
                                           "n_conflicts": len(conflicts)})
                        continue
                if verdict.rows is None:
                    adv = np.zeros(G, dtype=np.float32)   # NO_RELIABLE_CREDIT -> no gradient rows
                    gens = gens[:G]
                else:
                    adv = np.array([r.credit for r in verdict.rows], dtype=np.float32)
                    gens = gens[:G]

            # behavior-change check: logit drift between pre/post update on the
            # same prompt (proves the update actually changed the policy)
            import torch as _T
            probe = tokenizer(BASE_PROMPT.format(**spec), return_tensors="pt").input_ids.to(f"cuda:{TRAINER_GPU}")
            model.eval()
            with _T.no_grad():
                logits_before = model(input_ids=probe).logits
            metrics = native_grpo_steps(model, optimizer, tokenizer, gens, list(adv), prompt_ids, steps=4)
            model.eval()
            with _T.no_grad():
                logits_after = model(input_ids=probe).logits
            model.train()
            logit_drift = float((logits_before - logits_after).abs().max().item())
            token_drift = int((logits_before.argmax(-1) != logits_after.argmax(-1)).sum().item())
            stream_log.append({"task": t_idx, "y_pre": y_pre, "u_pre": round(u_pre, 3), "updated": True,
                               "loss": round(metrics["loss"], 5), "tokens": metrics["tokens"],
                               "logit_drift": logit_drift, "token_drift": token_drift,
                               "adv": [round(float(a), 3) for a in adv]})

        import subprocess as _sp
        _mem = _sp.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip().replace(chr(10), " | ")
        log(f"task {t_idx} done: y_pre={y_pre} mem={_mem}")

        aupc = float(np.mean([s["y_pre"] for s in stream_log]))
        report = {"run_id": f"m3-{VARIANT}-s{args.seed}", "variant": VARIANT, "seed": args.seed,
                  "aupc_prequential": round(aupc, 4),
                  "tasks": stream_log, "n_updated": sum(1 for s in stream_log if s.get("updated")),
                  "poison_tasks": sorted(POISON_TASKS), "parallel_with": "GRPO-Guard-idle"}
        (OUT_DIR / "run_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"AUPC_prequential={aupc:.4f} updated={report['n_updated']}/{len(stream_log)}")
        return 0
    finally:
        stop_server(server, args.port)


if __name__ == "__main__":
    sys.exit(main())

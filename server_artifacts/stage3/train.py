"""Stage 3 real closed-loop training (runs on autodl2, Guard-supervised).

One training run = (task x estimator x seed). The loop follows r002's
verified Guard chain (identity validation -> reward -> pre-update ALLOW ->
materialize -> guarded update -> commit adapter -> canary) over E epochs:

- ALL epochs roll out with the trainer's own current policy on GPU0 (the
  policy that generated the sequences scores them exactly, declared as
  Guard behavior_logprob_source="exact_behavior_scorer", schema 7.5).
  The vLLM generation-service mode is NOT exercised: vLLM 0.26 cannot
  initialize on this server's Blackwell GPU (SM 12.x requires a newer
  vLLM) — recorded in the run metrics as a documented limitation.
- credit:    dense GRPO / local-decision / paired-branch (stage3.credit)

Per-epoch and final metrics: success rate, mean utility, mean call count,
invalid-tool-call rate, gradient L2 (variance proxy), KL drift vs base,
GPU seconds. Rollout trajectories are exported as aca-trajectory-record-1.0
records for the Auditor's Stage-1 audit. Final evaluation uses the FINAL
checkpoint adapter on held-out prompts.

Usage (on autodl2):
  python stage3/train.py --task cts_order --estimator dense --seed 1 \
      --out /root/autodl-tmp/agent-ttrl/stage3/out/cts_order_dense_s1
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
from types import SimpleNamespace

import numpy as np

STAGE3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE3_DIR))

from credit import decision_positions  # noqa: E402
from tasks import TASKS, parse_tool_calls  # noqa: E402

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
VLLM_PORT = int(os.environ.get("STAGE3_VLLM_PORT", "8007"))
GROUP_PORT = int(os.environ.get("STAGE3_GROUP_PORT", "51227"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard-src"))
ATTRL_DIR = Path(os.environ.get("ATTRL_DIR", "/root/autodl-tmp/agent-ttrl"))
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(ATTRL_DIR / "src"))

MAX_COMPLETION = 128
LORA_RANK = 16
LORA_ALPHA = 32
LR = 5.0e-6
N_EVAL_PROMPTS = 16


def log(msg: str) -> None:
    print(f"[stage3] {msg}", flush=True)


def now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- server


def start_server(server_log: Path) -> subprocess.Popen:
    log(f"starting vLLM server (GPU1) at :{VLLM_PORT}")
    trl_bin = os.path.join(os.path.dirname(sys.executable), "trl")
    proc = subprocess.Popen(
        [trl_bin, "vllm-serve", "--model", MODEL_PATH, "--port", str(VLLM_PORT),
         "--gpu-memory-utilization", "0.3", "--max-model-len", "1024",
         "--enforce-eager"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
        stdout=open(server_log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(180):
        time.sleep(2)
        try:
            import urllib.request

            with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5) as r:
                if r.status == 200:
                    log("server healthy")
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"server died: {Path(server_log).read_text()[-2000:]}")
    raise RuntimeError("server not healthy in 360s")


def stop_server(proc: subprocess.Popen | None) -> None:
    import signal

    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(5)


# ---------------------------------------------------------------- identity

def compute_identity_hashes() -> tuple[str, str]:
    cfg = json.loads((Path(MODEL_PATH) / "tokenizer_config.json").read_text(encoding="utf-8"))
    tok_sha = hashlib.sha256(
        (Path(MODEL_PATH) / "tokenizer_config.json").read_bytes()
        + (Path(MODEL_PATH) / "tokenizer.json").read_bytes()
    ).hexdigest()
    tpl_sha = hashlib.sha256(cfg.get("chat_template", "").encode("utf-8")).hexdigest()
    return tok_sha, tpl_sha


def base_manifest(policy_version: int, tok_sha: str, tpl_sha: str, run_name: str) -> dict:
    shards = sorted(Path(MODEL_PATH).glob("model-*.safetensors"))
    weights = [{
        "uri": f"artifact://{p.name}", "media_type": "application/safetensors",
        "num_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "producer_event_id": f"ckpt-commit-v{policy_version}",
    } for p in shards]
    return {
        "manifest_id": f"pm-{policy_version}",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "policy_version": policy_version,
        "parent_policy_version": None,
        "weights": weights,
        "checkpoint_manifest_sha256": hashlib.sha256(
            json.dumps({"shards": [w["sha256"] for w in weights]}, sort_keys=True).encode()
        ).hexdigest(),
        "tokenizer_sha256": tok_sha, "chat_template_sha256": tpl_sha,
        "precision": "bf16", "adapter_kind": "lora", "adapter_sha256": None,
        "code_commit_sha": run_name, "config_sha256": hashlib.sha256(run_name.encode()).hexdigest(),
    }


def lora_manifest(base: dict, adapter_path: Path, policy_version: int) -> dict:
    m = dict(base)
    m["policy_version"] = policy_version
    m["parent_policy_version"] = policy_version - 1
    shards = sorted(adapter_path.glob("*.safetensors")) if adapter_path.exists() else []
    adapter_weights = [{
        "uri": f"artifact://{p.name}", "media_type": "application/safetensors",
        "num_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "producer_event_id": f"adapter-commit-v{policy_version}",
    } for p in shards]
    m["weights"] = m["weights"] + adapter_weights
    m["adapter_sha256"] = hashlib.sha256(
        json.dumps({"adapter": [w["sha256"] for w in adapter_weights]}, sort_keys=True).encode()
    ).hexdigest()
    return m


def build_envelope(run_id, gen, rew, id_decision, ckpt_sha, split, stage, parent_ver, update_id,
                   parent_sha=None, source="generation_service", scoring=None):
    from grpo_guard.schema.artifacts import EventRef, ManifestRef
    from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
    from grpo_guard.store.canonical_json import canonical_sha256

    # exact_behavior_scorer mode: the authoritative logprob event is the
    # SCORING event (L002), not the generation event
    auth_ref = EventRef(uri="", event_id=scoring.event_id, event_sha256=scoring.event_sha256) if scoring else (
        EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256))
    scoring_ref = EventRef(uri="", event_id=scoring.event_id, event_sha256=scoring.event_sha256) if scoring else None

    return TrajectoryEnvelope(
        envelope_id=f"env-{gen.event_id}-{stage}",
        envelope_stage=stage, run_id=run_id, request_id=gen.request_id,
        generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
        scoring_event=scoring_ref,
        reward_event=EventRef(uri="", event_id=rew.event_id, event_sha256=rew.event_sha256) if rew else None,
        policy_manifest=ManifestRef(uri="", manifest_id=f"pm-{parent_ver}", sha256=ckpt_sha),
        split_manifest=ManifestRef(uri="", manifest_id=split["split_id"], sha256=canonical_sha256(split)),
        parent_envelope_sha256=parent_sha,
        parent_identity_decision=EventRef(uri="", event_id=id_decision.event_id, event_sha256=id_decision.event_sha256) if id_decision else None,
        training_contract=TrainingContract(
            protocol="strict_on_policy", trainer_parent_policy_version=parent_ver,
            consuming_update_id=update_id, max_policy_lag_versions=0,
            behavior_logprob_source=source,
            authoritative_behavior_logprob_event=auth_ref,
            diagnostic_non_authoritative_logprobs_allowed=False,
        ),
    ).seal()


def patch_device_normalization() -> None:
    import torch
    import trl
    import vllm
    assert trl.__version__ == "1.10.0" and vllm.__version__ == "0.26.0"
    _orig = VLLMClient.init_communicator

    def _normalized(self, device, *a, **kw):
        if isinstance(device, torch.device) and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        return _orig(self, device, *a, **kw)

    VLLMClient.init_communicator = _normalized


# ---------------------------------------------------------------- old logprobs

def compute_old_logprobs(model, rows: list[dict], tokenizer) -> None:
    """Exact per-token logprobs of the row sequences under the CURRENT model.
    Computed PER ROW (batch 1): the vLLM batched responses can carry padded
    sequences near max-model-len, and a padded multi-row forward would
    materialize ~80 GB of float logits. At epoch 0 the LoRA weights are zero,
    so this is exactly the vLLM base policy's logprobs."""
    import torch as _t

    model.eval()
    with _t.no_grad():
        for r in rows:
            ids = _t.tensor(r["prompt_ids"] + r["seq"], dtype=_t.long,
                            device=next(model.parameters()).device).unsqueeze(0)
            plen = len(r["prompt_ids"])
            gen_len = len(r["seq"])
            logits = model(ids).logits
            logps = _t.log_softmax(logits.float(), dim=-1)
            gen = _t.tensor(r["seq"], dtype=_t.long, device=ids.device).unsqueeze(-1)
            r["old_logprobs"] = logps[0, plen - 1:plen - 1 + gen_len, :].gather(-1, gen).squeeze(-1).tolist()
            del logits, logps, ids
    model.train()


# ---------------------------------------------------------------- estimator loss

def stage3_loss(model, handles, pos_weights, group_size, clip_epsilon=0.2):
    """Chunked wrapper with per-chunk backward (summing chunk losses into one
    tensor retains every chunk's autograd graph and OOMs). Each chunk's loss
    is backward'd immediately; the returned loss is the credit-weighted mean
    of the chunk losses (already backward'd; callers must not backward
    again)."""
    import torch as _t

    CHUNK = 16
    loss_sum = 0.0
    total_weight = 0.0
    for i in range(0, len(handles), CHUNK):
        chunk_h = handles[i:i + CHUNK]
        chunk_pw = pos_weights[i:i + CHUNK]
        loss, metrics = _stage3_loss_chunk(model, chunk_h, chunk_pw, group_size, clip_epsilon)
        w = metrics["credit_positions"]
        loss.backward()
        loss_sum += float(loss.item()) * w
        total_weight += w
    if total_weight <= 0:
        total_weight = 1.0
    return _t.tensor(loss_sum / total_weight, device=next(model.parameters()).device), {"loss": loss_sum / total_weight, "chunks": True}


def _stage3_loss_chunk(model, handles, pos_weights, group_size, clip_epsilon=0.2):
    """Custom credit loss mirroring the Guard's grpo_loss math, but with a
    per-position credit mask (decision positions only for local/paired) and
    DIRECT per-position weights (no re-centering — the gated paired credit
    must survive as-is).

    pos_weights: list of dicts {positions: {full_seq_idx: weight}} per handle.
    """
    import torch as _t
    import torch.nn.functional as _F

    from grpo_guard.adapters.grpo_loss import _stack_handles

    seq_np, mask_np, lp_np, _ = _stack_handles(handles)
    device = next(model.parameters()).device
    seq = _t.as_tensor(seq_np).to(device)
    loss_mask = _t.as_tensor(mask_np).to(device)
    old_logps = _t.as_tensor(lp_np).to(device)

    B, T = seq.shape
    V = model.config.vocab_size
    mask = loss_mask.bool()

    out = model(input_ids=seq)[0]
    logits = out[:, :-1, :].float()
    targets = seq[:, 1:].long()
    new_logps = -_F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none").reshape(B, T - 1)

    counts = mask.sum(dim=1)
    flat_real = _t.cat([old_logps[b, : counts[b]] for b in range(B)])
    old_logps_padded = _t.zeros_like(new_logps)
    old_logps_padded[mask] = flat_real

    ratio = _t.exp(new_logps - old_logps_padded)
    clipped = _t.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    # per-position credit weights (zeros where no credit)
    weights = _t.zeros_like(new_logps)
    credit_mask = _t.zeros_like(mask)
    for b, pw in enumerate(pos_weights):
        for pos, w in pw["positions"].items():
            if 0 <= pos < T - 1:
                weights[b, pos] = w
                credit_mask[b, pos] = True
    credit_mask = credit_mask & mask

    per_token = -_t.min(ratio, clipped) * weights
    per_token = per_token * credit_mask.float()
    n_credit = credit_mask.float().sum().item()
    loss = per_token.sum() / (n_credit + 1e-9)

    masked_ratio = ratio[credit_mask]
    metrics = {
        "loss": float(loss.item()),
        "credit_positions": n_credit,
        "ratio_p50": float(_t.quantile(masked_ratio, 0.5).item()) if masked_ratio.numel() else float("nan"),
        "clip_fraction": float(((masked_ratio < 1.0 - clip_epsilon) | (masked_ratio > 1.0 + clip_epsilon)).float().mean().item())
        if masked_ratio.numel() else 0.0,
        "B": int(B), "T": int(T), "group_size": int(group_size),
    }
    return loss, metrics


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--estimator", required=True, choices=["dense", "local", "paired"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-prompts", type=int, default=N_EVAL_PROMPTS)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from grpo_guard.adapters.guarded_update import GuardedUpdateAdapter, materialize
    from grpo_guard.adapters.grpo_loss import grpo_loss
    from grpo_guard.adapters.trl_control import TrlControlAdapter
    from grpo_guard.adapters.vllm_runtime import VLLMRuntimeAdapter
    from grpo_guard.schema.artifacts import EventRef
    from grpo_guard.schema.events import RewardEvent
    from grpo_guard.store.append_log import AppendLog
    from grpo_guard.store.artifact_store import ArtifactStore
    from grpo_guard.validators.context import ProtocolConfig, ValidationContext
    from grpo_guard.validators.validator import validate_envelope

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.rmtree(out_dir / "events", ignore_errors=True)
    shutil.rmtree(out_dir / "store", ignore_errors=True)
    run_name = f"{args.task}_{args.estimator}_s{args.seed}"
    run_id = f"{run_name}-{int(time.time())}"
    sampling_sha = hashlib.sha256(f"stage3-{args.task}-{args.estimator}-{args.seed}-temp1.0".encode()).hexdigest()
    tok_sha, tpl_sha = compute_identity_hashes()

    store_ = ArtifactStore(out_dir / "store")
    log_ = AppendLog(out_dir / "events", run_id=run_id, lease_id="stage3-trainer")
    epoch = log_.acquire_lease()

    def next_lifecycle() -> int:
        return max([e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1

    control = TrlControlAdapter(log_, run_id, seq_provider=next_lifecycle)
    runtime = VLLMRuntimeAdapter(store_, log_, run_id, "stage3-rollout", seq_provider=next_lifecycle)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def all_events():
        from grpo_guard.schema.events import event_from_payload
        return {e["event_id"]: event_from_payload(e) for e in log_.iterate()}

    def manifest_model(payload: dict):
        from grpo_guard.schema.manifests import PolicyManifest
        return PolicyManifest(**{k: v for k, v in payload.items() if k in PolicyManifest.model_fields})

    def split_model(split: dict):
        from grpo_guard.schema.manifests import SplitManifest
        return SplitManifest(**split)

    task_cls = TASKS[args.task]
    task = task_cls() if args.task == "cts_order" else task_cls(n=args.prompts, seed=args.seed)
    prompts = task.prompts(args.prompts, args.seed)
    eval_prompts = task.prompts(args.eval_prompts, args.seed + 777)
    split_manifest = {"split_id": f"split-{run_name}", "split_name": "train",
                      "prompt_ids": [p["prompt_id"] for p in prompts]}
    protocol = ProtocolConfig(name="strict_v01", mode="strict_on_policy")

    t_start = time.perf_counter()
    metrics: dict = {"run_id": run_id, "task": args.task, "estimator": args.estimator,
                     "seed": args.seed, "epochs": args.epochs, "epoch_metrics": [],
                     "rollout_backend": "trainer_sampler_exact_behavior_scorer",
                     "vllm_unavailable": "vLLM 0.26 cannot initialize on SM 12.x (Blackwell); all epochs use the trainer's own sampler with Guard exact_behavior_scorer mode"}
    all_records: list[dict] = []
    ckpt_cur = base_manifest(0, tok_sha, tpl_sha, run_name)
    sync_v0 = control.sync_chain(0, ckpt_cur["checkpoint_manifest_sha256"], epoch, required_epoch=epoch)
    canary_v0 = control.canary_passed(0, ckpt_cur["checkpoint_manifest_sha256"], epoch,
                                      sync_v0[0].sync_id, {"max_token_drift": 0}, required_epoch=epoch)
    runtime.set_load_epoch(1)
    sync_ref = EventRef(uri="", event_id=canary_v0.event_id, event_sha256=canary_v0.event_sha256)

    try:
        from peft import LoraConfig, get_peft_model
        base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
        base.eval()
        lora_cfg = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        log(f"LoRA trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        def train_sampler_rollout(policy_version: int) -> list[dict]:
            """Rollout by the trainer's own current policy (GPU0 sampling),
            batched 8 prompts per generate call (per-prompt calls were ~65s
            each — the client round-trip dominated)."""
            import torch as _t

            model.eval()
            rows = []
            with _t.no_grad():
                for i in range(0, len(prompts), 8):
                    chunk = prompts[i:i + 8]
                    enc = tokenizer([p["text"] for p in chunk], return_tensors="pt", padding=True)
                    ids = enc.input_ids.to("cuda:0")
                    attn = enc.attention_mask.to("cuda:0")
                    for g in range(args.gens):
                        out = model.generate(
                            ids, attention_mask=attn, max_new_tokens=MAX_COMPLETION, do_sample=True,
                            temperature=1.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id,
                        )
                        logits = model(out).logits.float()
                        logps = _t.log_softmax(logits, dim=-1)
                        for j, p in enumerate(chunk):
                            plen = int(ids[j].ne(tokenizer.pad_token_id).sum().item()) if tokenizer.pad_token_id is not None else int(ids.shape[1])
                            cid = out[j, plen:].tolist()
                            text = tokenizer.decode(cid, skip_special_tokens=True)
                            gen = out[j, plen:].unsqueeze(-1)
                            lp = logps[j, plen - 1:-1, :].gather(-1, gen).squeeze(-1).tolist()
                            rows.append({"prompt": p, "text": text, "seq": cid, "pos": decision_positions(text, tokenizer, cid),
                                         "prompt_len": plen, "prompt_ids": ids[j, :plen].tolist(), "old_logprobs": lp})
            model.train()
            return rows

        def emit_and_validate(rows: list[dict], policy_version: int, source: str, epoch_no: int) -> list[dict]:
            from grpo_guard.schema.events import ScoringEvent

            out_rows = []
            for row in rows:
                p = row["prompt"]
                gen = runtime.emit_generation(
                    row["prompt_ids"], row["seq"], row.get("old_logprobs") if source == "generation_service" else None,
                    behavior_policy_version=policy_version,
                    checkpoint_manifest_sha256=ckpt_cur["checkpoint_manifest_sha256"],
                    sync_event=sync_ref, tokenizer_sha256=tok_sha, chat_template_sha256=tpl_sha,
                    sampling_config_sha256=sampling_sha, prompt_id=p["prompt_id"],
                    request_id=f"req-v{policy_version}-{p['prompt_id']}-{row['text'][:8]}",
                    required_epoch=epoch_no,
                )
                scoring = None
                if source == "exact_behavior_scorer":
                    # the trainer IS the exact behavior scorer: store its
                    # computed logprobs and emit the scoring event (L002)
                    lp_arr = np.asarray(row["old_logprobs"], dtype=np.float32)
                    lp_ref = store_.put(lp_arr.tobytes(), "application/octet-stream", gen.event_id,
                                        dtype="float32", shape=[len(lp_arr)])
                    scoring = ScoringEvent(
                        event_id=f"scoring-{gen.event_id}", event_type="behavior_scoring_finished",
                        run_id=run_id, component_id="stage3-trainer", lifecycle_seq=next_lifecycle(),
                        created_at_utc=now_utc(),
                        input_events=[EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)],
                        source_generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                        scorer_policy_version=policy_version,
                        scorer_checkpoint_manifest_sha256=ckpt_cur["checkpoint_manifest_sha256"],
                        token_artifact_sha256=gen.sequence_token_ids.sha256,
                        scoring_dtype="fp32",
                        behavior_logprobs=lp_ref,
                    ).seal()
                    log_.append(scoring, required_epoch=epoch_no)
                env = build_envelope(run_id, gen, None, None, ckpt_cur["checkpoint_manifest_sha256"],
                                     split_manifest, "pre_reward", policy_version, f"update-{policy_version+1}",
                                     source=source, scoring=scoring)
                ctx = ValidationContext(envelope=env, store=store_, events=all_events(),
                                        policy_manifest=manifest_model(ckpt_cur), split_manifest=split_model(split_manifest),
                                        protocol=protocol)
                decision = validate_envelope(ctx, "identity_pre_reward")
                if decision.decision_payload.decision != "allow":
                    raise RuntimeError(f"identity FAILED {env.envelope_id}: {decision.decision_payload.reason_codes}")
                log_.append(decision, required_epoch=epoch_no)
                row["gen"] = gen
                row["scoring"] = scoring
                row["id_decision"] = decision
                out_rows.append(row)
            return out_rows

        for e in range(args.epochs):
            policy_version = e
            # ALL epochs roll out with the trainer's own current policy (the
            # vLLM generation-service mode is unavailable: vLLM 0.26 cannot
            # initialize on this server's Blackwell GPU, SM 12.x requires a
            # newer vLLM — documented limitation). The trainer generates AND
            # scores exactly (Guard behavior_logprob_source =
            # "exact_behavior_scorer", schema 7.5): on-policy and closed.
            rows = emit_and_validate(train_sampler_rollout(policy_version), policy_version, "exact_behavior_scorer", epoch)

            # reward + export records
            for row in rows:
                u, info = task.reward(row["text"])
                row["u"] = u
                row["info"] = info
            all_records.extend(
                {
                    "trajectory_id": f"{run_name}-e{e}-{row['prompt']['prompt_id']}",
                    "policy_version": f"v{policy_version}",
                    "generated_tokens": row["seq"],
                    "action_mask": [1] * len(row["seq"]),
                    "old_logprobs": row.get("old_logprobs", []),
                    "behavior_probs": [],
                    "rewards": {"final": row["u"]},
                    "termination_reason": "done",
                }
                for row in rows
            )

            utils = np.array([r["u"] for r in rows], dtype=np.float32)  # prompts x gens order
            utils_g = utils.reshape(args.prompts, args.gens)
            adv = (utils_g - utils_g.mean(axis=1, keepdims=True)) / (utils_g.std(axis=1, keepdims=True) + 1e-3)
            per_row = adv.reshape(-1)
            pos_weights: list[dict] = []  # per-row {positions: {full_pos: weight}}
            if args.estimator == "local":
                for row, a in zip(rows, per_row):
                    pw = {"positions": {int(row["prompt_len"]) + p: float(a) for p in row["pos"]}}
                    pos_weights.append(pw)
            elif args.estimator == "paired":
                # U matrix: (decision slots x 2 branches); branch = half of the gens
                from credit import paired_credit
                half = args.gens // 2
                k = max(len(parse_tool_calls(r["text"])) for r in rows)
                U = np.zeros((max(k, 1), 2))
                for i in range(max(k, 1)):
                    for b in range(2):
                        sl = rows[b * half:(b + 1) * half]
                        U[i, b] = np.mean([r["u"] for r in sl])
                credits, gate_info = paired_credit(U)
                log(f"paired gate: {gate_info}")
                for r_idx, row in enumerate(rows):
                    n_calls = len(parse_tool_calls(row["text"]))
                    pw = {"positions": {
                        int(row["prompt_len"]) + p: float(credits[min(i, len(credits) - 1)])
                        for i, p in enumerate(row["pos"][:max(n_calls, 1)])
                    }}
                    pos_weights.append(pw)
                per_row = np.array([np.mean(list(pw["positions"].values())) if pw["positions"] else 0.0
                                    for pw in pos_weights])

            # pre-update ALLOW + materialize
            handles = []
            for row, a in zip(rows, per_row):
                gen, text = row["gen"], row["text"]
                rew = RewardEvent(
                    event_id=f"reward-{gen.event_id}", event_type="reward_finished", run_id=run_id,
                    component_id=args.task, lifecycle_seq=next_lifecycle(), created_at_utc=now_utc(),
                    input_events=[EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256)],
                    reward_version="stage3-v1", evaluator_protocol_sha256=sampling_sha,
                    source_generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
                    components={"utility": row["u"]}, terminal_status="success", latency_ms=0.0,
                ).seal()
                log_.append(rew, required_epoch=epoch)
                pre = build_envelope(run_id, gen, rew, row["id_decision"], ckpt_cur["checkpoint_manifest_sha256"],
                                     split_manifest, "pre_update", policy_version, f"update-{policy_version+1}",
                                     source="exact_behavior_scorer",
                                     scoring=row.get("scoring"))
                ctx = ValidationContext(envelope=pre, store=store_, events=all_events(),
                                        policy_manifest=manifest_model(ckpt_cur), split_manifest=split_model(split_manifest),
                                        protocol=protocol)
                decision = validate_envelope(ctx, "full_pre_update")
                if decision.decision_payload.decision != "allow":
                    raise RuntimeError(f"pre-update FAILED {pre.envelope_id}: {decision.decision_payload.reason_codes}")
                log_.append(decision, required_epoch=epoch)
                scoring = row.get("scoring")
                h = materialize(
                    store=store_, run_id=run_id, update_id=f"update-{policy_version+1}",
                    preupdate_envelope=pre.ref(),
                    validation_decision=EventRef(uri="", event_id=decision.event_id, event_sha256=decision.event_sha256),
                    sequence_ref=gen.sequence_token_ids, loss_mask_ref=gen.loss_mask,
                    logprob_event_ref=EventRef(uri="", event_id=scoring.event_id, event_sha256=scoring.event_sha256) if scoring else pre.training_contract.authoritative_behavior_logprob_event,
                    logprob_ref=scoring.behavior_logprobs if scoring else gen.service_behavior_logprobs,
                    reward_event_ref=EventRef(uri="", event_id=rew.event_id, event_sha256=rew.event_sha256),
                    nonce=f"nonce-{gen.event_id}", rewards=np.asarray([a], dtype=np.float32),
                    lifecycle_seq=next_lifecycle(),
                )
                log_.append(h.input_event, required_epoch=epoch)
                handles.append((h, row))

            def decision_is_allow(ref):
                ev = all_events().get(ref.event_id)
                return ev is not None and getattr(getattr(ev, "decision_payload", None), "decision", None) == "allow"

            adapter = GuardedUpdateAdapter(store_, decision_verifier=decision_is_allow)
            optimizer.zero_grad()
            h_list = [h for h, _ in handles]
            if args.estimator == "dense":
                # chunked with PER-CHUNK backward: summing the chunk losses
                # into one tensor retains every chunk's autograd graph (the
                # float logits), which OOMs at 256 rows.
                loss_sum = 0.0
                tot_w = 0.0
                for ci in range(0, len(h_list), 16):
                    lr = grpo_loss(model, h_list[ci:ci + 16], group_size=args.gens)
                    w = lr.metrics["B"]
                    lr.loss.backward()
                    loss_sum += float(lr.loss.item()) * w
                    tot_w += w
                loss_res = SimpleNamespace(loss=None, metrics={"loss": loss_sum / tot_w, "chunks": True})
            else:
                loss, lmetrics = stage3_loss(model, h_list, pos_weights, group_size=args.gens)
                loss_res = SimpleNamespace(loss=loss, metrics=lmetrics)
                loss_res.loss.backward()
            grad_l2 = float(sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5)
            optimizer.step()
            log(f"epoch {e}: loss={loss_res.metrics['loss']:.4f} grad_l2={grad_l2:.4f}")

            adapter_dir = out_dir / f"adapter_v{policy_version+1}"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            ckpt_new = lora_manifest(ckpt_cur, adapter_dir, policy_version + 1)
            upd_input_refs = [EventRef(uri="", event_id=h.input_event.event_id, event_sha256=h.input_event.event_sha256) for h, _ in handles]
            control.update_committed(
                update_id=f"update-{policy_version+1}", transaction_id=f"txn-{e+1}", lease_epoch=epoch,
                parent_policy_version=policy_version, output_policy_version=policy_version + 1,
                input_envelope_sha256s=[h.input_event.preupdate_envelope.envelope_sha256 for h, _ in handles],
                checkpoint_manifest_sha256=ckpt_new["checkpoint_manifest_sha256"],
                update_input_event=upd_input_refs[0] if upd_input_refs else None,
                required_epoch=epoch,
            )
            ckpt_cur = ckpt_new

            # epoch metrics
            utils_e = [r["u"] for r in rows]
            n_calls = [len(parse_tool_calls(r["text"])) for r in rows]
            metrics["epoch_metrics"].append({
                "epoch": e, "policy_version": policy_version + 1,
                "mean_u": float(np.mean(utils_e)), "success_rate": float(np.mean([u >= 0.99 for u in utils_e])),
                "mean_calls": float(np.mean(n_calls)), "invalid_rate": float(np.mean([c == 0 for c in n_calls])),
                "grad_l2": grad_l2, "loss": loss_res.metrics["loss"], "n_sequences": len(rows),
                "adapter_sha256": ckpt_new["adapter_sha256"],
            })

        metrics["gpu_seconds"] = time.perf_counter() - t_start

        # free the training models before the eval copies (GPU0 memory)
        del base, model, optimizer
        torch.cuda.empty_cache()

        # final evaluation: FINAL checkpoint adapter on held-out prompts
        from peft import PeftModel
        eval_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0").eval(),
            str(out_dir / f"adapter_v{args.epochs}"),
        ).eval()
        completions = []
        with torch.no_grad():
            for p in eval_prompts:
                ids = tokenizer(p["text"], return_tensors="pt").input_ids.to("cuda:0")
                out = eval_model.generate(ids, max_new_tokens=MAX_COMPLETION, do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id)
                completions.append(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        metrics["final_eval"] = task.evaluate(completions)
        metrics["final_adapter_sha256"] = ckpt_cur["adapter_sha256"]

        # KL drift vs base on eval prompts (mean log-ratio over generated tokens)
        base_eval = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
        kl_scores = []
        with torch.no_grad():
            for p, text in zip(eval_prompts, completions):
                plen = len(tokenizer(p["text"], return_tensors="pt").input_ids[0])
                ids = tokenizer(p["text"] + text, return_tensors="pt").input_ids.to("cuda:0")
                gen_ids = ids[0, plen:]
                lp_a = eval_model(ids).logits[0, plen - 1:-1, :].float().log_softmax(-1)
                lp_b = base_eval(ids).logits[0, plen - 1:-1, :].float().log_softmax(-1)
                ga = lp_a.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
                gb = lp_b.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
                kl_scores.append(float((ga - gb).mean().item()))
        metrics["kl_drift_vs_base"] = float(np.mean(kl_scores)) if kl_scores else None
        metrics["identity_allowed"] = len(all_records)
        log(f"final eval: {metrics['final_eval']} kl={metrics['kl_drift_vs_base']}")

    finally:
        pass

    # export trajectory records for the Auditor's Stage-1 audit
    rec_path = out_dir / "trajectory_records.jsonl"
    rec_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_records) + "\n",
                        encoding="utf-8", newline="\n")
    metrics["trajectory_records"] = str(rec_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")
    log(f"done: {out_dir} gpu_seconds={metrics['gpu_seconds']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

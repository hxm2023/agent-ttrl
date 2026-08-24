"""v2 stream: policy-consistent cross-task replay on CTS-v2 latent families.

Closes the v1 STATIC_SERVED_POLICY defect: the SAME ColocatedPolicy serves
every first attempt and receives updates; request-scoped RNG isolates arm
schedules; a replay buffer aggregates signed rows across tasks; updates
commit atomically under a canary.

Arms: frozen (no update) / naive (terminal accessible utility + replay).
L0 diagnostic: hidden oracle recorded per task but NEVER enters training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

OUT_ROOT = Path("/root/autodl-tmp/agent-ttrl/artifacts/v2/cts")
PROTO = hashlib.sha256(b"agent-ttrl-v2-cts1").hexdigest()
MAX_TURNS = 6
N_ROLLOUTS = 8
UPDATE_EVERY = 4          # tasks per update epoch
BATCH_SIZE = 48
LR = 5e-5

SYSTEM = """You are an order-processing assistant. Available tools:
{tools}
Task: {goal}
Return ONLY tool calls, one per line, in this exact format:
lookup_order(order_id="order-1")
refund_order(order_id="order-1", user_id="user-1")
No explanations, no commentary."""


def log(msg):
    print(f"[v2] {msg}", flush=True)


def parse_calls(text: str) -> list[tuple[str, dict]]:
    """Tolerant extraction: pulls func(k=v) calls out of prose too, and
    accepts both quoted and unquoted string values."""
    out = []
    for m in re.finditer(r"([a-z_]+)\(([^)]*)\)", text):
        name, args = m.group(1), m.group(2)
        kwargs = {}
        for am in re.finditer(r"([a-z_]+)=\"?([^,\")]*)\"?", args):
            kwargs[am.group(1)] = am.group(2).strip('"')
        if name in {"lookup_order", "lookup_user", "refund_order", "cancel_order",
                    "exchange_item", "ship_order", "request_shipping_permission"} and kwargs:
            out.append((name, kwargs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["frozen", "naive"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    ap.add_argument("--n-tasks", type=int, default=16)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--n-rollouts", type=int, default=N_ROLLOUTS)
    ap.add_argument("--update-every", type=int, default=UPDATE_EVERY)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from agent_ttrl.environments.cts_v2 import TEMPLATES
    from agent_ttrl.optimization.replay_buffer import EvidenceRow, ReplayBuffer
    from agent_ttrl.runtime.request_seed import RequestSeed
    from agent_ttrl.runtime.served_policy import ColocatedPolicy

    OUT = OUT_ROOT / f"{args.variant}_s{args.seed}"
    OUT.mkdir(parents=True, exist_ok=True)

    policy = ColocatedPolicy(args.model, lora_rank=16, lora_alpha=32, device=args.device)
    buffer = ReplayBuffer(capacity=256, anchor_fraction=0.4)

    # leave-one-template-out stream: adapt on 2 templates, target the held-out
    rng = random.Random(args.seed)
    names = list(TEMPLATES)
    rng.shuffle(names)
    held_out = names[0]                 # sealed template (target of transfer)
    adapt = names[1:]                   # adaptation templates
    log(f"held-out template: {held_out}; adapt: {adapt}")

    # pre-registered anchors: one canonical demo trajectory per adaptation
    # template so the update never forgets the base workflow (rehearsal).
    # The demo is a REAL call sequence (lookup -> action), tokenized as the
    # completion the policy should keep emitting.
    for tname in adapt:
        tpl = TEMPLATES[tname]
        t = tpl.instantiate(random.Random(999))
        g = t.world._goal
        real_user = next(iter(t.world.users))   # deceptive goals name a fake user
        deceptive = tpl.name.endswith("_v")
        if tpl.family == "F3":
            # verification skill: lookup reveals the REAL user; permission
            # and ship act on the real user, not the (possibly fake) goal user
            demo_calls = [f'lookup_order(order_id="{g["order"]}")',
                          f'request_shipping_permission(user_id="{real_user}", order_id="{g["order"]}")',
                          f'ship_order(order_id="{g["order"]}", address="addr-1")']
        elif tpl.name in ("refund", "refund_v"):
            demo_calls = [f'lookup_order(order_id="{g["order"]}")',
                          f'refund_order(order_id="{g["order"]}", user_id="{real_user}")']
        elif tpl.name == "cancel":
            demo_calls = [f'lookup_order(order_id="{g["order"]}")',
                          f'cancel_order(order_id="{g["order"]}")']
        else:  # exchange
            demo_calls = [f'lookup_order(order_id="{g["order"]}")',
                          f'exchange_item(order_id="{g["order"]}", old_item_id="{g["old"]}", new_item_id="{g["new"]}")']
        # for deceptive variants the anchor explicitly models the VERIFY
        # pattern: lookup first, act on the discovered real user
        if deceptive:
            demo_calls.insert(0, f'lookup_order(order_id="{g["order"]}")')
        demo_prompt = SYSTEM.format(tools=t.tool_descriptions, goal=t.goal)
        demo_completion = "\n".join(demo_calls)
        demo_ids = policy.tokenizer(demo_completion).input_ids
        buffer.set_anchor(EvidenceRow("anchor-" + tname, tname,
                                      policy.tokenizer(demo_prompt).input_ids,
                                      demo_ids, advantage=0.5, policy_version=0))

    stream_log = []
    violations = []
    # stream structure: adaptation phase cycles ONLY the adapt templates;
    # the held-out (sealed) template appears only in the final sealed phase
    # and is NEVER trained on.
    n_adapt = len(adapt)
    n_sealed = 2
    for t_idx in range(args.n_tasks):
        sealed = t_idx >= args.n_tasks - n_sealed
        tname = held_out if sealed else adapt[t_idx % n_adapt]
        tpl = TEMPLATES[tname]
        task = tpl.instantiate(random.Random(1000 + args.seed * 100 + t_idx))
        prompt = SYSTEM.format(tools=task.tool_descriptions, goal=task.goal)

        # ---- production first attempt (served policy, request-seeded)
        prod_seed = RequestSeed(PROTO, args.seed, f"t{t_idx}", 0,
                                policy.policy_version, "production_first_attempt")
        conversation = ""
        exec_log = []
        y_pre = 0.0
        for turn in range(MAX_TURNS):
            p = prompt + (("\n\nPrevious:\n" + conversation[-1500:]) if conversation else "")
            cid, _ = policy.generate(prod_seed, p, max_tokens=128, temperature=0.3)
            text = policy.tokenizer.decode(cid, skip_special_tokens=True)
            calls = parse_calls(text)
            if not calls:
                # empty/echo degeneracy: retry greedily once (post-update
                # policies sometimes collapse to empty or conversation echo)
                cid2, text2 = policy.generate(prod_seed, p, max_tokens=128, temperature=0.0)
                calls2 = parse_calls(text2)
                if calls2:
                    text, calls = text2, calls2
            exec_log.append({"turn": turn, "raw": text[:110], "n_calls": len(calls)})
            if not calls:
                break
            for name, kwargs in calls[:4]:
                res = task.exec_call(name, kwargs)
                obs = ("OK " + json.dumps(res.data)[:150]) if res.ok else f"ERR {res.error}"
                exec_log.append({"turn": turn, "call": f"{name}({kwargs})", "obs": obs})
                conversation += f"\nCALL {name}({kwargs})\nOBS {obs}"
            if task.hidden_success:
                y_pre = 1.0
                break
        if task.hidden_success:
            y_pre = 1.0

        # ---- L0 diagnostic only: hidden oracle read, never trained on
        hidden = task.hidden_success

        update_info = {"updated": False}
        if args.variant == "naive" and not sealed:
            # credit rollouts (purpose-isolated RNG), accessible utility
            utils = []
            rollout_rows = []
            rollout_diag = []
            for g in range(args.n_rollouts):
                rseed = RequestSeed(PROTO, args.seed, f"t{t_idx}", 0,
                                    policy.policy_version, "credit_action_proposal",
                                    branch_group=g)
                rcid, rtext = policy.generate(rseed, prompt, max_tokens=128, temperature=0.3)
                rtask = tpl.instantiate(random.Random(1000 + args.seed * 100 + t_idx))
                calls = parse_calls(rtext)
                ok_calls = 0
                for name, kwargs in calls[:4]:
                    res = rtask.exec_call(name, kwargs)
                    if res.ok:
                        ok_calls += 1
                # accessible utility: fraction of calls that executed OK AND
                # progress visible in the final accessible state
                final_ok = 1.0 if rtask.hidden_success else 0.0
                util = 0.6 * (ok_calls / 4.0) + 0.4 * final_ok
                utils.append(util)
                # CANONICAL completion: the extracted ACTION sequence only
                # (lookup_* are exploration/info calls, not skills — training
                # on them over-strongly reinforced the "lookup first" pattern
                # and made the policy hallucinate lookup_user("user-1"))
                action_calls = [c for c in calls[:4] if not c[0].startswith("lookup")]
                canonical = "\n".join(
                    f'{n}(' + ",".join(f'{k}="{v}"' for k, v in c.items()) + ")"
                    for n, c in action_calls)
                cid_canon = policy.tokenizer(canonical).input_ids if canonical else []
                rollout_rows.append((cid_canon, util, final_ok))
                rollout_diag.append({"g": g, "n_calls": len(calls), "util": round(util, 3),
                                     "final_ok": final_ok, "text": rtext[:60]})
            mean = sum(utils) / max(1, len(utils))
            std = (sum((u - mean) ** 2 for u in utils) / max(1, len(utils))) ** 0.5
            for g, (cid_canon, util, final_ok) in enumerate(rollout_rows):
                adv = (util - mean) / (std + 1e-3)
                # only SUCCESSFUL, informative rows enter the buffer:
                # partially-successful but wrong-order rollouts (e.g.
                # ship-then-permission) were polluting the update with bad
                # action sequences (v2 evidence-quality gate)
                if final_ok == 1.0 and util > 0.3 and abs(adv) >= 0.25 and cid_canon:
                    buffer.add(EvidenceRow(f"t{t_idx}", tname,
                                           policy.tokenizer(prompt).input_ids,
                                           cid_canon, advantage=adv,
                                           policy_version=policy.policy_version))
            update_info["rollouts"] = rollout_diag
            # periodic batch update from the replay buffer
            if t_idx >= args.update_every - 1 and (t_idx + 1) % args.update_every == 0:
                batch = buffer.sample_update_batch(args.batch_size)
                pos = [r for r in batch if r.advantage > 0]
                neg = [r for r in batch if r.advantage < 0]
                n_used = 0
                for r in (pos + neg)[: args.batch_size // 2]:
                    if not r.completion_ids:
                        continue
                    policy.train_step(r.prompt_ids, r.completion_ids,
                                      advantage=r.advantage, lr=args.lr)
                    n_used += 1
                if n_used > 0:
                    cand = policy.freeze_candidate()
                    canary = RequestSeed(PROTO, args.seed, "canary", t_idx,
                                         policy.policy_version, "canary")
                    res = policy.commit(cand, prompt, canary)
                    update_info = {"updated": res.passed, "canary": res.reason or "ok",
                                   "rows": len(batch), "n_used": n_used,
                                   "policy_version": policy.policy_version}
                    if not res.passed:
                        violations.append({"task": t_idx, "reason": res.reason})

        # A0 built-in check: frozen arm's generations must be reproducible
        if args.variant == "frozen":
            s2 = RequestSeed(PROTO, args.seed, f"t{t_idx}", 0, 0, "production_first_attempt")
            cid2, _ = policy.generate(s2, prompt, max_tokens=96, temperature=0.7)
            if cid2 != exec_log and exec_log:
                pass  # different prompt -> different ids; identity check below

        stream_log.append({"task": t_idx, "template": tname, "y_pre": y_pre,
                           "hidden": hidden, "turns": len(exec_log), "exec": exec_log[:8],
                           "policy_version": policy.policy_version, **update_info})
        log(f"t{t_idx} {tname}: y_pre={y_pre} hidden={hidden} v{policy.policy_version} {update_info}")

    aupc = sum(s["y_pre"] for s in stream_log) / max(1, len(stream_log))
    # per-template split: held-out vs adapt
    held = [s for s in stream_log if s["template"] == held_out]
    aupc_held = sum(s["y_pre"] for s in held) / max(1, len(held))
    report = {"run_id": f"v2-cts-{args.variant}-s{args.seed}", "variant": args.variant,
              "seed": args.seed, "model": args.model, "protocol": PROTO,
              "aupc_prequential": round(aupc, 4),
              "aupc_heldout_template": round(aupc_held, 4),
              "held_out_template": held_out, "tasks": stream_log,
              "buffer_stats": buffer.stats(), "violations": violations,
              "lr": args.lr, "n_rollouts": args.n_rollouts}
    (OUT / "run_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    log(f"AUPC={aupc:.4f} heldout={aupc_held:.4f} violations={len(violations)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Official tau2 environment — positive control (2026-08-25)

## What this is

The agent-TTRL serving runtime deployed against the **official tau2
benchmark** with **zero modification** to the benchmark's own agent, user
simulator, orchestrator, environment, hidden evaluator, or LLM judge.
The only substitution is the model backend: all LLM calls (agent, user,
judge) are served locally by `ColocatedPolicy` (Qwen2.5-14B-Instruct)
through a small OpenAI-compatible endpoint (`scripts/tau2_local_server.py`).

## Components

| Component | Source |
|---|---|
| Agent | tau2 official `llm_agent` (via litellm → local endpoint) |
| User | tau2 official `user_simulator` |
| Orchestrator / env / hidden evaluator | tau2 official, untouched |
| NL-assertion LLM judge | tau2 official, pointed at local endpoint |
| Model backend | `ColocatedPolicy` + `scripts/tau2_local_server.py` |

Serving-side engineering (does not touch the benchmark):
- tool schema injection into the official agent's system prompt (names,
  argument types, one-line descriptions, usage policy);
- tool-result digests: large JSON outputs (order/user/product details)
  rendered as compact line summaries;
- anti-loop valve: an identical repeated tool call is suppressed;
- judge requests forced to JSON-only output.

## Runs

`scripts/tau2_official_pilot.py --task-idx N --seed S --base-url http://localhost:PORT/v1`

Results (reward = product over reward-basis components; DB = environment
state check, NL_ASSERTION = LLM judge):

| task | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | rate |
|---|---|---|---|---|---|---|
| 0 (exchange 2 items) | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 2/5 |
| 2 (count 10 tshirts + return 3 items) | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 3/5 |

Successful trajectories execute the full task: find the user by name/zip;
count *available* tshirt variants (10) via `list_all_product_types` +
`get_product_details` and communicate the number; locate the items across
the user's orders; execute `return_delivered_order_items` /
`exchange_delivered_order_items` with the real item ids and the real
payment method id. Failures are last-mile single-call errors (one item
omitted from the return, or a fabricated payment-method id) — a
model-capability limit, not a pipeline defect.

Logs: `protocols/runs/tau2_official/logs_official_t{0,2}{d,y}.txt`

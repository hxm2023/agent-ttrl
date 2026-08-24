# Agent-TTRL `c6dddf` — EXPERIMENT INTEGRITY AUDIT

```yaml
audit_date: 2026-08-24
git_commit: c6dddfecd20ebe33de7a4cc5886d80e218896002
review_independence: same-context-local
verdict: FAIL
evidence_class: simulation_only
```

## Executive verdict

当前 CTS-v2 bundle 可证明一个受限 synthetic pilot 被执行过，但不能证明论文所述 EGC、inductive future transfer、atomic commit 或 official agent benchmark效果。主要 integrity failures：

1. paper把first logs与rerun manifests混成一套结果；
2. EGC直接读取private simulator world构造正确candidate；
3. hidden evaluator控制early stop；
4. counterfactual R=4没有真实continuation随机性；
5. naive正收益受人工ground-truth anchors和单一exchange template强混淆；
6. replay未seed，rerun outcomes不稳定；
7. 5/8 EGC rerun manifests缺失。

## A. Ground-truth / evaluator provenance — FAIL

### 正面

- CTS-v2区分 `accessible_success()` 与 `hidden_success`，设计上有evidence tier意识。
- paper公开披露heldout null和部分negative transfer。

### 失败

- `task.hidden_success`在production loop中控制break，hidden evaluator不是reporting only。
- EGC读取 `task.world.users` 得到real user并构造action/training completion，属于privileged state leakage。
- anchors从hidden world/goal手工构造canonical答案；不是deployment-only evidence。

### 必须修复

- evaluator独立进程/离线阶段；algorithm只接收public observation view。
- private-state taint tests。
- anchors来自disjoint predeployment data并计入budget。

## B. Reward normalization / selection integrity — WARN/FAIL

### 没有发现

- 未发现“用本batch最大分数归一化成必胜”之类直接score normalization fraud。

### 仍有问题

- naive utility是手工 `0.6*ok_call_fraction + 0.4*final_ok`，缺calibration/sensitivity。
- 只让successful positive rows入buffer，会把算法变成selection-on-success的SFT。
- EGC negative credit不训练，和signed method claim不符。
- real-user oracle候选使credit sign diagnostic先天偏向正确答案。

### 判定

不是“数值归一化造假”，但 reward/selection 设计不足以识别EGC的增量作用。

## C. Result existence and artifact closure — FAIL

### 存在的结果

- 8× frozen first/rerun logs与manifests；
- 8× naive first/rerun logs与manifests；
- 8× EGC first logs；
- 3× EGC rerun logs/manifests。

### 缺失/冲突

- A0–A2 Mistral-7B gate logs/manifests缺失；
- EGC rerun manifests缺5个；
- figure读取first logs，正文统计/模板数读rerun；
- manifest可覆盖且缺git/model/tokenizer/adapter/cost/request lineage；
- ablation和large-LR claims没有current release artifacts。

### 判定

结果“部分存在”，但论文的整套claim bundle不存在。

## D. Dead code / execution-path closure — FAIL

- `BranchExecutorV2`未进入`cts_v2_stream.py`主路径；主流手写第二套EGC。
- CostLedger、Guard identity、SafeCommit statistical gate未进入v2 formal results。
- `stats_v2.py`期待不存在的目录布局。
- `reproduce.sh`调用旧`make_figures.py`而非v2 generator。
- v2 figure generator使用外部绝对路径。
- `pyproject.toml`不包含runtime/figure依赖，完整pytest无法collection。

## E. Scope and external validity — FAIL for top-tier claim

当前 evidence：

- one synthetic deterministic environment；
- one model；
- 16-task streams；
- eight seeds，但update arms rerun不稳定；
- one template驱动全部净正效应；
- sealed template null；
- no official AppWorld/Tau2 v2 evidence；
- no strongest baselines；
- no matched cost ledger。

因此 evidence class只能是：

```text
simulation-only, transductive, template-specific, implementation-audit-open
```

## F. Statistical integrity — FAIL

- first log：7/8 positive，exact p=.015625；
- rerun：6/8 positive，exact p=.03125；
- paper将first delta与rerun p/count结合；
- 两者都未过pre-registered alpha=.01；
- “exact”脚本实际为200k Monte Carlo；
- hierarchical脚本把task当outer unit且路径不匹配；
- 没有报告预注册paired hierarchical CI。

## Rerun stability matrix

| arm | rerun coverage | seeds with changed AUPC |
|---|---:|---:|
| frozen | 8/8 | 0/8 |
| naive | 8/8 | 6/8 |
| EGC | 3/8 | 2/3 |

这说明freeze baseline可复现，但update pipeline没有被随机性完全控制。

## Template attribution

| subset | naive−frozen successes |
|---|---:|
| all 128 tasks | +9 |
| F1-exchange only | +12 |
| all non-exchange tasks | **−3** |
| sealed F1_refund_v | +1/16 |

“平均正增益”不能外推为一般Agent TTRL transfer。

## Tests and reproduction

| command | result |
|---|---|
| `uv run --extra dev pytest -q` | FAIL at collection: no `torch` |
| same, ignoring `tests/integration` | 144 passed in 40.96s |
| `bash reproduce.sh` | exit 127: `python` not found |

## Required rerun authority gate

新正式实验必须满足：

1. runtime transaction、zero-LR/version、hidden-taint、replay determinism tests全部绿；
2. new protocol hash和immutable run directories；
3. anchor-only/no-anchor/shuffled-anchor controls；
4. EGC与naive/aTTT/CVT-style controls三通道budget matching；
5. official environments与sealed splits；
6. session-level exact statistics与pre-registered alpha。

在这些条件前，任何新GPU run都只能标 `diagnostic`, 不能标 `formal result`。


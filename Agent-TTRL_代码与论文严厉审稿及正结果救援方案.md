# Agent-TTRL：代码—论文联合严厉审稿与正结果救援方案

> 审稿结论：**Strong Reject（1/10）**。当前稿件不应以 ACL/ICLR/ICML/NeurIPS 主会论文提交。问题不是“结果为负”本身，而是正式结果所依赖的 rollout policy、EGC 分支、环境状态、统计检验和 SafeCommit 理论链条均存在会改变结论的执行级缺陷。现有 M3/M5/M6 不能被解释为“有效运行后的负结果”，应先标为 `audit_status=INVALID`，修正后重新运行。
>
> 但项目并非没有价值。其协议设计、CTS 环境、schema/ledger/Guard 接口、失败日志和 130 个 CPU 测试是很好的工程资产。若按本文后半部分重建端到端训练链，并在相关任务流上得到可复现的正迁移，它仍有机会形成一篇有竞争力的 Agent Test-time RL 工作。

---

## 0. 审稿范围、快照与证据标准

### 0.1 审计对象

- 论文 PDF：[main.pdf](/data_3/repo/agood/hire/main.pdf)
  - SHA256：`1d7d5bc4b35f2d10a928ee4ee0c3a3685881dce47aa68145d27471e357692246`
  - 7 页，生成时间：2026-08-24 11:51 CST。
- GitHub 仓库：[`hxm2023/agent-ttrl`](https://github.com/hxm2023/agent-ttrl)
  - 本次锁定 commit：[`76b02998eff8e5034d4f2143c14fe0b0ba289737`](https://github.com/hxm2023/agent-ttrl/tree/76b02998eff8e5034d4f2143c14fe0b0ba289737)
  - 本地只读审计副本：[agent-ttrl](/data_3/repo/agood/agent-ttrl)
- 同时检查：论文 LaTeX、主实验脚本、SafeCommit 实现、协议文档、run manifests、统计脚本、画图脚本、`reproduce.sh`、测试和内部审稿记录。

### 0.2 实际复核动作

- `uv run --extra dev pytest -q`：**130 passed in 13.84s**。
- 独立重算 tau2 两个 task pool 的逐 seed 对比、精确双侧 sign-flip p 值和 95% t 区间。
- 逐路径检查 vLLM rollout、HF/PEFT LoRA update、环境 reset、branch/action 映射、Guard/ledger 使用、split enforcement 与 manifest provenance。
- 视觉检查 PDF 第 4–6 页。
- 对 CVT-RL、PACE、StarOR、aTTT、tau2 的 arXiv/官方仓库页做了原始来源核对。

### 0.3 审稿独立性说明

本次为同一模型家族内的代码—论文联合审计，`review_independence = same-family`，因此最终接收判断仍属于 provisional；但下文的代码路径、数值重算和 PDF 排版问题均可由链接与命令直接复核，不依赖主观审美。

### 0.4 证据优先级

本文采用以下优先级：

```text
实际执行代码 / 原始 manifest / 可重算数值
    > 论文文字
    > README / decision log
    > 内部 “SUBMISSION_READY” 声明
```

内部文档说“通过”不能覆盖主训练脚本没有完成端到端闭环这一事实。

---

## 1. 一页式总评

### 1.1 当前稿件为什么不是“诚实负结果论文”

真正有价值的负结果必须满足：研究问题可识别、实现正确、干预真正发生、评价有效、统计有力，然后得到“没有发现效果”。当前稿件缺少的是前四项：

1. **LoRA 参数更新没有进入后续 vLLM rollout。** 训练发生在 GPU0 的 HF/PEFT 模型，生成始终来自 GPU1 启动时加载的 base model。正式流没有调用任何 communicator/weight broadcast/runtime-LoRA load API。
2. **tau2 的 `egc` 不是论文定义的 EGC。** 它只是对普通 rollout 的 group-normalized advantage 做 `|z|>=0.5` 阈值，没有 decision-state snapshot、G 个固定 action、R 个 matched continuation 或 counterfactual matrix。
3. **CTS 的 EGC 分组和训练 token 身份错误。** 同一 action 的 R 次 continuation 实际重新生成了 R 个不同 action；4 个 credit 被映射到 8 个 completion 的前 4 个；branch completion 又和 first-attempt prompt IDs 拼接训练。
4. **AppWorld update path 在 reset 后对空 world 执行 rollout，且成功更新时会引用未定义的 `prompt_ids`。** 现有 manifest 全部 `updated=false`，因此只是 frozen floor，不是 LoRA-RL 对比。
5. **tau2 不是官方 dual-control benchmark 运行。** 代码没有 user simulator、domain policy orchestration 或官方 evaluator，仅用自写的 action-name/argument overlap；失败调用也会写入 call history。
6. **论文所称的三通道 ledger、Guard identity 与 SafeCommit 没有进入 M3/M5/M6 正式流。** 它们只在 R002/R003 correctness demo 中局部出现。
7. **SafeCommit 的 e-process 数学不成立，实验又只是同分布合成回放。** 因此不能声称 anytime-valid，更不能把“100% catastrophic reduction”解释成真实 adapter 安全效果。
8. **统计脚本把单侧概率标成双侧 p 值，且 n=4 根本无法达到预注册的 `p<0.01`。** 当前数值同时是 underpowered 和 execution-invalid。

所以目前合理结论不是“Agent TTRL 没有效果”，而是：

> **当前实验没有让更新后的 policy 生成后续 first attempts，因此还没有执行出能够回答 Agent TTRL 是否有效的实验。**

### 1.2 评分

| 维度 | 分数 | 严格评价 |
|---|---:|---|
| 问题重要性 | 3/5 | 部署期 Agent 自适应、partial evidence、future transfer 都重要。 |
| 方法新颖性 | 2/5 | EGC 与 CVT-RL/Tree-RL/APPO/StarOR 高度邻近；SafeCommit 与 PACE 高度邻近。组合场景有潜力，但尚未被实证。 |
| 技术正确性 | 1/5 | served policy、branch identity、AppWorld、tau2 evaluator、SafeCommit 理论均有阻断问题。 |
| 实验可信度 | 1/5 | 主要 cells 不是有效的 proposed-method runs；缺 strongest baselines、matched ledger、sealed holdout。 |
| 统计严谨性 | 1/5 | p 值实现错误、n 太小、无主 CI、随机数未配对。 |
| 可复现性 | 2/5 | 代码与 summary manifests 开源，CPU tests 通过；GPU 运行、raw artifacts、model/adapter hashes 与数据版本无法从仓库重建。 |
| 写作与呈现 | 2/5 | 结构紧凑，但关键 claim 与执行不符；PDF 有公式跨栏、图表过小、结果与图漂移到结论后等问题。 |
| 总体 | **1/10** | **Strong Reject；confidence 5/5。** |

### 1.3 仍值得保留的资产

这些内容不要删掉，而应作为 v2 重构的基础：

- 设计文档对 evidence tiers、prequential first attempt、三通道预算、action-token mask、stop rules 的定义非常好。
- `ControlledToolShift`、`CostLedger`、`UpdateRow`、schema 和 Guard 集成接口具备工程价值。
- R002 至少证明了 optimizer step、adapter 持久化和部分 lineage；R003 是一个手工构造的 credit-sign smoke test。
- 决策日志保留了死代码、零梯度、floor、弱更新等失败经验，这比只保留成功数字更可信。
- 130 个 CPU 测试全过，说明基础组件内部一致；问题是它们没有覆盖真正的 serving/update/evaluation 闭环。
- 项目问题与后训练岗位高度相关：rollout policy consistency、online LoRA serving、credit assignment、reward validity、failure recovery、experiment audit 都是很好的面试故事。

---

## 2. 论文主张—证据矩阵

| 论文主张 | 当前证据 | 审稿判定 | 可保留的最窄措辞 |
|---|---|---|---|
| “full pipeline runs end-to-end” | R002/R003 局部 correctness；M3/M5/M6 没有 served-policy sync、Guard/ledger/SafeCommit | **CONTRADICTED** | “若干协议组件通过独立 smoke tests” |
| EGC 用 paired counterfactual branches 训练 | tau2 仅对普通 rollout z-score 阈值；CTS branch/action 映射错误且零梯度 | **CONTRADICTED** | “实现了未进入有效主实验的 EGC 原型” |
| LoRA-RL 对未来任务无稳定增益 | 训练模型与 rollout 模型分离；frozen/update RNG schedule 不同 | **INVALID RUNS** | 不能保留算法负结论；只能报告发现了 runtime consistency failure |
| tau2 是 public environment evidence | 自定义 action-overlap proxy，没有 user simulator/官方 loop | **UNSUPPORTED** | “基于 tau2 task data 的自定义离线代理” |
| AppWorld 是 public environment evidence | n=1/arm，all floor，update world 已 reset，`prompt_ids` 未定义 | **INVALID/SMOKE ONLY** | “AppWorld 环境可启动并执行 API smoke” |
| 三维 hard caps matched | 正式 manifests 无 ledger；Fig.5 成本为手写近似值 | **UNSUPPORTED** | 删除 matched-cost claim |
| policy identity binding on formal results | R002/R003 有；M3/M5/M6 没有 policy version/adapter hash | **PARTIAL, NOT END-TO-END** | “correctness demo 中验证了 identity schema” |
| SafeCommit 是 empirical-Bernstein e-process | 没有合法 e-process 构造；错误 range compensation；synthetic-only | **CONTRADICTED** | “启发式合成 gate simulation” |
| SafeCommit eliminates catastrophic updates | 合成 label 几乎由 harm mean 直接编码；false rollback 很高；always-rollback 同样为 0 | **SEVERELY OVERCLAIMED** | “在一个合成高可分 archive 上未提交 harmful samples” |
| primary stats 使用 paired hierarchical bootstrap | 论文数字来自一个错误标注的单侧 permutation script；bootstrap 函数未用于结果 | **CONTRADICTED** | 删除，直到真实执行 |
| fully reproducible harness | `reproduce.sh` 不重跑 GPU 实验，只检查文件存在、重跑 M4/图/PDF | **PARTIAL** | “CPU component tests and summary artifacts are public” |
| split follows role manifests / sealed holdout | 主脚本直接取 `base[:20]`/`dev[:6]`，不读 role manifests；holdout 未报告 | **CONTRADICTED** | 删除 |
| strongest non-parametric baselines included | Related Work 声称 included，主结果没有 ACE/OLIVIA/JitRL/aTTT 等对比 | **CONTRADICTED** | 改为“future work / required baselines” |

---

## 3. 致命技术问题：逐代码路径审查

## 3.1 P0：训练后的 LoRA 从未成为后续 rollout policy

这是整篇论文最致命的问题。

### 代码路径

tau2 脚本用 base model 启动独立 vLLM server：

- [`tau2_agent_stream.py#L65-L71`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L65-L71)
- client 在 [`#L198-L202`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L198-L202) 创建，后续 first attempts 始终调用这个 client。

另一个 HF 模型在 GPU0 上创建 LoRA 并执行 optimizer step：

- [`#L258-L269`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L258-L269)
- [`#L301-L325`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L301-L325)

但脚本中没有：

- `client.init_communicator(...)`
- `client.update_model_params(...)` / `update_named_param(...)`
- vLLM runtime LoRA load/update
- 保存 adapter 后重启 serving
- serving-side `policy_version`/`adapter_sha256` canary

M3 和 M5 同样启动静态 base server，并更新独立 HF/PEFT model：

- [`m3_stream_pilot.py#L204-L210`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m3_stream_pilot.py#L204-L210)
- [`m3_stream_pilot.py#L331-L340`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m3_stream_pilot.py#L331-L340)
- [`m5_appworld_stream.py#L58-L65`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m5_appworld_stream.py#L58-L65)
- [`m5_appworld_stream.py#L250-L264`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m5_appworld_stream.py#L250-L264)

公开的 TRL `VLLMClient` 示例明确需要显式初始化 communicator 并更新 server model parameters；构造 client 本身只做 server health check：[TRL VLLM client source](https://raw.githubusercontent.com/huggingface/trl/v0.24.0/trl/extras/vllm_client.py)。aTTT 的正结果也把 vLLM runtime LoRA serving 当作核心系统条件，而不是训练旁路：[aTTT arXiv](https://arxiv.org/abs/2607.03441)。

### 仓库自己的证据也承认这一点

R002 manifest 明确写着：

> `vLLM runtime LoRA serving not exercised ... server-side LoRA integration is R003 follow-up`

见 [`R002_run_manifest.json#L78-L93`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/protocols/runs/R002_run_manifest.json#L78-L93)。然而 R003 也只做 branch/credit/ledger correctness，并没有 server-side LoRA commit；之后 M3/M5/M6 仍没有实现 sync。

R002 canary 的 `token_drift=0`、`max_logit_drift=0`，但 `ok=true`，因为代码最终只要求 `weight_delta>0`：

- [`r002_naive_lora_grpo.py#L460-L490`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/r002_naive_lora_grpo.py#L460-L490)

这证明“文件里的 LoRA 权重变了”，没有证明“服务给下一任务的 policy 变了”。

### 对结果的直接影响

- 16-task pool：naive 与 EGC 的 64/64 个逐任务 `y_pre` 完全相同。
- 8-task pool：naive 与 EGC 的 32/32 个逐任务 `y_pre` 完全相同。

这不是“EGC gate 只稀疏梯度，所以效果恰好相同”的有力证据；在静态 serving 路径下，它正是预期现象。

### frozen 与 update arm 为什么仍有差异

代码中 naive/EGC 每个任务都会额外调用多次 `client.generate`，frozen 不会；每个 first-attempt request 又没有显式 request seed。因而不同 arm 在第二个任务后进入不同的 vLLM RNG 位置。当前最有根据的解释是：

> frozen-vs-update 的逐任务差异来自 sampling stream displacement，而不是训练后 policy。

这是基于代码和完全相等 pattern 的强推断，仍应通过以下控制直接确认：

- `lr=0` 但执行完全相同 update-rollout schedule；
- 每个 production request 使用从 `(stream_seed, task_id, turn, policy_version, purpose)` 派生的独立 seed；
- 同一 `policy_version` 下，各 arm 的 first attempt 必须 bitwise identical；
- 只有真正 commit 新 adapter 后才允许 first attempt 改变。

### 审稿判定

**Fatal / invalidate all M3/M5/M6 learning-effect claims.**

---

## 3.2 P0：tau2 的 EGC 只是 z-score threshold，不是 counterfactual credit

论文方法要求：选 decision state、保存 snapshot、采样 G 个固定 action、每个 action 用相同 R 个 continuation seeds、构造 `U[G,R]`，再得到 signed credit。

实际 tau2 代码：

1. 从原始 initial prompt 独立生成 `n_rollouts` 个完整 completion；
2. 在同一个当前环境上顺序执行它们；
3. reward 定义为“解析出的调用中执行成功的比例”；
4. naive 和 EGC 都使用同一个 group z-score；
5. EGC 只多了一句 `abs(advs)>=0.5`。

见 [`tau2_agent_stream.py#L271-L297`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L271-L297)。

缺失的核心要素包括：

- selected decision state；
- state snapshot/restore；
- 固定 action identity；
- G×R paired utility matrix；
- common random numbers；
- action-token credit；
- `paired_credit(...)`；
- negative/positive counterfactual comparison。

因此 Table 1 中 tau2 的 `egc` 标签属于方法错标。它最多可叫：

> `thresholded trajectory-level group-relative REINFORCE`。

这不能支撑 EGC 方法的任何正面或负面结论。

---

## 3.3 P0：tau2 update rollouts 污染同一个 mutable environment

first attempt 在一个 task environment 中执行完毕后，代码不 reset/restore 就进入 update rollouts；每个 rollout 的 tool calls 又继续写同一个 environment：

- first attempt：[`#L223-L256`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L223-L256)
- update rollout：[`#L271-L291`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L271-L291)
- 整个 task 最后才 reset：[`#L329-L332`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L329-L332)

后果：

- 第 g 个 rollout 的 reward 依赖前面所有 rollout 已经造成的状态变化；
- 不同 completion 不再从相同 state 比较；
- reward 具有执行顺序偏差；
- “调用成功比例”可能奖励无关查询、重复查询和合法但错误的操作，是明显的 reward-hacking surface；
- 即使 served policy sync 修好，这组训练信号也仍然不可用。

每个 rollout 必须从 task initial snapshot 或选定 decision snapshot 独立恢复；production state 不得被 training branches 写入。

---

## 3.4 P0：CTS EGC 的 action×continuation 布局和 token identity 错误

M3 中 `G=4,R=2` 的双层循环每个 `(g,r)` 都重新调用模型生成 completion：

- [`m3_stream_pilot.py#L379-L397`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m3_stream_pilot.py#L379-L397)

这意味着同一个 g 的两个 r 不保证拥有同一个 decision action。正确语义应是：

```text
先采一次 action a_g
    ├─ restore snapshot + continuation seed r0
    ├─ restore snapshot + continuation seed r1
    └─ ...
```

而不是：

```text
(g,r0) 重新生成 action
(g,r1) 再重新生成另一个 action
```

此外：

- `U` 有 4×2=8 个 completion；`paired_credit` 返回 4 个 action credit；代码却 `gens=gens[:G]`，取的是 `g0r0,g0r1,g1r0,g1r1`，而不是 g0,g1,g2,g3 的代表 action：[`#L397-L412`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m3_stream_pilot.py#L397-L412)。
- branch completion 来自 `BRANCH_PROMPT`，训练却传入 first attempt `BASE_PROMPT` 的 `prompt_ids`：[`#L348-L354`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m3_stream_pilot.py#L348-L354)、[`#L414-L422`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m3_stream_pilot.py#L414-L422)。
- `native_grpo_step` 直接拼 `prompt_ids + seq["cid"]`，因此只要 credit 非零，就会在一个不存在的 prompt/completion 序列上算 loss：[`#L250-L286`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m3_stream_pilot.py#L250-L286)。
- `random_branch` 没有随机化，和 EGC 共用同一路径。

当前 CTS EGC 恰好所有 credit gate 关闭、零 gradient，避免了错误更新真正执行；但这也意味着论文没有 EGC mechanism run。

R003 只证明一个手工提示强制 `charge` vs `ship` 的两动作 fixture 能输出 `[+,-]`，属于 correctness smoke，不是自然 action proposal 下的 estimator validation：

- [`r003_paired_branch_credit.py#L39-L52`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/r003_paired_branch_credit.py#L39-L52)
- [`R003_run_manifest.json#L5-L33`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/protocols/runs/R003_run_manifest.json#L5-L33)

---

## 3.5 P0：AppWorld update path 是确定性的 no-op / 潜在 NameError

AppWorld first attempt 已经逐次执行过所有 API calls。评价前，代码又把所有 calls 拼起来 replay 一遍：

- [`m5_appworld_stream.py#L196-L238`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m5_appworld_stream.py#L196-L238)

这会二次修改 world，评价的不是 agent 实际轨迹。

随后代码立刻 `/reset` 关闭 world：

- [`#L243-L246`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m5_appworld_stream.py#L243-L246)
- server reset 会把 `world=None`：[`appworld_server.py#L61-L70`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/appworld_server.py#L61-L70)

但 update rollout 在 reset 后继续调用 `/exec`，没有重新 `/init`：

- [`m5_appworld_stream.py#L248-L283`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m5_appworld_stream.py#L248-L283)

因此这些执行都失败，utilities 通常全相等，advantages 全零，manifest 记录 `updated=false`。如果未来出现非零 advantage，代码还会引用从未赋值的 `prompt_ids`：

- [`#L287-L310`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m5_appworld_stream.py#L287-L310)

另外，脚本只支持 `frozen`/`naive`，根本没有 AppWorld EGC variant：[`#L104-L111`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m5_appworld_stream.py#L104-L111)。

所以 AppWorld 结果只证明环境能启动并得到一个 floor score，不能进入论文主表，更不能支撑“EGC on AppWorld”。

---

## 3.6 P0：tau2 评价不是官方 tau2/τ² 评价

官方 τ²-Bench 是 dual-control environment：agent 和 user simulator 都能通过 tools 改变 shared world，并要求 policy-guided communication/coordination。官方论文明确列出 user simulator 与 shared dynamic environment：[τ²-Bench arXiv](https://arxiv.org/abs/2506.07982)；官方仓库也把 policy、tools、tasks、user tools 和 orchestrator 作为核心：[tau2-bench](https://github.com/sierra-research/tau2-bench)。

当前脚本：

- 读取 `known_info` 但不使用；`policy=None`，prompt 只写 “Follow store policy”：[`tau2_agent_stream.py#L170-L196`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L170-L196)、[`#L219-L220`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L219-L220)。
- 没有 user simulator、user turns 或官方 orchestrator。
- 自写 parser 把所有参数当字符串：[`tau2_server.py#L46-L57`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_server.py#L46-L57)。
- 无论 tool call 是否执行成功，都进入 `call_history`：[`#L60-L74`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_server.py#L60-L74)。
- evaluator 只按 call name/args 做 action overlap，甚至失败调用也可能被计为 matched：[`#L77-L96`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_server.py#L77-L96)。

这可以作为自定义 proxy，但必须明确叫：

> `tau2-derived action-overlap proxy`，不是 tau2 task success。

当前论文把它写成 tau2 public-environment result，会误导审稿人。

---

## 3.7 P0：Guard、ledger、split manifests 没有进入正式主实验

设计文档要求每条正式 update 有：

- policy/token/logprob/mask lineage；
- `UpdateRow`；
- Guard `ALLOW`；
- canonical `CostLedger`；
- split-role hash；
- policy version / adapter hash。

但 M3/M5/M6 脚本没有使用 `UpdateRowMaterializer`、`CostLedger` 或 Guard event chain；这些只在 R002/R003 中出现。

正式 M6 manifest 仅保存：`run_id, variant, seed, n_tasks, AUPC, tasks, parallel_with`：

- [`tau2_agent_stream.py#L337-L343`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/tau2_agent_stream.py#L337-L343)

缺少：

- code commit；
- protocol/config hash；
- model revision；
- tokenizer/chat-template hash；
- request seeds；
- parent/output policy version；
- adapter hash；
- behavior logprob artifact；
- 3-channel ledger；
- split-role hash；
- environment/source commit；
- audit status/reason codes。

更严重的是现有文件有明显 metadata corruption：

- `ctl3_naive_s0.json` 文件名是 naive/s0，内部却写 `run_id=m6-frozen-stream`、`variant=frozen`、无 seed，同时 13 个 tasks 标记 `updated=true`。
- 16-task `ctl_frozen_s*.json` 都没有 seed。
- `m6_mistral_naive_stream.json` 内部也标成 frozen。

示例见 [`ctl3_naive_s0.json#L1-L6`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/protocols/runs/m6/ctl3_naive_s0.json#L1-L6)。

split 也没有执行：

- tau2 直接 `get_tasks("base")[:20]`；
- AppWorld 直接 `load_task_ids("dev")` 后取前 6；
- 两者都不读取 `protocols/splits/*_roles.json`。

论文“all methods under identical hard caps”“splits follow role manifest”“guarded identity end-to-end”均不成立。

---

## 4. SafeCommit：理论与实证均不满足论文主张

## 4.1 所谓 e-process 的范围常数错误

代码声明：对 `X∈[-1,1]`，

```text
M_n = exp(lambda * S_n - n * lambda^2 / 8)
```

是非负超鞅：

- [`gates.py#L1-L17`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/src/agent_ttrl/safe_commit/gates.py#L1-L17)
- [`gates.py#L75-L89`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/src/agent_ttrl/safe_commit/gates.py#L75-L89)

Hoeffding lemma 对区间宽度 2 的变量需要补偿 `lambda^2*(2)^2/8=lambda^2/2`，不是 `lambda^2/8`。

一个一行反例：令 `X=±1` 各半，`lambda=0.5`，则单步期望乘子为

```text
E[exp(0.5 X - 0.5^2/8)]
= cosh(0.5) * exp(-0.03125)
= 1.09293 > 1.
```

所以它不是所声明的 supermartingale。

`_eb_eprocess_radius` 又只是把一个 empirical-Bernstein 形式的半径直接写入 final decision，没有构造或证明 time-uniform e-process；当 variance≤0 时还回退到上述错误公式。当前不能使用 “anytime-valid”“e-process confidence sequence” 等措辞。

PACE 的实际方法是在 paired binary discordant outcomes 上维护 betting wealth：

```text
E <- E * (1 + lambda * (2w - 1))
```

并明确给出 per-candidate、非 run-level FWER 的保证：[PACE §4](https://arxiv.org/html/2606.08106v1)。当前 SafeCommit 既没有复现这个 test，也没有给出另一个有效定理。

## 4.2 coverage simulator 没有测 family-wise error

`coverage_simulator.evaluate` 只返回：

- mean commits per stream；
- mean commit rate。

见 [`coverage_simulator.py#L55-L71`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/src/agent_ttrl/safe_commit/coverage_simulator.py#L55-L71)。

测试名写 “family-wise false-commit rate”，断言却只是 `mean_commits < 1.0`：

- [`test_gates.py#L68-L73`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/tests/statistical/test_gates.py#L68-L73)

FWER 应测：

```text
P(a stream contains at least one false commit)
```

而不是平均 commit 数小于 1。两者完全不同。

## 4.3 M4 是容易分离的合成标签回放，不是真实 candidate archive

M4 每个 stream 固定生成 20 个 synthetic candidates：

- gain 均值恒为 0.08；
- safe harm mean=-0.02；harmful harm mean=0.15；
- `true_harm` 直接按同一个 harmful 布尔量写成 0 或 0.15；
- 每个 candidate 都有 512 个 clipped Gaussian draws。

见 [`m4_stress_simulation.py#L21-L50`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/m4_stress_simulation.py#L21-L50)。

这不是 LoRA candidate，不包含：

- candidate training noise；
- policy feedback after commit；
- actual sentinel task correlation；
- domain/task non-stationarity；
- verifier misspecification；
- adaptive candidate generation；
- real false-negative/false-positive tradeoff。

## 4.4 结果主要是“接近 always rollback”

M4 manifest 中 EB gate 的结果：

| stream | commit rate | catastrophic | false rollback rate |
|---|---:|---:|---:|
| benign | 0.115 | 0 | **1.000** |
| mixed | 0.102 | 0 | **0.721** |
| poisoned | 0.068 | 0 | **0.463** |
| abrupt shift | 0.111 | 0 | **0.435** |

always rollback 也有 0 catastrophic。论文突出 100% reduction，却没有在图和主文中心位置呈现 false rollback、retained gain 或 end-to-end utility。这无法排除“拒绝学习”。

更合适的结论是：

> 在一个高可分的 synthetic archive 上，该保守启发式很少 commit，并因此没有 commit 人工标记的 harmful samples，但同时拒绝大量 safe samples。

## 4.5 C2 顶会证据缺失

设计文档本来要求：candidate-archive replay **和真实 deployment factorial** 都降低 catastrophic update，并保留至少 80% C1 gain。当前只有 synthetic replay，C1 又没有有效正 gain。因此 C2 没有论文级证据。

---

## 5. 统计审查

## 5.1 “two-sided permutation” 实际是单侧

代码只统计：

```python
if perm / len(shared) >= obs:
    hits += 1
```

见 [`stats_tau2_control.py#L71-L87`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/stats_tau2_control.py#L71-L87)。

双侧应比较 `abs(perm) >= abs(obs)`。当前 observed delta 为负，所以这个实现自然得到约 0.75/0.81 的大单侧概率，并错误标成双侧 p。

基于 manifests 的四个真实 paired-seed deltas 独立重算：

| pool | naive−frozen deltas | mean | exact two-sided sign-flip p | 95% t CI |
|---|---|---:|---:|---|
| 16-task | +0.00616, −0.00521, −0.01818, −0.00047 | −0.00443 | **0.625** | [−0.02079, +0.01194] |
| 8-task | +0.00385, −0.03636, −0.03125, +0.02273 | −0.01026 | **0.500** | [−0.05535, +0.03483] |

这些区间只说明“极不精确”，而且由于 served policy 与 RNG 配对无效，它们不应再被当作算法效应 CI。

## 5.2 n=4/5 与 `p<0.01` 预注册互相矛盾

对于 n 个 paired outer units 的 exact two-sided sign-flip test，最小 p 值为 `2/2^n`：

| n | 最小双侧 p |
|---:|---:|
| 4 | 0.125 |
| 5 | 0.0625 |
| 7 | 0.015625 |
| 8 | 0.0078125 |

合同要求 ≥5 seeds 且 `p<0.01`，实际只跑 4 seeds；即便跑 5 seeds，也不可能达到阈值。若 outer unit 就是 seed/domain stream，主实验至少需要 8 个独立 paired units，通常还应更多以获得功效。

## 5.3 论文声称 hierarchical bootstrap，但没有用于主结果

`paired_hierarchical_bootstrap` 的函数存在，但统计脚本和论文数字没有调用它：

- [`bootstrap.py#L39-L64`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/src/agent_ttrl/evaluation/bootstrap.py#L39-L64)

论文没有：

- 主 contrast 的 hierarchical CI；
- effect size；
- task-family clustering；
- environment/model heterogeneity；
- multiplicity correction；
- calibration-based SESOI/power analysis。

## 5.4 “null”不能等于“no effect”

即便执行正确，当前 n=4 的宽 CI 也只能叫 inconclusive，不能叫 reproducible negative effect。现在又存在 static serving，因此应写成：

> execution-invalid；不进入 meta-analysis，不给算法 PASS/FAIL。

---

## 6. 实验完整性审计

| 审计轴 | 判定 | 证据与解释 |
|---|---|---|
| A. ground-truth provenance | **FAIL/WARN** | CTS 是 synthetic oracle；AppWorld hidden evaluator真实但执行路径坏；tau2 是自定义 proxy；M4 true harm 是合成标签。 |
| B. normalization | **PASS with caveat** | AUPC 是简单均值，group z-score 本身无自我归一化造假；但 reward 定义与任务成功脱节。 |
| C. result existence/provenance | **FAIL** | summary manifests 存在，但多处 metadata 错，缺 raw logs/adapters/hashes/configs；旧 seed manifests 被覆盖。 |
| D. dead/ineffective code | **FAIL** | serving update 不存在；AppWorld update no-op；CTS EGC 零梯度；random_branch 未实现；tests 主要做字符串检查。 |
| E. scope matching | **FAIL** | synthetic/proxy/smoke 被扩展成 public-env end-to-end、safe commit、general negative result。 |
| F. evaluation type | **MIXED/UNCLEAR** | CTS=`simulation_only`；M4=`simulation_only`；tau2=`custom_proxy`；AppWorld=`real_gt_but_invalid_run`。 |

结论：`experiment_integrity_status = FAIL`。

---

## 7. 可复现性：130 tests 通过，但并不等于论文可复现

## 7.1 测试覆盖缺口

测试全过是好事，但 `test_stream_script_gates.py` 主要用 regex 检查源代码是否包含某个字符串：

- [`test_stream_script_gates.py#L16-L34`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/tests/unit/test_stream_script_gates.py#L16-L34)

它不会发现：

- served policy 没更新；
- prompt/completion token 拼错；
- branch action identity 错；
- world reset 后执行；
- RNG schedule 混杂；
- evaluator 与官方 benchmark 不一致。

必须增加 GPU/end-to-end integration tests，而不是继续增加静态 contract tests。

## 7.2 `reproduce.sh` 只复现展示层

脚本实际做的是：

1. 检查 manifest 路径是否存在；
2. 重跑 synthetic M4；
3. 重画图；
4. 编译 PDF。

见 [`reproduce.sh#L17-L79`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/reproduce.sh#L17-L79)。

它没有：

- 校验 manifests against schema；
- 校验 manifest hash/commit/model revision；
- 重跑任何 GPU/agent experiment；
- 下载并 pin official benchmark；
- 复建 adapters；
- 验证 raw logs 与 summary 一致；
- 重算 hierarchical stats；
- 检验 serving policy change。

因此不能输出 `REPRODUCIBLE: all checks passed` 并把它解释为论文实验可复现。更准确的名字是 `rebuild_paper_from_committed_summaries.sh`。

## 7.3 图表数据问题

- Fig.3 的均值手写：[`make_figures.py#L183-L203`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/make_figures.py#L183-L203)。
- Fig.5 的 cost `[1,8,5,9]` 明确是 approximate sketch，不是 canonical ledger：[`#L249-L269`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/make_figures.py#L249-L269)。
- Fig.4 只画 catastrophic=0，不画 false rollback 和 retained gain，视觉上放大安全结论：[`#L207-L244`](https://github.com/hxm2023/agent-ttrl/blob/76b02998eff8e5034d4f2143c14fe0b0ba289737/scripts/make_figures.py#L207-L244)。

正式论文所有图必须从一个 immutable analysis table 生成；不能再混用 hardcoded summaries。

---

## 8. 新颖性与顶会差距

## 8.1 EGC 的近邻工作比论文描述得更强

- **CVT-RL** 已经提出 policy-conditioned counterfactual contribution、controlled interventions、validity gating、doubly robust adjustment，并在长程 agent 上报告正结果：[CVT-RL](https://arxiv.org/abs/2606.05263)。
- **Tree-RL/SHEPHERD、APPO** 已经使用分叉/兄弟轨迹作局部 advantage。
- **StarOR** 已经在 test time 用 MCTS siblings + transient LoRA-GRPO 做 instance-specific refinement，并报告 4B 正结果：[StarOR](https://arxiv.org/abs/2606.15197)。
- **aTTT** 已经在 live agent episode 内做 LoRA test-time update，并实现 runtime serving，报告 ALFWorld/SWE-bench 正提升：[aTTT](https://arxiv.org/abs/2607.03441)。

当前论文的潜在增量不是“第一次 counterfactual credit”或“第一次 agent LoRA TTRL”，而只能是：

> **在连续的、相关但未见过的 stateful tasks 上，仅使用 partial endogenous evidence，执行 policy-consistent cross-task LoRA-RL，并测量 inductive future transfer。**

这个增量必须靠真实正结果、严格 matched budget 与 credit mechanism evidence 建立。现在没有。

## 8.2 SafeCommit 与 PACE 的区分不足

PACE 已把 self-evolving agent 的 commit decision 定义成 paired anytime-valid test。把 candidate 从 prompt 换成 LoRA adapter，不自动构成算法创新。

要有新贡献，至少需要：

- candidate 由 online RL 生成并产生 policy feedback；
- gain 与 anchor harm 是两个约束而非单一 correctness；
- actual adapter candidates、fresh sentinel、stream-level factorial；
- 有效且明确的 per-candidate / across-candidate error guarantee；
- 比直接 PACE baseline 有可解释的功效—风险优势。

当前实现没有这些证据。

## 8.3 “把多个正确模块串起来”不是顶会算法贡献

Guard、ledger、prequential、counterfactual branch、SafeCommit 分别合理，但组合本身需要回答一个新问题：

> 为什么这种组合产生了以前做不到的、可重复的未来迁移？

如果最终仍没有正效果，顶会算法主线很难成立。可以转为 systems/audit/benchmark paper，但标题、实验和 venue 都应重构。

---

## 9. 写作、引用与版式问题

## 9.1 关键措辞必须删除或降级

当前不能使用：

- `validated protocol machinery end-to-end`
- `all methods under identical three-channel hard caps`
- `EGC on AppWorld/tau2`
- `SafeCommit e-process / anytime-valid`
- `fully reproducible harness`
- `reproducible negative result on LoRA-RL effectiveness`
- `strong updates measurably change behavior`

“后续 first-attempt 分数有变化”不能证明 policy 行为因 update 改变，因为 arm 的 request RNG 不同且 served model 静态。

## 9.2 Related Work 中出现不存在的实验

论文写：

> “our budget-matched baselines include them”

但主结果没有 ACE、OLIVIA、JitRL、aTTT、CausalFlow-style repair 等。要么实际运行并报告，要么改为“required future baselines”。

## 9.3 引用元数据错误

内部 `CITATION_AUDIT` 声称 24 KEEP/1 FIX，但至少三条作者元数据错误：

- CVT-RL 的作者是 **Renwei Meng**，不是 `Wang and others`：[arXiv 2606.05263](https://arxiv.org/abs/2606.05263)。
- PACE 的作者是 **Zayx Shawn**，不是 `Mukherjee and others`：[arXiv 2606.08106](https://arxiv.org/abs/2606.08106)。
- StarOR 的作者首位是 **Jiajun Li**，不是 `Zhang and others`：[arXiv 2606.15197](https://arxiv.org/abs/2606.15197)。
- τ²-Bench 应优先引用 Victor Barres 等人的正式论文，而不是只写 `Sierra Research` GitHub misc：[arXiv 2506.07982](https://arxiv.org/abs/2506.07982)。

`CITATION_AUDIT.md` 与 JSON 的 KEEP/FIX 计数也互相不一致。提交前必须重新做逐条 existence/metadata/context audit。

## 9.4 PDF 视觉问题

- 第 4 页公式中的 `ALLOW` 穿过栏间并覆盖右栏正文，是直接的 camera-ready blocker。
- 第 5 页一次堆四幅图，字号极小；图 2/3/4 在正常缩放下难以阅读。
- 大量 figure floats 被推到结论之后；Table 1 也在结论文本之后出现，论述和证据分离。
- Fig.4 标题写 “eliminates”，但不显示 false rollback，属于视觉叙事偏置。
- Fig.5 是 approximate sketch，却占据主文 figure slot；顶会主文应换成真实 Pareto/ledger curve。
- 主方法没有清晰 Algorithm box；implementation simplification 与 nominal method 分裂，读者难以知道实际运行的算法。

## 9.5 篇章结构建议

当前 7 页不应继续硬塞 6 图。重写后建议：

1. Introduction：问题、发现、真实贡献，1 页。
2. Setting + identifiability：prequential、partial evidence、policy consistency，0.75 页。
3. Method：一个可执行 Algorithm 1 + serving transaction，1.5 页。
4. Experimental protocol：official env、split、caps、stats，1 页。
5. Main results：1 主表 + 1 curve，1.5 页。
6. Mechanism + safety：1 表/图，1 页。
7. Limitations/conclusion，0.5 页。
8. 其余工程细节、manifest schema、所有 secondary plots 放 appendix。

---

## 10. 正结果救援的核心原则

“不要只有负结果”不等于事后调到正。正确策略是提高问题可学习性和实验识别力，并预先设 Go/Kill Gate：

```text
先证明更新真的进入 serving
→ 再证明该任务流在 oracle/accessibile evidence 下可学习
→ 再证明 EGC credit 比 naive credit 更准
→ 最后才在 sealed public streams 检验未来迁移
```

如果 oracle upper bound 都不能在后续未见任务上改善，那么继续扩 seed、加模型或调 gate 没有意义。

---

## 11. 建议的新论文主线

### 11.1 推荐主线：Policy-Consistent EGC-TTRL

建议把 v2 的核心问题改成：

> **在具有可复用 latent workflow 的连续 stateful-agent tasks 上，怎样保证每次 deployment-time LoRA update 真正、原子地进入后续 rollout policy，并利用 snapshot-matched partial-evidence credit 获得正的 future transfer？**

比当前标题更诚实且更有区分度的候选标题：

> **Policy-Consistent Evidence-Gated Test-Time RL for Stateful Tool Agents**

或：

> **When Does Agent Test-Time RL Actually Learn? Serving-Consistent Counterfactual Credit under Partial Evidence**

新贡献应分为：

1. **Policy-consistent runtime transaction**：candidate adapter、served version、producer tokens/logprobs、request RNG 和 environment snapshot 原子绑定。
2. **Matched partial-evidence action credit**：真正的 G×R counterfactual branch，而非 trajectory z-score。
3. **Cross-task evidence replay**：把多个相关任务上的可靠 action rows 累积成可学习 batch，解决单任务 4-row update 信噪比过低的问题。
4. **Actual-adapter commit control**：在真实 candidates 上与 PACE 对比，报告风险和 retained gain，不再只做合成“0 catastrophic”。
5. **Learnability frontier**：量化 task recurrence × evidence coverage × update batch size 何时产生 future transfer。

### 11.2 为正结果增加一个必要但克制的方法组件：Cross-task Evidence Replay

当前每个任务只用极少 rows 做 1–8 次重复 update，既弱又噪。建议引入 session-scoped replay buffer：

```text
可靠的 action-level UpdateRows
    ↓ 按 tool intent / state signature / task family 分桶
累积 M 个任务或 K 个有效 rows
    ↓ 去重、recency weighting、anchor rehearsal
一次小步 LoRA update
    ↓ served-policy atomic commit
```

建议默认：

- 每 4–8 个 tasks 或累计 64–128 个有效 action rows 更新一次；
- positive/negative signed rows 均保留；
- 只使用 accessible evidence；
- 每个 row 保留 original producer token/logprob/hash；
- 加 10–20% anchor/rehearsal rows 控制遗忘；
- 所有方法使用相同 buffer capacity 与 update-token cap。

这比盲目提高 learning rate 更可能产生稳定迁移，因为它把跨任务重复技能聚合成一个可辨识信号。

必须做的消融：

- per-task micro-update；
- replay without EGC；
- EGC without replay；
- EGC + replay；
- shuffled task-family replay；
- same number of ordinary rollouts。

如果 replay 才有效但 EGC 不增益，论文应诚实转为 evidence replay，而不是继续挂 EGC 标题。

---

## 12. v2 工程重构方案

## 12.1 Runtime：让训练 policy 真正成为 serving policy

建议新模块：

```text
src/agent_ttrl/runtime/
├── served_policy.py
├── adapter_transaction.py
├── request_seed.py
├── canary.py
└── policy_manifest.py
```

支持三种后端，按顺序实现：

1. **Correctness backend：HF colocated generation**
   - 训练和生成使用同一个 PEFT model；慢，但最容易证明 policy consistency。
   - 先用它跑 CTS overfit/positive upper-bound。
2. **Production backend：vLLM runtime LoRA API**
   - adapter 保存为 immutable directory；
   - server load candidate under new `policy_version`；
   - canary 通过后 atomic switch；
   - rollback 恢复 parent adapter。
3. **Weight-broadcast backend**
   - 只有在版本 API 和参数命名完全验证后使用 TRL communicator/update calls；
   - PEFT LoRA 需验证是 merge 后全量广播还是 runtime adapter，不能假定 parameter names 自动匹配。

每次 commit 必须输出：

```json
{
  "parent_policy_version": 7,
  "candidate_policy_version": 8,
  "base_model_sha256": "...",
  "adapter_sha256": "...",
  "served_adapter_sha256": "...",
  "server_ack_event_sha256": "...",
  "canary_prompt_set_sha256": "...",
  "parent_candidate_logit_kl": 0.0,
  "deterministic_output_changed": true,
  "commit_status": "COMMITTED|ROLLED_BACK"
}
```

`weight_delta>0` 不能再单独作为 canary pass。至少同时要求：

- server ack 的 adapter hash 与 candidate 一致；
- 独立 FP32 scorer 上 parent/candidate logits 超过预注册数值容差；
- 固定 prompts+seeds 至少一个 rollout 可观察变化；
- zero-LR candidate 不应产生变化。

## 12.2 Request-scoped RNG：消除 rollout schedule 混杂

每个请求的 seed 固定为：

```text
seed = H(protocol_hash, stream_seed, task_id, turn_id,
         policy_version, purpose, branch_group, action_id, continuation_id)
```

`purpose` 至少区分：

- `production_first_attempt`
- `within_task_recovery`
- `credit_action_proposal`
- `credit_continuation`
- `shadow_gain`
- `shadow_anchor`
- `canary`

更新 arm 多生成多少 branch，都不能改变未来 production request 的 seed。

## 12.3 正确的 EGC branch executor

建议模块：

```text
src/agent_ttrl/credit/
├── decision_selector.py
├── action_proposer.py
├── branch_executor.py
├── paired_credit_v2.py
└── row_materializer.py
```

执行顺序必须严格为：

```text
1. seal decision snapshot S_t
2. parent policy 采样 G 个去重 action，一次一个
3. 保存每个 action 的 producer prompt_ids/action_ids/logprobs/hash
4. 对 r=1..R：
      对所有 g：restore S_t
                 强制执行固定 action a_g
                 使用同一 continuation seed ξ_r
                 运行相同 continuation policy/horizon
5. 得到 U[g,r]
6. credit[g] 只映射回 action g 的原始 token span
7. Guard/ledger validate
8. update buffer
```

v2 primary 至少用 `G=4,R=4`。`R=2` 的 t-interval 极不稳定，只可作极小 smoke。

每个 branch record 必须带：

- snapshot hash before/after restore；
- action token artifact/hash；
- action semantic hash；
- continuation seed；
- exogenous/user simulator seed；
- policy version；
- evidence vector与 utility decomposition；
- charged env/model tokens；
- production-world non-interference proof。

## 12.4 官方环境 wrapper

### tau2

- pin 官方 commit/release，不再只写 GitHub latest；
- 使用官方 orchestrator、user simulator、domain policy 和 evaluator；
- 记录 agent/user turns 与双方 tool actions；
- 适应只读取 agent 当时可见 evidence，不读取 hidden criteria；
- production score 使用官方 task reward；partial action match 只能作 diagnostic；
- 先做 10 条保存轨迹的 parity test：wrapper score 必须和官方 CLI 完全一致。

### AppWorld

- 每个 production attempt 一次 world init，一次真实执行，一次 evaluate；禁止 replay；
- 每个 training/branch rollout 创建独立 snapshot/world process；
- reset 后不得继续 `/exec`；
- branch state restore 必须通过 state hash；
- 使用官方 allowed apps/API docs 和 evaluator；
- 不再只取前三个同 ID 前缀任务。

### ControlledToolShift

- 保留为 mechanism/upper-bound 环境；
- 增加真实 latent skill families，而不只是替换 SKU/address；
- 每个 family 有 adaptation/dev/sealed task templates；
- hidden oracle 只用于 estimator fidelity，不进入 public primary update。

## 12.5 Manifest 与 artifact contract

每个 run 至少包括：

```text
run_id
execution_status / audit_status / claim_outcome
code_commit / dirty_worktree
protocol_sha256 / config_sha256
environment_commit / data_manifest_sha256 / split_role_sha256
model_revision / tokenizer_revision / chat_template_sha256
seed list / per-request seed namespace version
parent and served policy versions / adapter hashes
all first-attempt trajectory artifact hashes
all UpdateRow/branch/evidence hashes
3-channel ledger + conservation audit
raw metric table sha256
analysis script commit
allowed_claim / forbidden_claims / reason_codes
```

旧 manifests 不修改；为 v2 新建 protocol namespace，避免把 invalid 运行伪装成已修复运行。

---

## 13. 让“正结果”变得科学上可期待的实验设计

## 13.1 先选具有可迁移结构的 domain session

当前 random first-20 tasks 未必共享可学习 latent structure。Test-time cross-task RL 需要重复但不相同的技能。建议每个 stream 明确一个 latent family，例如：

- policy-rule transfer：退款/换货/取消在不同用户、商品、状态上的同一规则；
- tool-composition transfer：search→verify→mutate→confirm 的重复 workflow；
- error-recovery transfer：permission/schema/API failure 后的共同恢复模式；
- dual-control coordination：指导用户执行一类共同操作，但参数与上下文未见；
- safety/invariant transfer：receipt 与 state invariant 冲突时优先验证。

任务 ID、实体、语言表述和部分工具组合应未见，但 latent workflow 可重复。否则“未来迁移”在统计上没有信号来源。

要报告：

- within-family transfer；
- leave-one-template-out transfer；
- cross-family negative control；
- sealed domain shift。

## 13.2 避免 floor/ceiling：选择有基础能力的模型

AppWorld Qwen3-4B 为 0，tau2 proxy 仅数个百分点，不能用于检验细粒度 credit。aTTT 也观察到增益主要出现在模型已有任务能力但会漂移的区域。

主模型建议使用：

- 一个 8B/14B、原生 tool-use 较好的开放模型作为 primary；
- 第二个独立架构/训练家族的 7B/8B 模型作 generality；
- 所有 arms 从同一个 tool-format warm start 开始。

只在 dev/calibration 上筛模型，冻结后再解封 test。主任务 base success 建议处于 20%–70% 区间；低于 10% 是 floor，高于 85% 是 ceiling。

可使用 constrained decoding/native function calling 消除无意义的 JSON/语法噪声。这样 credit 研究针对 state/action quality，而不是 parser 失败。

## 13.3 Learnability upper-bound ladder

在任何大规模主实验前，逐级验证：

| Level | 信号 | 目的 | 允许的 claim |
|---|---|---|---|
| L0 | hidden oracle on CTS，仅 diagnostic | 检查 serving/update 能否学习 | 不进入 public result |
| L1 | exact accessible state-delta verifier | 检查合法 evidence 能否产生正迁移 | controlled evidence |
| L2 | terminal accessible verifier + cross-task replay | 检查 naive TTRL 是否有正 upper bound | feasibility |
| L3 | EGC exact branches | 检查 credit 是否优于 terminal/unpaired | mechanism |
| L4 | public official env | 检查外部效度 | primary claim |
| L5 | EGC + real SafeCommit | 检查 risk-adjusted retained gain | C2 |

Go Gate 建议：

- L0/L1 在 CTS leave-one-template-out stream 上，future AUPC 相对 frozen 至少 +0.05，且 95% paired CI 方向为正；
- 如果 L0 失败：修 runtime/model/task recurrence，不准调 EGC；
- 如果 L0 过、L1 失败：accessible evidence 不够，改 reward/evidence；
- 如果 L1/L2 过、L3 失败：EGC 无贡献，停止 EGC headline；
- 只有 L3 过，才进入两公共环境主实验。

这些是 v2 预注册建议，不是现有结果。

---

## 14. 必跑实验矩阵

## 14.1 Block A：端到端正确性

| ID | 实验 | 成功标准 | 失败处理 |
|---|---|---|---|
| A0 | zero-LR, same rollout schedule | 各 arm production first attempts bitwise identical | RNG/serving 未隔离，停止 |
| A1 | one-task intentional overfit | served adapter hash 改变；固定 canary logits/output 改变 | policy sync 失败，停止 |
| A2 | rollback | rollback 后 canary 与 parent 恢复一致 | transaction 失败，停止 |
| A3 | branch isolation | 所有 branches 后 production snapshot hash 不变 | environment invalid，停止 |
| A4 | action-row mapping | 每个 credit 只对应同一固定 action token span | EGC invalid，停止 |
| A5 | official-eval parity | wrapper 与官方 CLI 10/10 trajectory scores 相同 | public env invalid，停止 |
| A6 | ledger conservation | 独立 recount 与 canonical ledger 完全一致 | cost claim invalid |
| A7 | split enforcement | 故意越权 task ID 被 fail-closed | sealed claim invalid |

## 14.2 Block B：机制验证

至少比较：

1. terminal trajectory reward；
2. unpaired branches；
3. random decision state；
4. equal-extra ordinary rollouts；
5. exact-oracle counterfactual diagnostic；
6. EGC；
7. EGC without reliability gate；
8. EGC + replay；
9. shuffled evidence negative control。

指标：

- credit sign accuracy；
- Spearman correlation with controlled oracle advantage；
- mean squared error；
- gradient variance；
- nonzero-credit coverage；
- invalid/no-support rate；
- update KL 与 action-token drift；
- future AUPC under matched caps。

机制 Gate：

- EGC sign accuracy 比 random/unpaired 高至少 10pp；
- rank correlation 至少高 0.15；
- nonzero-credit groups 处于 10%–90%，不能全关或全开；
- equal-extra rollout 不能解释全部 AUPC gain；
- credit fidelity 与 downstream gain 在 stream 间正相关。

## 14.3 Block C：public-env positive pilot

只用 dev/calibration，2–3 seeds 探索方向，不做显著性声明：

| arm | 作用 |
|---|---|
| frozen | 静态基线 |
| Best-of-N / self-reflection | inference-compute baseline |
| aTTT-style update | 最近的 live LoRA TTT baseline |
| naive accessible-evidence RL + replay | 判断参数更新上限 |
| EGC + replay | C1 proposed |
| EGC + replay + SafeCommit | full |

Pilot Go：

- base success 不处于 floor/ceiling；
- naive upper bound 在两个 dev streams 都比 frozen 正向；
- EGC 相对 naive 至少在机制指标上明显更好，并在 AUPC 上同向；
- no runtime/split/ledger violation。

若方向不满足，不进入 test；记录 valid FAIL 并选择 portfolio/audit route。

## 14.4 Block D：顶会主实验

建议至少：

```text
2 public environments
× 2 model families
× 8 paired seeds（或 power analysis 得到更多）
× 4 independent domain streams per seed
× frozen / strongest nonparam / naive RL / aTTT / EGC / full
```

Primary contrasts：

1. `EGC+replay - naive+replay`：credit contribution；
2. `full - EGC+replay`：SafeCommit risk/gain tradeoff。

Primary endpoints：

- `AUPC_prequential`；
- `sealed_future_holdout_score`；
- `catastrophic_update_rate` 作为 safety co-primary 或严格 secondary。

成功阈值可沿用原设计：两个公开环境中 paired 95% CI 下界 >0，平均绝对 AUPC gain ≥0.03 或 relative error reduction ≥8%；sealed holdout 同向。

## 14.5 Block E：真实 SafeCommit

第一版不要再自创未证明的 EB e-process。建议：

- 直接实现 PACE binary paired gate 作为强 baseline；
- 连续 bounded score 使用经过正式定理验证的 fixed-n bound 或标准 confidence sequence；
- 清楚区分 per-candidate error 与 across-candidate FWER；
- α-spending/online error control 在 protocol v2 预先冻结；
- 使用真实 LoRA candidate archive，而非 Gaussian synthetic labels。

对比：

- always commit；
- always rollback；
- fixed point threshold；
- fixed-n paired test；
- PACE；
- proposed two-constraint gain+anchor gate；
- oracle audit only。

必须同时报告：

- false commit；
- false rollback；
- commit rate；
- retained future gain；
- worst anchor drop；
- evaluation cost；
- end-to-end AUPC；
- 每个 candidate 的 parent/candidate policy hashes。

C2 只有在实际 adapters 上保留 ≥80% C1 gain 且降低 harmful commits 时才成立。

---

## 15. 统计协议 v2

1. **Outer units**：seed × independent domain stream；不能把 token/branch/task 当独立样本。
2. **配对**：同一 task order、production request seed、user simulator seed、base policy、caps 在所有 arms 一致。
3. **样本量**：先用 label-blinded calibration variance 做 power analysis；若坚持 exact two-sided `p<0.01`，至少 8 independent paired outer units。
4. **主 CI**：paired hierarchical bootstrap，outer resample domain streams，inner resample task families；同时给 mixed-effects sensitivity。
5. **SESOI**：test 解封前固定，不能见结果后降低。
6. **多重比较**：只冻结两个 primary contrasts；其余明确 exploratory。
7. **缺失/失败**：环境 crash、invalid manifests、zero-support groups 预先定义处理；不得静默删除。
8. **Invalid vs FAIL**：serving/split/ledger/eval violation=`INVALID`；执行正确但未过效果门槛=`VALID FAIL/INCONCLUSIVE`。
9. **报告**：point estimate、95% CI、standardized effect、raw per-stream data、p value；不要只写“方向一致”。
10. **不追 seed**：完成预注册 seed 后封盘，不因接近显著而续跑。

---

## 16. 预计算力与止损

以下是量级规划，不是精确报价；agent environment wall-clock 往往由 API/tool execution 和长 generation 主导，GPU 利用率未必满。

| 阶段 | 目标 | 预计 GPUh | 是否值得继续的条件 |
|---|---|---:|---|
| P0 correctness | serving sync、RNG、branch、official eval parity | 20–40 | 所有 A0–A7 通过 |
| Controlled positive upper bound | CTS L0–L3、replay、credit fidelity | 40–80 | oracle/access evidence 有明显 future gain |
| Public dev pilot | 1 model × 2 env × 2–3 seeds × 精简 arms | 80–160 | naive 和 EGC 都出现预注册正方向 |
| Workshop/portfolio | 1–2 env、3–5 seeds、主要消融 | 180–350 | claim 只到实际证据范围 |
| 主会 full matrix | 2 env × 2 models × ≥8 seeds × strongest baselines | 800–1600 | pilot Gate 已通过 |

在 2×RTX 6000D 上，理论最短 wall time 约为 GPUh/2，但 environment 串行、checkpoint、shadow eval 和调度会显著拉长。

止损规则：

- 40 GPUh 内 policy-sync correctness 不过：停止实验，只修工程。
- 120 GPUh 内 oracle/access upper bound 不过：停止 Agent-TTRL 正结果路线。
- 250 GPUh 内 EGC mechanism 不优于 terminal/unpaired：删除 EGC headline。
- public pilot 不同向：不解封主 test，不靠更强模型事后救同一 protocol。

---

## 17. 8 周执行建议

### Week 1：冻结 v2 protocol，清理有效性状态

- 将旧 M3/M5/M6 标成 `INVALID`，reason codes：
  - `STATIC_SERVED_POLICY`
  - `RNG_SCHEDULE_CONFOUND`
  - `NON_COUNTERFACTUAL_EGC`
  - `BRANCH_ACTION_IDENTITY_BREACH`
  - `ENV_RESET_BREACH`
  - `OFFICIAL_EVAL_BYPASS`
  - `LEDGER_NOT_APPLIED`
  - `SPLIT_MANIFEST_BYPASS`
- M4 标为 `SIMULATION_ONLY / THEORETICAL_GUARANTEE_UNSUPPORTED`。
- 新建 protocol v2，不覆盖旧 artifacts。

### Week 2：runtime policy consistency

- HF colocated correctness backend；
- vLLM runtime LoRA transaction；
- request-scoped RNG；
- parent/candidate/rollback canaries；
- A0–A2 tests。

### Week 3：branch 和 environment correctness

- 重写 G×R branch executor；
- action-token artifact mapping；
- branch isolation；
- official tau2 wrapper parity；
- AppWorld per-rollout world lifecycle；
- A3–A7 tests。

### Week 4：learnability upper bound

- CTS latent skill families；
- oracle → accessible verifier ladder；
- per-task vs cross-task replay；
- 决定是否存在可学的正迁移。

### Week 5：EGC mechanism

- terminal/unpaired/random/equal-extra controls；
- sign accuracy、rho、MSE、gradient variance；
- 固定 EGC v2 或执行 Kill。

### Week 6：public dev pilot

- 8B primary；
- tau2 official + AppWorld；
- frozen/BoN/aTTT/naive/EGC；
- 2–3 seeds，只做 Go/Kill。

### Week 7：真实 SafeCommit 与第二模型

- PACE baseline；
- actual adapter archive；
- two-constraint gate；
- second model smoke 和方向性检查。

### Week 8：冻结主实验或诚实 pivot

- 若所有 Gates 通过：冻结 ≥8 seeds 主表 protocol，开始规模化运行；
- 若 EGC 失败但 runtime/replay 成功：改为 policy-consistent Agent TTRL / evidence replay；
- 若正迁移整体失败：改为 Agent-TTRL integrity benchmark / systems audit，不再投稿算法主会。

---

## 18. 建议新增的关键测试

```text
tests/integration/test_served_policy_changes_after_commit.py
tests/integration/test_zero_lr_rng_matched_identity.py
tests/integration/test_atomic_rollback_restores_parent.py
tests/integration/test_branch_action_fixed_across_continuations.py
tests/integration/test_branch_does_not_mutate_production_world.py
tests/integration/test_update_row_uses_producer_prompt_ids.py
tests/integration/test_vllm_served_adapter_hash.py
tests/integration/test_tau2_official_score_parity.py
tests/integration/test_appworld_world_lifecycle.py
tests/integration/test_split_role_fail_closed.py
tests/integration/test_ledger_external_recount.py
tests/statistical/test_exact_two_sided_signflip.py
tests/statistical/test_fwer_definition.py
tests/statistical/test_eprocess_supermartingale_adversarial.py
```

其中前 11 个必须是真实运行或最小 GPU integration test，不能再用 regex 搜源码代替。

---

## 19. 论文重写方案

## 19.1 当前稿件处理

建议立即：

1. 不提交当前 PDF。
2. 在 README 顶部加醒目状态：`CURRENT MAIN RESULTS INVALIDATED BY SERVED-POLICY AUDIT`。
3. 保留旧稿和 manifests 作为 `v0 forensic snapshot`，不要删除或改写历史。
4. 创建 `AUDIT_INVALIDATION.md`，逐条列出 invalid reason codes。
5. 等 v2 至少完成 correctness + positive pilot 后再写新论文，不要在旧稿上继续补句子。

## 19.2 正结果版本的主表应长什么样

主表至少包含：

| Env / Model | Frozen | Best nonparam | aTTT | Naive RL+Replay | EGC+Replay | Full+Gate | Δ EGC−Naive [95% CI] | Sealed Δ | B_env/B_model/B_update |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|

另一个 mechanism table：

| Credit method | sign acc | Spearman | MSE | nonzero coverage | grad var | future Δ | cost |
|---|---:|---:|---:|---:|---:|---:|---:|

SafeCommit figure 必须是 risk–retained-gain Pareto，而不是只画 catastrophic=0。

## 19.3 条件式摘要模板

以下只能在相应结果真实出现后填数字：

> Deployment-time parameter updates for tool agents are difficult to evaluate because the policy used for later rollouts can silently diverge from the policy updated by the trainer, while partial execution evidence is endogenous and noisy. We introduce **Policy-Consistent EGC-TTRL**, an online LoRA-RL system that atomically binds served adapter versions, producer tokens and log-probabilities, request-scoped randomness, and snapshot-matched counterfactual branches. A cross-task evidence replay buffer aggregates reliable signed action credit across related but unseen tasks. On **[two official environments]** with **[two model families]**, our method improves prequential first-attempt AUPC over matched naive online RL by **[Δ, 95% CI]**, with the same environment/model/update-token caps; controlled experiments show **[credit fidelity gain]**. A paired commit gate evaluated on actual candidate adapters reduces **[harm metric]** while retaining **[x%]** of the prequential gain. All primary runs use official evaluators, sealed future holdouts, request-level RNG pairing, and immutable policy/artifact manifests.

不能出现占位符对应证据之前的任何“eliminates”“validated”“fully reproducible”。

---

## 20. 若始终没有算法正结果，仍可形成的高质量替代工作

### Route B：Agent-TTRL Integrity Benchmark / Systems Audit

如果 v2 证明很多 apparent gains 来自 serving/RNG/evaluator bugs，而修正后增益消失，可以转为：

> **Silent Failure Modes in Agent Test-Time RL: Served-Policy Drift, RNG Confounding, and Stateful Evaluation Leakage**

需要的正面贡献不再是任务成功率，而是：

- 构建 12–20 个可复现 fault injections；
- 证明哪些 bug 会制造 phantom gain/phantom negative；
- 给出 policy-consistent reference implementation；
- 在多个开源 TTRL stacks 上复现 failure prevalence；
- 给出 end-to-end audit suite 和 overhead；
- 展示修正后至少一个小规模真实 learning case。

这是一条 systems/evaluation 论文路线，可能比当前“无效负结果 + 合成安全 gate”更有说服力，也更能体现后训练工程能力。

但若只审自己的一个仓库，没有跨实现证据，也不足以成为顶会论文。

---

## 21. 求职/简历层面的诚实写法

当前不建议把论文写成“提出 EGC-TTRL 并证明 SafeCommit 消除灾难更新”。可以写：

> **Agent-TTRL｜大模型推理时强化学习与训练一致性审计（开源）**  
> 搭建 stateful tool-agent 的 prequential LoRA-RL 原型与实验审计框架；在端到端复核中定位 rollout serving 未同步、stateful branch 污染、request RNG 混杂和统计检验错误，并重构 policy-version/adapter-hash/producer-token/ledger 绑定的在线更新链。实现 CTS、schema、Guard integration 与 130 个 CPU tests，保留 invalid/negative runs 及 reason-coded failure reports。

等 v2 真正出现正结果后再加入：

> 在官方 tau2/AppWorld 的相关任务流上，相对 matched naive RL 提升 prequential AUPC **X**，并将 rollout-policy sync、branch isolation 与真实 adapter commit/rollback 做成可复现 integration tests。

这比把当前失效实验包装成“顶会负结果”更能得到后训练面试官信任。

---

## 22. 优先级清单

### P0：不修就不能再跑实验

- [ ] 将 M3/M5/M6 现有结果标记 INVALID。
- [ ] 实现 served-policy sync 与 hash/behavior canary。
- [ ] 加 request-scoped RNG；完成 zero-LR matched control。
- [ ] 重写 CTS G×R action/continuation 与 producer token mapping。
- [ ] 修 AppWorld world lifecycle、double replay、undefined `prompt_ids`。
- [ ] 用官方 tau2 orchestrator/user simulator/evaluator。
- [ ] 把 Guard/ledger/split enforcement 真正接入 formal stream。
- [ ] 撤回 SafeCommit anytime-valid/e-process 声明。
- [ ] 修双侧统计，并重新做 power design。

### P1：决定能否得到正结果

- [ ] 建 CTS learnability upper bound。
- [ ] 设计 latent-skill-related task streams。
- [ ] 引入 cross-task evidence replay。
- [ ] 选择非 floor 的 8B/14B primary model。
- [ ] 做 terminal/unpaired/random/equal-extra/aTTT baselines。
- [ ] 冻结 dev pilot Go/Kill Gate。

### P2：顶会证据

- [ ] 两个官方 public environments。
- [ ] 两个模型家族。
- [ ] ≥8 independent paired units 或正式 power analysis 更大样本。
- [ ] sealed future holdout。
- [ ] real adapter SafeCommit + PACE baseline。
- [ ] immutable raw artifacts、analysis table、ledger、policy manifests。
- [ ] 外部无上下文 reviewer 无 BLOCKER/HIGH。

### P3：写作

- [ ] 重新做 citation audit。
- [ ] 删除所有未执行/过度 claim。
- [ ] 一个 Algorithm box 对应实际代码。
- [ ] 修公式跨栏和 figure readability。
- [ ] 所有图从 immutable results table 自动生成。

---

## 23. 模拟顶会审稿意见

### Summary

The paper proposes evidence-gated counterfactual test-time RL for stateful tool agents, combining paired branch credit, session LoRA updates, a statistical commit gate, and prequential evaluation. The problem is timely and the authors have invested in protocol documentation and an open scaffold. However, the released implementation does not realize the claimed learning loop: the LoRA-updated HF model is never synchronized to the vLLM server used for subsequent first attempts; the tau2 “EGC” arm is not counterfactual; the CTS branch/action mapping is inconsistent; and the AppWorld update path runs after the environment is reset. The claimed public-environment negative result is therefore not identifiable. The SafeCommit guarantee is also unsupported: the exponential compensation is invalid for the stated range, the primary empirical-Bernstein routine is not shown to be an e-process, and experiments use a hand-generated Gaussian archive rather than actual adapters. Statistics are underpowered and the reported “two-sided” permutation p-values are computed one-sided. Passing unit tests and rebuilding figures do not address these end-to-end failures.

### Strengths

- Important deployment-time learning problem.
- Clear intended evidence-tier and prequential protocol.
- Open source, extensive component tests, useful failure logs.
- Good potential systems story around policy identity and rollback.

### Major weaknesses

1. No served-policy update in formal streams; primary effect estimates are invalid.
2. Proposed EGC is not executed on tau2/AppWorld and is malformed on CTS.
3. Public benchmark evaluation does not follow official tau2 protocol.
4. SafeCommit has neither a valid stated guarantee nor real-adapter evidence.
5. Missing matched baselines, real cost ledgers, split enforcement, sealed holdout and adequate statistics.
6. Reproducibility claims exceed the released artifacts.
7. Several citations and visual elements are submission-blocking.

### Questions for the authors

1. Which exact adapter hash generated each subsequent first attempt, and where is the server-side load/ack event?
2. Why do naive and EGC match on all 96 task outcomes, and what zero-LR RNG-matched control rules out sampling-stream displacement?
3. How can tau2 be called EGC without a decision snapshot or G×R action-continuation matrix?
4. How are R credit rows mapped to fixed actions when each `(g,r)` call regenerates an action?
5. Why does AppWorld reset before update rollout execution, and where is `prompt_ids` defined?
6. What theorem establishes the claimed empirical-Bernstein e-process for `[-1,1]` data?
7. Why is a mean-commit statistic called FWER, and why is PACE not directly implemented?
8. Where are the actual 3-channel ledgers and split hashes for Table 1?

### Recommendation

**1/10 — Strong Reject.** A new submission would require rebuilding the runtime/evaluation path and rerunning the experiments; textual revision of the current draft is insufficient.

---

## 24. 最终判断

当前仓库更像：

> 一个规格设计很强、组件较多、失败日志诚实，但主实验闭环尚未真正成立的 Agent-TTRL research scaffold。

它还不是：

> 一篇证明 EGC-TTRL 无效或 SafeCommit 有效的顶会论文。

最值得做的不是继续在旧结果上补图、加 seed 或换更大模型，而是先完成以下闭环：

```text
训练后的 adapter
→ 原子进入 serving
→ 固定 request RNG 生成下一任务 first attempt
→ official stateful evaluator 评分
→ exact branch 从同一 snapshot 产生 action credit
→ Guard/ledger/split 全链记录
→ 相关任务 replay 产生可学习信号
→ sealed stream 上得到 paired positive transfer
```

一旦这个闭环成立，即使最后效果有限，项目也会成为很强的后训练工程作品；若 EGC+replay 能在两个官方环境上稳定超过 naive RL，它才重新具备顶会算法论文资格。


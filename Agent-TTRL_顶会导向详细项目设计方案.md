# Agent-TTRL：顶会导向的详细项目设计、实验协议与求职交付方案

> 文档状态：`RESEARCH_PROPOSAL / NOT_IMPLEMENTED`  
> 冻结日期：2026-08-22  
> 工作方法名：**Evidence-Gated Counterfactual Test-Time RL（EGC-TTRL，暂名）**  
> 目标读者：后续实现电脑、项目合作者、论文作者、技术面试准备者  
> 目标会议：优先按 ICLR / ICML / NeurIPS 主会的算法与实验标准设计；若最终贡献更偏语言 Agent 与评测，可转 ACL / EMNLP 主会  
> 当前 claim ceiling：这里只交付研究设计，不代表算法有效、创新性已最终确认、项目已经开源或论文必然录用

---

## 0. 一页结论

这个项目**可以成为简历核心项目**，但不能只是“在 Agent benchmark 上套 GRPO”，也不能把 Test-time Scaling、memory prompt 更新或拿测试标签训练统称为 Test-time RL。

推荐研究问题是：

> 在部署期连续遇到相关但未见过的状态化工具任务时，Agent 能否只利用当时可获得的、部分可靠的执行证据，进行 session-scoped 参数更新；并在固定交互预算下提升后续任务表现，同时控制错误奖励强化、负迁移和灾难性更新？

候选方法 EGC-TTRL 只保留两条论文主张：

1. **部分可验证条件下的反事实 evidence credit**：使用可重放环境的 paired branch evidence，为关键动作构造带置信度的 signed credit；在固定 deployment budget 下，比 trajectory-level/self-consistency reward 更有效地提升 prequential learning efficiency。
2. **风险受控的 adapter commit**：候选 LoRA 更新先进入 shadow evaluation，由预注册、跨 candidate 分配错误预算的 fixed-sample confidence Gate 决定 commit 或 rollback，从而降低 negative transfer / poisoned update，而不是默认每个 test-time gradient 都永久生效。

这两条仍是**候选创新**，不是已完成查新后的“首次”。截至 2026-08-22，Agent test-time adaptation 已经非常拥挤：GTTA、ACE、OLIVIA、MemoPilot、TTRL/T3RL 等工作分别覆盖了 adaptation vector、environment dynamics、context evolution、action-layer bandit、memory RL 和 tool-verified pseudo reward；CausalFlow 还已覆盖失败轨迹上的 step-level counterfactual intervention、causal responsibility 与最小修复。因此本项目必须通过 §4 的 Novelty Gate；若近期工作已经同时覆盖上述两条主张，应立即收缩或换题，不能只改名继续投稿。

项目执行顺序建议固定为：

```text
GRPO-Guard correctness Gate
        ↓
ControlledToolShift exact sandbox
        ↓
强 baseline 复现
        ↓
EGC credit 单机制实验
        ↓
SafeCommit 单机制实验
        ↓
AppWorld + tau2 公共环境
        ↓
第二模型家族、完整统计与论文
```

如果 GRPO-Guard 尚未证明 rollout policy、token、mask、behavior log-prob 和 update identity 闭合，本项目不得开始正式结果实验。否则旧 `grpo-credit-assignment` 的 silent off-policy 事故会原样重演。

---

## 1. 为什么值得做，以及什么时候不值得做

### 1.1 对求职组合的价值

完成后的理想项目组合是：

| 项目 | 主要证明能力 | 简历角色 |
|---|---|---|
| Agent-TTRL / EGC-TTRL | 在线 Agent RL、算法设计、连续适应、严谨评测 | Agent/后训练算法岗第一或第二核心项目 |
| GRPO Reward Hacking | reward robustness、研究能力、负结果与评测 | 研究核心项目 |
| GRPO-Guard | rollout/update identity、训练系统、故障注入、工程正确性 | 后训练工程核心项目 |
| Agent-RL Credit Auditor | estimand、matched cost、exact oracle、机制审计 | 内部工具或面试深挖，不再占主项目位 |

针对 Agent/后训练算法岗位，建议排序：

1. Agent-TTRL；
2. GRPO Reward Hacking；
3. GRPO-Guard。

针对华为计算产品线、训练系统或 AI Infra，建议排序：

1. GRPO-Guard；
2. Agent-TTRL；
3. GRPO Reward Hacking。

### 1.2 不能作为核心项目的版本

以下任一形态都不够：

- 只复现 TTRL，在数学题上做 majority vote + GRPO；
- 只把历史轨迹写进 prompt，却称“推理时强化学习”；
- 只做 Best-of-N、tree search 或 MCTS，却称“模型在线学习”；
- 拿 benchmark hidden evaluator 或 test answer 直接当 reward，再在同一批任务上报提升；
- 只报 post-update same-task success，不报 future-task/prequential performance；
- baseline 少算 branch、verifier、rollback 或 update 成本；
- 只跑一个 seed、一个环境、一个模型，并把最好点数写成稳定结论；
- algorithm gain 实际来自更多 generated tokens、更多 tool calls 或更大的 context；
- Guard 尚未过 Gate，却直接相信 rollout/update 链路；
- 只有设计文档、README 或架构图，没有可验证运行 artifact。

### 1.3 非目标

v1 不是：

- 任意真实生产 Agent 的通用安全保证；
- 对恶意 producer 的密码学防御；
- 完全无监督、完全无 verifier 的自我进化；
- 任意不可重置现实环境上的 counterfactual oracle；
- 新的通用 credit-assignment 理论；
- 完整重写 verl、TRL、OpenRLHF、vLLM 或 tau2；
- 为追求“顶会感”堆叠多个互不相关模块。

---

## 2. 术语、任务边界和可审计定义

### 2.1 四个容易混淆的概念

| 名称 | 测试时发生什么 | 是否更新参数 | 本项目中的用法 |
|---|---|---:|---|
| Test-time Inference / Scaling | 增加采样、搜索、投票、验证 | 否 | baseline |
| Test-time Context Learning | 更新 memory、playbook、prompt | 否 | 强 baseline，不叫参数 RL |
| Test-time Adaptation / Training | 用部署期数据调整参数或 activation/adaptation vector | 是或局部是 | 上位范式 |
| Test-time Reinforcement Learning | 用部署期 interaction/reward 对 policy 参数做 RL 更新 | 是 | 本项目主设置 |

本项目的 policy 是：

\[
\pi_{\theta,\phi_k}(a_t\mid h_t),
\]

其中：

- `θ` 是冻结的 backbone；
- `φ_k` 是第 `k` 个 session/domain 的 LoRA adapter；
- 每条 rollout 必须绑定唯一的 `(base_sha256, adapter_sha256, policy_version)`；
- update 只能发生在 episode 边界；一条 episode 内 policy 不允许变化。

### 2.2 “测试时”到底指什么

必须在 protocol 中固定 adaptation scope：

```yaml
adaptation_scope: domain_session
reset_unit: domain_seed
update_boundary: between_episodes
cross_tenant_transfer: forbidden
base_model_frozen: true
adapter_kind: lora
```

默认语义：同一未知 domain/session 中连续到达一串相关任务，Agent 可以从前面任务的可用证据中学习，随后在**尚未用于梯度**的未来任务上接受评测。

不允许把不同 test split、不同用户或不同 seed 的 adapter 状态串起来。每个 sequence 开始时从相同的 `φ_0` 初始化。

### 2.3 部分可验证，而非“无标签”口号

对第 `i` 条轨迹 `τ_i`，部署时可用信号分为：

```text
E_hard: schema、API return、权限、policy rule、state invariant、transaction receipt
E_soft: 在 dev 上训练并校准的过程/结果 verifier
R_hidden: benchmark 官方 hidden evaluator，只允许最终研究评测
```

核心隔离原则：

- `E_hard/E_soft` 可以进入 adaptation objective；
- `R_hidden` 不得进入 rollout selection、branch selection、gradient、commit Gate 或超参数选择；
- 若某环境把最终 state test 完整公开给 agent，则该设置只能叫 `verifiable online RL`，不能叫 label-free；
- 任何结果必须同时报告“可用于学习的 evidence”与“仅用于评测的 hidden outcome”。

### 2.4 三类性能必须分开

1. **Within-task recovery**：同一任务失败后 retry 是否成功；
2. **Transductive adaptation**：在已经参与更新的任务集合上是否改善；
3. **Inductive future transfer**：前序任务更新后，对后续从未参与更新的任务是否改善。

论文主指标必须是第 3 类和 prequential 指标。第 1、2 类只能作为补充，否则 reviewer 会认为只是“测试集训练后测试集记忆”。

---

## 3. 截至 2026-08-22 的相关工作地图

### 3.1 必须正面对比的工作

| 工作 | 已覆盖内容 | 本项目不能重复声称 | 仍可能存在的窄缺口 |
|---|---|---|---|
| [TTRL](https://arxiv.org/abs/2504.16084) | 无标签 reasoning 数据、majority-vote pseudo reward、参数更新 | “第一次做 test-time RL” | 状态化多步 Agent、partial evidence、safe commit |
| [T3RL](https://arxiv.org/abs/2603.02203) | 工具验证加权 pseudo-label，用于数学 reasoning | “第一次用工具验证稳定 TTRL” | 工具动作本身改变环境、branch credit、future-task transfer |
| [GTTA](https://arxiv.org/abs/2511.04847) | Agent 环境交互；syntactic adaptation vector；dynamics grounding | “第一次让 Agent 在未知环境部署期适应” | RL 参数更新的证据归因、风险受控 commit |
| [ACE](https://arxiv.org/abs/2510.04618) | 在线演化 context/playbook，利用自然 execution feedback | “第一次让 Agent 从执行反馈持续学习” | 参数更新、policy identity、counterfactual evidence |
| [OLIVIA](https://arxiv.org/abs/2605.11169) | ReAct action layer 的 contextual bandit 与 UCB 在线更新 | “第一次做轻量 action-level online adaptation” | LLM policy adapter、多步 signed credit、commit Gate |
| [MemoPilot](https://arxiv.org/abs/2606.08656) | 冻结 player；RL 训练 memory updater；multi-turn GRPO | “第一次用 RL 学 Agent memory 更新” | deployment-time policy gradient、partial verifier 与 rollback |
| [SEAL](https://arxiv.org/abs/2506.10943) | 模型生成 self-edits 和 finetuning data，由下游性能训练 adaptation policy | “第一次做 self-directed parameter adaptation” | stateful tool evidence 与安全在线提交 |
| [Monitoring Risks in TTA](https://openreview.net/forum?id=TzHX2RWUdE) | confidence sequence 风险监控；模型持续变化 | “第一次监控 test-time adaptation 风险” | 将 risk monitor 变成 Agent adapter commit protocol |
| [Amplification Effects in TTRL](https://arxiv.org/abs/2603.15417) | 自一致性 TTRL 可能放大有害行为并产生 reasoning tax | 不能假设 test-time update 天然安全 | Agent 工具流中的可回滚更新与污染隔离 |
| [CausalFlow](https://arxiv.org/abs/2605.25338) | 对失败 Agent trace 做 step-level counterfactual intervention，估计 causal responsibility、执行 minimal repair，并将修复转为可复用 supervision | “第一次用反事实干预定位或修复 Agent 失败步骤” | partial-evidence credit 是否能在部署流中直接驱动 online LoRA policy optimization，以及 candidate commit/rollback 的 future-task 风险控制 |
| [Counterfactual Shapley Credit Assignment](https://arxiv.org/abs/2607.16999) | 一般 RL 中 counterfactual Shapley credit | “第一次做反事实 credit” | 部分可验证的 LLM Agent test-time RL 与成本受控 branch |

另需持续跟踪：MiGrATe、MATTRL、continual/test-time agent learning、speculative rollback、agent verifier、safe PEFT 与 2026 年 8 月之后的新工作。

### 3.2 当前可 defensible 的差异

候选差异不是“反事实定位/修复”本身，而是一个严格限定的问题组合：

> **Stateful + replayable tool agents；deployment-time LoRA RL；只有 partial observable evidence；paired counterfactual branches 用于 signed action credit；candidate update 经过 sequential SafeCommit 后才改变后续任务 policy。**

与 CausalFlow 的必要边界是：CausalFlow-style repair 可以改当前轨迹或形成离线 supervision；本项目研究的是在**没有 hidden outcome、只有部分可靠 evidence**时，反事实差分能否作为 action-token online policy-update signal，并以严格 prequential protocol 衡量其对**后续未参与更新任务**的迁移。若 matched-cost 的 CausalFlow-style repair/reuse control 与本方法持平，C1 的算法创新不成立。

即使组合在 2026-08-22 看起来尚未被上述代表性工作完整覆盖，也不能直接声称新颖。组合创新要过两个检验：

1. 每个组件必须解决同一个核心 bottleneck，而不是系统拼盘；
2. 删除任一核心组件都应在预注册机制指标上失败。

### 3.3 为什么“简单加 verifier”不够

T3RL 已证明 tool verification 可以改善 reasoning TTRL 的 pseudo label。若本项目只把 Python verifier 换成 Agent environment checker，reviewer 会认为是直接迁移。

本项目必须展示多步 Agent 特有问题：

- 同一个 terminal outcome 可能由完全不同 action path 达成；
- 合法 tool call 不等于有助于目标；
- 早期不可逆 action 会造成 collateral damage；
- sparse terminal reward 无法区分关键动作与无关动作；
- test-time 参数更新会影响后续任务，而不是只影响当前答案；
- verifier 只覆盖部分约束，错误共识仍可能被强化。

### 3.4 为什么“加 rollback”也不够

rollback 作为工程模式并不新。论文贡献必须是：

- 定义什么 evidence 足以允许一次 policy commit；
- 在 adaptive repeated testing 下如何控制 false commit；
- 证明 Gate 不是简单牺牲全部更新换安全；
- 明确保证对象是 proxy risk 还是 hidden task success；
- 量化 rollback 对 learning efficiency、latency 和 failure severity 的影响。

---

## 4. Novelty Gate：开工前必须完成的查新

### 4.1 查询范围

在正式实现和投稿前各做一次增量查新，至少覆盖：

- arXiv：`cs.LG`、`cs.CL`、`cs.AI`、`cs.CR`；
- OpenReview：ICLR、ICML、NeurIPS、ACL ARR 及 Agent/TTT workshops；
- Semantic Scholar / OpenAlex 的引用与相似工作；
- 相关项目 GitHub 最新 README、release、issues；
- 关键词：
  - `LLM agent test-time reinforcement learning`；
  - `deployment-time policy adaptation tool agents`；
  - `counterfactual credit stateful agent`；
  - `CausalFlow counterfactual intervention agent trace repair`；
  - `partial verifier online RL`；
  - `safe test-time update rollback LoRA`；
  - `confidence sequence agent adaptation`；
  - `prequential evaluation self-improving agents`。

### 4.2 机器可读 claim matrix

建立 `novelty/claim_matrix.csv`：

```text
paper_id,online_weight_update,stateful_agent,partial_verifier,
exact_branch_credit,trace_repair_or_reuse,sequential_commit_gate,
prequential_future_transfer,hidden_holdout,matched_cost,
closest_claim,overlap_risk,notes
```

每篇 closest work 必须由两个人或两个独立 reader 复核，不能只看摘要。

### 4.3 Go / Pivot / Kill

| 结果 | 决策 |
|---|---|
| 没有工作同时覆盖 C1 与 C2，且两者共同解决一个 bottleneck | `GO` |
| C1 已被覆盖、C2 尚有空白 | 删除 counterfactual headline，改成 safe deployment-time policy improvement |
| C2 已被覆盖、C1 尚有空白 | SafeCommit 降为工程组件，主攻 evidence credit |
| 两者均被直接覆盖 | `KILL/PIVOT`，项目只保留求职工程版，不投算法主会 |
| CausalFlow-style matched control 解释全部 C1 gain | 删除“反事实 credit”贡献；只在 C2 独立成立时转为 safe online adaptation 论文 |
| 只剩 benchmark 或系统组合贡献 | 改投 systems/demo/benchmark 方向，不伪装算法创新 |

禁止使用“据我们所知首次”直到：

- 完成 full-text claim matrix；
- 相关工作检索日期距投稿不超过 14 天；
- 至少一名无上下文 reviewer 未找到直接覆盖论文。

---

## 5. 研究问题、假设与 claim map

### 5.1 Problem Anchor

部署期 Agent 面临的关键问题不是“没有更多 token”，而是：

> 可获得反馈通常不完整、不稳定且会被 policy 自身行为影响；如果直接把这些 evidence 当 reward 做在线梯度更新，错误会传播到后续任务。

因此研究目标是在固定 deployment budget 下，最大化未来任务的 prequential performance，并对 collateral damage / safety regression 施加约束。

### 5.2 主张 C1：Evidence-grounded action credit

**候选主张：**

> 在可重放、状态化工具环境中，使用 matched partial-evidence counterfactual branch 构造的 signed action credit，相比 trajectory-level、majority/self-consistency、仅 hard-validity reward 以及 CausalFlow-style trace repair/reuse，在相同 deployment cost 下提高 online LoRA policy optimization 的 prequential future-task learning efficiency。

最低可信证据：

- 两个公共 stateful agent environments；
- 两个模型家族，至少一个 3B–4B、一个 7B–8B；
- seed/stream 数由不含 method label 的 calibration variance 与 power analysis 一次冻结；
- primary metric 的 paired/hierarchical bootstrap 95% CI 在两个环境均支持改善；
- branch、verifier、generated tokens、tool calls、update FLOPs 全部计费；
- random-branch、unpaired-branch、terminal-only、hard-evidence-only 删除实验；
- CausalFlow-style minimal repair 与“同一 branch 数据转成 repaired demonstration/preference data”的等成本控制；
- 机制指标显示 credit 与真实 hidden action contribution 在 controlled environment 上更一致；
- 不把 shaped credit 称为原始 policy gradient 的无偏估计，除非另有严格证明。

### 5.3 主张 C2：Confidence-gated SafeCommit

**候选支持主张：**

> 对每个候选 adapter 使用预注册、带跨 candidate 错误预算的 gain/risk confidence Gate，再 commit 或 rollback，可显著降低 poisoned/negative-transfer update 的频率和严重度，同时保留大部分 C1 的 learning gain。

最低可信证据：

- benign、mixed、poisoned、abrupt-shift 四类 streams；
- paired candidate-vs-parent shadow evaluation；
- false commit、false rollback、catastrophic-update rate、worst-case anchor regression；
- 与 always-commit、fixed-threshold、periodic reset、oracle commit upper bound 对比；
- Gate 的 coverage 在独立 simulator 上校验；
- 明确 guarantee 只针对可观测 proxy risk 时，不外推为真实安全保证。

### 5.4 Anti-claims

必须排除：

- A1：提升只是因为更多 rollout/branch compute；
- A2：提升来自 hidden evaluator 泄漏；
- A3：提升来自更大 LoRA rank 或更多 update steps；
- A4：提升只是 same-task memorization；
- A5：提升来自更长 memory/context；
- A6：SafeCommit 只是拒绝所有更新；
- A7：反事实 branch 只是额外搜索，不是更好的训练 signal；
- A8：结果只在一个简单 deterministic 环境成立；
- A9：行为变化来自 stale rollout、token/mask 错位或错误 old log-prob；
- A10：所谓 action credit 实际传播到 prefix/observation token，优化对象不清。

### 5.5 研究路线停止条件

下列情况任一满足，就停止“顶会算法”叙事：

- matched-cost 下 C1 不优于最强 baseline；
- future-holdout 无提升，只改善 same-task retry；
- C1 仅在 controlled toy 环境成立；
- random branch 与 proposed branch 无显著差异；
- SafeCommit 通过几乎不更新获得低风险；
- hard verifier 使用了测试 hidden labels；
- public environment 无法可靠 reset/branch，方法主要假设失效；
- 冻结的 primary runs 完成后方向性不一致且效应小于预注册 SESOI；
- 相关新论文已直接覆盖核心 claim；
- Guard correctness Gate 无法通过。

### 5.6 默认 quantitative paper Gate

以下是求职/投稿规划阈值，不是已有结果。它们只能在看不到 method label 的 calibration variance 表明尺度明显不合适时，于 test 解封前整体修订并重新 hash：

| Gate | 默认阈值 | 未达处理 |
|---|---|---|
| C1 primary contrast | 两个公共环境中 `EGC×always-commit - naive×always-commit` 的 paired 95% CI 下界均 `>0`；平均绝对 `AUPC_prequential` 增益至少 `0.03` 或 relative error reduction 至少 `8%` | C1 FAIL 或缩窄 scope，不能靠 full system 掩盖 |
| Closest-work control | 相对 CausalFlow-style repair/reuse 的方向在两环境一致，且 pooled paired CI `>0` | 删除 counterfactual-credit novelty headline |
| Mechanism | CTS credit-sign accuracy 至少比 random/unpaired control 高 `10` 个百分点；rank correlation 至少高 `0.15` | 只能称 exploration/repair benefit |
| Sealed transfer | `sealed_future_holdout_score` 与 C1 同向，且 harm CI 不跨过 `-0.01` non-inferiority margin | 不声称 future transfer |
| C2 risk | 相对 always-commit 的 catastrophic rate 相对下降至少 `30%`，且预注册 CI 支持 risk ratio `<1` | C2 FAIL/工程组件 |
| C2 non-degeneracy | 保留至少 `80%` 的 C1 gain；commit rate 在 `[0.10,0.90]` | 判定为拒绝学习或无选择性 |
| Generality | 第二模型家族两个环境均同向；任何明显反向效应必须解释并限定模型范围 | 不声称跨模型普适 |

顶会主会版本至少需要 C1 通过；C2 若失败，应降为 integrity/engineering safeguard，不能继续把两个弱效应拼成“完整算法”。

---

## 6. 形式化问题

### 6.1 Continual deployment stream

环境由 latent domain `z` 和任务序列组成：

\[
\mathcal{S}^{(z)}=(x_1,x_2,\dots,x_N),
\]

每个任务对应一个 POMDP episode。第 `k` 个任务到来前使用当前 adapter `φ_{k-1}` 完成第一次尝试并记录 prequential outcome：

\[
Y_k^{\mathrm{pre}}=R_{\mathrm{hidden}}(\tau_k^{\mathrm{first}}).
\]

注意：`R_hidden` 只由离线 evaluator 计算，用于研究报告；online algorithm 看不到它。

任务完成后，algorithm 可从允许的 evidence 和额外 branch budget 构建 batch，产生候选 adapter `φ'_k`。SafeCommit 决定：

\[
\phi_k =
\begin{cases}
\phi'_k, & \text{commit}\;\\
\phi_{k-1}, & \text{rollback}.
\end{cases}
\]

### 6.2 Primary objective

归一化 prequential area，正式名称固定为 `AUPC_prequential`：

\[
\operatorname{AUPC}_{\mathrm{prequential}}=\frac{1}{N}\sum_{k=1}^{N}Y_k^{\mathrm{pre}}.
\]

每个任务必须先以当前 adapter 得到 `Y_k^{pre}`，之后该任务才允许进入更新；因此这不是“更新后在同题重测”。若 task outcome 是 `[0,1]` score，直接使用；若是 binary success，则报告 mean 与 sequence curve。另用完全不参与 update、selector、SafeCommit 或超参选择的冻结 `future_holdout` 报告 confirmatory `sealed_future_holdout_score`。它是强制确认指标，但不与 `AUPC_prequential` 混成一个含义不清的复合指标。最后 `q%` tasks 的 tail performance 仅为 secondary endpoint。

设计目标：

\[
\max \operatorname{AUPC}_{\mathrm{prequential}}
\quad\text{s.t.}\quad
\mathbf C\preceq\mathbf B,\quad
\operatorname{LCB}(G_{proxy})\ge\epsilon_{gain},\quad
\operatorname{UCB}(H_{proxy})\le\epsilon_{harm}.
\]

其中 `C` 必须包含生成、验证、branch、restore、scoring、update 和 shadow evaluation 成本。这里的置信约束只针对冻结 proxy task distribution；hidden catastrophic risk 只能作为 sealed offline endpoint 经验评估，除非额外证明 proxy 与 hidden risk 的关系。禁止把 proxy mean bound 写成真实世界安全概率保证。

### 6.3 Evidence vector

每条轨迹产生：

\[
e(\tau)=[e_{\mathrm{schema}},e_{\mathrm{tool}},e_{\mathrm{policy}},
e_{\mathrm{state}},e_{\mathrm{user}},e_{\mathrm{soft}}].
\]

每个分量都保存：

- producer；
- input artifact hashes；
- value 与 unit；
- calibration version；
- missing/timeout semantics；
- 是否允许进入 gradient；
- 是否只用于 diagnostic。

组合 evidence score 在 calibration split 上归一化到 `[0,1]`：

\[
g(e)=w^\top e-\lambda_c c(\tau),
\]

其中 `w`、`λ_c`、归一化边界和 out-of-range clipping 规则只能在 dev/calibration 冻结；test stream 不能用 hidden outcome 重选。主实验固定 `λ_c=0`，以免在 credit 内惩罚成本、又在外层预算约束中重复控制；cost-aware reward 只能作为 appendix variant。

### 6.4 Partial verification calibration

`E_soft` 必须在 development tasks 上校准，报告：

- AUROC/AUPRC；
- expected calibration error；
- Brier score；
- selective risk-coverage curve；
- domain-shift calibration drift。

若 verifier 在新 domain 严重失准，algorithm 必须降低 soft evidence 权重或停止 parameter update，不能自动把高置信错误写回 policy。

---

## 7. 候选算法 EGC-TTRL

### 7.1 总体流程

```text
Parent Adapter φ_k
      │
      ├─ first-attempt rollout ──→ hidden evaluator（offline only）
      │                  │
      │                  └─→ accessible evidence E
      │
      ├─ critical-decision selector
      │                  │
      │                  └─→ exact state restore + matched branches
      │                                      │
      │                                      └─→ signed evidence credit
      │
      ├─ action-token LoRA-GRPO update
      │                  │
      │                  └─→ Candidate Adapter φ'_k
      │
      └─ paired shadow/sentinel evaluation
                         │
                    SafeCommit Gate
                    ├─ COMMIT φ'_k
                    └─ ROLLBACK φ_k
```

### 7.1.1 单一方法原理：two-scale evidence gating

EGC-TTRL 不是“credit 模块 + rollback 模块”的拼盘。它只依赖一个可检验原则：

> **不确定的部署期 evidence 不应无条件改变未来 policy；局部 Gate 决定哪些 action tokens 能产生梯度，全局 Gate 决定这些梯度形成的 candidate adapter 能否影响后续任务。**

局部 Gate 使用 matched branch 的 signed effect 区间；全局 Gate 使用未参与梯度的 shadow/sentinel paired difference 区间。两者分别控制 credit contamination 与 adapter contamination。删除局部 Gate 应增加错误梯度或 credit-sign error；删除全局 Gate 应增加 false commit/catastrophic update。若这两个机制预测在 2×2 消融中不成立，就不能把完整系统作为统一算法贡献。

### 7.2 Estimand、selector 与 support

主 estimand 明确限定在 selector 选出的 decision-state 分布，而不是所有 Agent tokens：

\[
C_{\mu}(s,a)=Q_E(s,a)-\mathbb E_{a'\sim\mu(\cdot|s)}Q_E(s,a'),
\qquad
Q_E(s,a)=\mathbb E_{\xi}[g(e(\tau(s,a,\xi)))].
\]

其中 `μ` 是冻结 parent adapter 上的 grammar-constrained action proposal，`ξ` 是 continuation/user/tool 随机性，`g(e)` 只使用 online 可获得 evidence。论文若没有额外 IPW 理论，只能声称估计 `s~D_selected` 上的 credit，不能外推成所有 turns 的平均 causal effect。

为控制 branch 成本，只在最多 `B_decision` 个 action turns 上分支。selector 输入只允许：

- action distribution entropy；
- tool-side effect class；
- verifier disagreement；
- observed state change magnitude；
- parser/permission risk；
- 当前 episode 可见 history。

不能使用 hidden evaluator 或未来 observation。

selector 规则在 dev 冻结。主实验至少比较：

- uncertainty selector；
- state-impact selector；
- uniform random selector；
- all-turn oracle diagnostic（只在 controlled env）；
- no branch。

每个 turn 保存 inclusion probability `q_t` 和 selector version。v0.1 **不做隐含 inverse-propensity correction**：`q_t` 只用于 support audit、coverage 分层和 selector 对照。若后续引入 Horvitz–Thompson/IPW，必须单独冻结 estimand、权重截断和方差估计，并作为新 variant 报告。

### 7.3 Matched branch protocol

对选中的 decision state `s_t`：

1. 保存可验证 snapshot `S_t`；
2. 从同一个 `μ` 采样 `G=4` 个**去重、语法合法**的 action；每个 action 必须有 `μ(a_i|s_t)>0`，最多重采样 8 次；少于 2 个有效不同 action 时标记 `NO_SUPPORT`，不更新；
3. production first-attempt action 只有在确由同一 `μ` 生成时才可进入训练 group；否则仅作 diagnostic，并重新采完整 group；
4. 对每个 action 使用同一组 `R=4` continuation seeds；同一 `r` 下各 action 使用 common random numbers；
5. 所有 branch 使用同一 snapshot、parent policy version、continuation horizon 和 stopping rule；
6. restore、prefix、continuation、verifier 和失败重试成本全部进入 ledger；
7. branch trajectory 不能写入正常 production history；
8. approximate restore、zero-support action 或 continuation protocol mismatch 一律不进入主分析。

`G=4、R=4、B_decision=1` 是 v0.1 public primary 默认；ControlledToolShift 可做更宽 group。任何调整必须在 calibration 结束、test stream 解封前写入 protocol hash。

### 7.4 Algorithm 1：从 branch 到训练 row

对第 `i` 个 action 和 continuation seed `r`，得到归一化 evidence utility：

\[
U_{i,r}=g(e(\tau(s_t,a_i,\xi_r)))\in[0,1],
\qquad
\bar U_i=\frac{1}{R}\sum_{r=1}^{R}U_{i,r}.
\]

用 proposal group 作为反事实 baseline：

\[
\hat c_i=\bar U_i-\frac{1}{G}\sum_{j=1}^{G}\bar U_j.
\]

对每个 continuation seed 构造 paired unit

\[
z_{i,r}=U_{i,r}-\frac{1}{G}\sum_{j=1}^{G}U_{j,r}\in[-1,1].
\]

v0.1 的局部 reliability 只作训练启发式，不宣称 finite-sample coverage。令

\[
\hat v_i=\frac{1}{R-1}\sum_r(z_{i,r}-\hat c_i)^2,\qquad
b_i=t_{R-1,0.90}\sqrt{\hat v_i/R},
\]

并取 `L_i=max(-1,hat_c_i-b_i)`、`U_i=min(1,hat_c_i+b_i)`；`t_{R-1,0.90}` 是 Student-t 的固定 90th percentile（单侧 10%）。reliability Gate 为：

\[
\alpha_i=\mathbb 1\{L_i>\eta_{credit}\ \lor\ U_i<-\eta_{credit}\},
\qquad
A_i=\operatorname{clip}\left(\alpha_i\hat c_i/s_{group},-A_{max},A_{max}\right),
\]

其中 `s_group=max(std(\bar U_{1:G},ddof=1),10^{-3})`，`A_max=5`，`η_credit=0.02`。区间与 `[-η_credit,+η_credit]` 相交时 `A_i=0`。`R<2`、非有限值或有效 paired seeds 不足时整组 `NO_RELIABLE_CREDIT`。主方法不再做第二次 group centering，避免把被 Gate 拒绝的 action 重新赋予非零 advantage。正式 statistical coverage claim 只属于 §7.8 的全局 Gate，不属于这个局部 t heuristic；局部可靠性是否有用由 exact-oracle calibration 与消融决定。

每个候选 action 产生一条 `UpdateRow`：

```yaml
state_prefix_tokens_ref: producer_artifact
action_tokens_ref: producer_artifact
action_loss_mask: only_candidate_action_tokens
behavior_logprobs_ref: same_parent_generation
advantage: A_i
decision_state_ref: s_t
branch_group_ref: group_t
evidence_and_cost_refs: [...]
```

只对候选 action `a_i` 自身的 generated action-token span 反传；continuation、observation、tool output、system/user/prefix token 全部 mask 为 0。某 branch 后续 action 若也被 selector 选中，必须形成另一个独立 group，不能沿用当前 credit。

Algorithm 1 的可执行顺序：

```text
select state -> seal snapshot -> sample G actions from parent μ
-> score/store old log-probs -> run G×R coupled continuations
-> build U matrix -> compute paired credits and reliability Gate
-> materialize G immutable UpdateRows -> Guard ALLOW
-> fixed-step LoRA update -> immutable candidate adapter
```

**重要限制：**`A_i` 是 selected-decision、partial-evidence objective 的 shaped advantage，不自动是 hidden return policy gradient 的无偏估计。论文只能声称经验 learning efficiency；若想声称 unbiased/consistent，必须另给定理、support 假设与 exact oracle 证明。

### 7.5 三动作手算 fixture

冻结 toy group 的 paired means 为 `U=[0.9,0.6,0.0]`，group mean 为 `0.5`，所以 `c=[+0.4,+0.1,-0.5]`。假设三个预注册区间依次为 `[0.30,0.50]`、`[-0.05,0.25]`、`[-0.65,-0.35]`，且 `η_credit=0.05`，则 `α=[1,0,1]`：第一条 action-token log-prob 被增加，第二条无梯度，第三条被降低。fixture 必须逐 token 检查：只有三条候选 action span 可出现非零 loss；改动任一 observation token 都应使测试失败。

trajectory-level accessible outcome 的混合仅作消融：

\[
A'_i=A_i+\beta(r_i^{proxy}-\bar r^{proxy}).
\]

主方法固定 `β=0`；terminal-only 与预注册 fixed-`β` 只用于解释机制，禁止 test 后择优。

### 7.6 Action-token objective

每个 action turn 映射到明确的 generated-token span。observation、tool return、system prompt、user message 和 prefix token 的 loss mask 必须为 0。

对 action tokens 使用 clipped policy objective，并绑定行为策略 log-prob：

\[
r_{i,u}(\phi)=
\exp\left(\log\pi_{\theta,\phi}(y_{i,u}|h_{i,u})-
\log\pi_{\theta,\phi_{\mathrm{beh}}}(y_{i,u}|h_{i,u})\right).
\]

v0.1 每条 row 先在 action tokens 内取均值，再在 rows 间取均值，避免长 action 自动获得更大权重：

\[
\mathcal L_i=-\frac{1}{M_i}\sum_u m_{i,u}
\min\left(r_{i,u}A_i,\operatorname{clip}(r_{i,u},1-\epsilon,1+\epsilon)A_i\right)
+\lambda_{KL}\frac{1}{M_i}\sum_um_{i,u}k_{i,u},
\]

\[
k_{i,u}=\exp(\ell^{parent}_{i,u}-\ell^{current}_{i,u})
-(\ell^{parent}_{i,u}-\ell^{current}_{i,u})-1,
\qquad
\mathcal L=\frac{1}{|\mathcal B|}\sum_i\mathcal L_i.
\]

其中 `M_i=sum_u m_{i,u}`；`ε=0.2`、`λ_KL=0.02`，不加 entropy bonus，做 4 个 optimizer steps。behavior 与 KL reference 都是产生该 group 的 sealed parent adapter。任何 row 的 `M_i=0`、old-logprob 缺失或 parent identity 不同，整 batch 拒绝。objective 只在 `m_{i,u}^{action}=1` 时累积，并必须满足：

- behavior log-prob 来自 rollout policy 或完全相同 checkpoint 的独立 scorer；
- current/new log-prob 不得冒充 old log-prob；
- tokenizer、chat template、truncation、padding side 和 token IDs 全部绑定；
- action span 由 structured recorder 产生，不用 `str.find`；
- update API 不允许重新 tokenize 文本作为 fallback；
- 每个 update 记录 source artifact hashes。

### 7.7 Session-scoped LoRA

v0.1 primary profile 不再在实现时二选一：

```yaml
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_revision: RESOLVE_FULL_SHA_IN_M0_AND_NEVER_TRACK_MAIN
trainable: lora_only
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
optimizer: adamw
learning_rate: 5.0e-6
adam_betas: [0.9, 0.95]
weight_decay: 0.0
max_grad_norm: 1.0
update_steps_per_batch: 4
clip_epsilon: 0.2
kl_to_parent_coefficient: 0.02
precision: bf16
max_context_tokens: 8192
max_action_tokens: 512
episode_mid_update: forbidden
```

所有 RL variants 共用该 profile。dev-only grid 只允许 `rank∈{8,16}`、`lr∈{2e-6,5e-6}`、`KL∈{0.01,0.02}`，选择规则为先通过 stability Gate，再取 dev `AUPC_prequential` 最高者；选择完成后写入 immutable protocol。论文必须做 rank/parameter-count matched control，不能让 proposed method 用更大 rank 或更多 update steps。

### 7.8 SafeCommit

每次 update 先产生不可服务的 candidate adapter `φ'`：

```text
TRAINED_CANDIDATE
      ↓
SHADOW_EVALUATING
      ↓
┌───────────────┬────────────────┐
│ COMMITTABLE   │ ROLLBACK       │
└───────────────┴────────────────┘
```

shadow set 由三部分组成：

- 当前 domain 的未用于梯度 sentinel tasks；
- anchor capability tasks；
- policy/safety/collateral-damage probes。

v0.1 将“跨 candidate 的顺序决策”和“candidate 内可选停止”分开：主协议对每个 candidate 使用固定 shadow 样本量，不做 candidate 内 early stopping。candidate `k` 必须先 sealed，随后 Gate service 才从未暴露 reservoir 抽取新任务。定义：

- gain task 上 `D_i^{gain}=score(candidate)-score(parent)∈[-1,1]`；
- anchor task 上 `D_i^{harm}=score(parent)-score(candidate)∈[-1,1]`，越大表示伤害；
- 两组 task 不进入 gradient；主协议中每个 item 最多使用一次；
- 每个 candidate 固定 `n_gain=n_anchor=64`，不足则 `INCONCLUSIVE -> ROLLBACK`；
- stream 最多 `K_max` 次 candidate；总错误预算 `α_total=0.05`，第 `k` 次使用 `α_k=6α_total/(π²k²)`，gain/harm 各分 `α_k/2`；
- 对范围长度为 2 的 paired difference，单侧 Hoeffding radius 为 `b_k=sqrt(2 log(1/α_side)/n)`；
- `LCB_gain=mean(D_gain)-b_k`，`UCB_harm=mean(D_harm)+b_k`；
- candidate、task draw、score artifact 与决定全部带 hash，禁止根据中途分数换样本或加样本。

该合同只在“candidate 在 sentinel draw 前固定、每个 shadow task 条件独立且来自冻结 proxy task distribution”假设下控制 proxy-mean decision error。更高功效的 empirical-Bernstein/mixture confidence sequence 只能在 coverage simulator 通过后作为下一冻结版本，不能 test 后替换。

commit 条件：

```text
LCB(proxy_gain) >= epsilon_gain
AND UCB(anchor_harm) <= epsilon_harm
AND GuardDecision == ALLOW
AND no integrity/split violation
```

否则 rollback。若 verifier 缺失、timeout、reservoir 不足或 calibration drift 超限，默认 fail closed。若上述 Hoeffding Gate 导致 commit rate 接近 0，则 C2 失败；不能事后放松阈值救结果。

### 7.9 避免 sentinel overfitting

反复用固定 sentinel 决策会产生 adaptive overfitting。主协议必须：

- sentinel reservoir 与 gradient tasks 分离；
- 每个 item 最大复用次数为 1；
- 使用 candidate sealed 之后才抽取的 fresh batches；
- 记录每次 exposure；
- 用上述 summable `α_k` 处理跨 candidate 重复决策；
- final future holdout 永不参与 commit；
- 若公共环境数据量不支持 fresh reservoir，则公开环境上的 C2 降级为 empirical risk/gain study，正式 coverage claim 只来自 ControlledToolShift simulator；不得悄悄复用 sentinel 仍宣称 time-uniform guarantee。

### 7.10 Atomic commit / rollback

adapter commit 使用单写者、fencing token 和原子 manifest：

```text
parent adapter + optimizer state
        ↓
candidate temp directory
        ↓
hash + Guard validation + shadow Gate
        ↓
atomic rename to committed version
        ↓
rollout runtime sync + canary
```

任何 UNKNOWN 状态停止新 generation。晚到 callback 不能覆盖更新版本。candidate checkpoint 即使写完但无 authoritative commit event，也必须 quarantine。

---

## 8. 数据与 split：防止“测试集训练”争议

### 8.1 六层数据角色

| Split | 可做什么 | 禁止做什么 |
|---|---|---|
| `dev` | 开发、debug、选择方法结构 | 报最终泛化结论 |
| `calibration` | verifier calibration、阈值、SESOI、sample-size planning | 进入 final test |
| `adaptation_stream` | test-time rollout 与允许的 update | 使用 hidden test labels |
| `sentinel_stream` | commit/rollback | 进入 gradient；无限复用 |
| `candidate_audit` | 离线标注 harmful candidate、false commit/rollback | 向 online/Gate 返回分数；影响 commit |
| `future_holdout` | 最终 inductive evaluation | 任何 update、branch selection、threshold selection |

sentinel 是**在线模型选择数据**，绝不能同时充当无偏风险评测集。

### 8.2 Capability-scoped manifests

禁止把全部 refs 放在一个训练进程可读 manifest。必须拆成：

1. `online_manifest`：只含 adaptation opaque handles、可见 prompt/tool schema 与 accessible evidence capability；不能列举 sentinel/audit/holdout IDs；
2. `gate_manifest`：只有独立 Gate service 可读的 sentinel/anchor opaque handles；service 只返回 `COMMIT|ROLLBACK|INCONCLUSIVE`、区间和不可变审计事件，不返回逐题 prompt、label 或 score；
3. `sealed_audit_manifest`：由独立 OS user/container/credential 持有 `candidate_audit` 与 `future_holdout`；训练/Gate 进程无 import、文件、网络或密钥权限。

`candidate_audit` 可在每次 decision 后异步评分，但输出进入 write-only sealed store，直到整个 protocol/run manifest 锁定才解封。`future_holdout` 只在 adaptation stream 完成且最终 adapter hash 锁定后运行一次。任何提前解封、可枚举 holdout handle、训练进程成功 import hidden evaluator 或根据 audit score 重跑的情况，都令相应 claim `INVALID`。

### 8.3 不只按 task ID 去重

需要检查：

- instruction template；
- goal graph；
- tool schema；
- entity/database state；
- solution program；
- simulator seed；
- user persona；
- benchmark source provenance。

每个 capability 域单独构建 manifest，公共 release 只公开 salted hashes 与 role counts；保存 ID、内容 hash、template family、domain、seed、allowed uses。任何禁止交集使相关 claim 为 `INVALID`，不是简单警告。

### 8.4 Prequential protocol

对每个 stream task：

1. 用当前 committed adapter 完成 first attempt；
2. first attempt 的 hidden score进入 prequential metric，但 algorithm 不可见；
3. task 完成后才允许读取 accessible evidence；
4. 允许 retry/branch/update；
5. SafeCommit；
6. 进入下一个从未参与更新的 task。

这能回答“是否从过去经验迁移到未来任务”，而不是“同一道题多练几次能否记住”。

### 8.5 Transductive 与 inductive 报告分栏

论文主表必须分开：

```text
same-task recovery
adaptation-stream post-update
adaptation-stream AUPC_prequential
sealed-future-holdout score
anchor retention
```

不得把四者平均成一个含义不明的总分。

---

## 9. 环境选择

### 9.1 ControlledToolShift：机制环境，必须先做

自建小型、完全 deterministic、可 snapshot 的工具环境，用于确认算法机制。

应包含：

- 8–12 个 tool schemas；
- syntax shift：参数名、枚举、返回 schema 改变；
- dynamics shift：相同行为的状态转移规则改变；
- delayed side effect；
- reversible 与 irreversible actions；
- deceptive but frequent pseudo reward；
- partial verifier：只观察部分 state invariants；
- exact hidden oracle：仅离线评测；
- exact branch contribution enumeration；
- poison evidence stream；
- no-op 与 all-success/all-fail 退化检测。

ControlledToolShift 只支撑机制与单元验证，不能独自支撑“真实 Agent 广泛有效”。

#### 9.1.1 v0 状态与工具合同

最小 world state 固定为：

```text
WorldState = {
  schema_version,
  inventory[sku],
  reservation[order_id],
  order_status[order_id],   # NONE|CREATED|PAID|SHIPPED|CANCELLED
  balance[user_id],
  address[user_id],
  delayed_effect_queue,
  permission_scope,
  audit_flags
}
```

v0.1 使用 10 个工具；所有转移为纯函数 `T(state, normalized_call, exogenous_seed)` 并保存 before/after hash：

| Tool | 前置条件 | 状态转移 | Accessible evidence | 典型 hidden collateral |
|---|---|---|---|---|
| `lookup_item` | sku/item_id 存在 | 无 | schema-valid + item summary | 无 |
| `reserve_item` | stock>0 | stock-1，建立 reservation | receipt + remaining stock bucket | 锁错商品 |
| `release_reservation` | 未 ship | stock+1，reservation cleared | receipt | 释放用户仍需要的库存 |
| `create_order` | valid reservation | status=CREATED | order_id | 重复订单 |
| `charge` | CREATED + permission | balance 减少；可延迟入账 | immediate/pending receipt | 重复扣款/错误金额 |
| `refund` | PAID/CANCELLED | balance 恢复 | receipt | 超额退款 |
| `ship` | PAID + address | status=SHIPPED，不可逆 | dispatch receipt | 错地址/错商品不可逆 |
| `cancel_order` | CREATED/PAID 且未 ship | status=CANCELLED | status receipt | 未退款或误取消 |
| `get_receipt` | order 存在 | 无 | partial state projection | projection 可过时/被 poison |
| `complete_task` | 任意 | episode terminal | acknowledgement | goal 未完成或有 collateral |

shift families 固定为：`syntax_v1`（`sku→item_id`、枚举重命名）、`dynamics_v1`（cancel/refund 顺序变化）、`delay_v1`（charge 在 2 个 turns 后生效）、`permission_v1`（高风险 action 需新 scope）、`poison_v1`（receipt 以冻结概率错误但 DB state 正确）。hidden oracle 比较完整 goal state、禁止 side effects、金额守恒与 permission policy；online evidence 只见 receipt、允许的投影与 schema，不得调用 oracle。

#### 9.1.2 冻结 fixtures 与 expected sign

实现前至少生成下列 12 个 machine-readable fixtures；每个 fixture 固定 initial state、goal、candidate calls、continuation seeds、`U` matrix、expected credit sign、hidden outcome 和 failure reason：

| ID | Case | Expected property |
|---|---|---|
| CTS-F01 | 正确 reserve vs 错 sku | correct `+`，wrong `-` |
| CTS-F02 | 正确地址 ship vs 错地址 ship | wrong irreversible action 强负 credit |
| CTS-F03 | schema-valid no-op vs effective call | validity reward 不足，state evidence 区分 |
| CTS-F04 | repeated reserve | duplicate side effect 被惩罚 |
| CTS-F05 | delayed charge | immediate receipt 不能冒充最终成功 |
| CTS-F06 | partial verifier 漏掉 collateral | proxy/hidden disagreement 被记录 |
| CTS-F07 | syntax_v1 参数迁移 | old syntax fail、new syntax pass |
| CTS-F08 | poisoned success receipt | frequent pseudo reward 与 DB invariant 冲突 |
| CTS-F09 | cancel-after-ship | restore 仅在 sandbox，有不可逆标签 |
| CTS-F10 | all-success proposal group | `DEGENERATE_GROUP`，不更新 |
| CTS-F11 | all-fail proposal group | `DEGENERATE_GROUP`，不更新 |
| CTS-F12 | branch writes parent DB | state-hash mismatch，整组 `INVALID` |

golden acceptance：F01–F09 的 credit sign 全部正确；F10–F12 必须以固定 reason code fail closed；三动作 §7.5 fixture 的 token-level gradient sign 完全一致。

### 9.2 AppWorld：第一公共主环境

[AppWorld](https://arxiv.org/abs/2407.18901) 提供本地 app/API world、复杂交互任务和 state-based unit tests，适合检查不同完成路径及 collateral damage。

使用原则：

- 官方 hidden/held-out evaluator 只做最终评测；
- accessible evidence 只使用 agent 当时能看到的 API receipt、schema、policy 与公开 state invariants；
- branch 通过隔离 world snapshot 实现；
- train/dev/test_normal/test_challenge 按官方限制使用；
- 禁止为了适配访问 encrypted evaluator 或 ground-truth solution；
- 记录 AppWorld version、data hash、task IDs 和 engine state hash。

冻结角色映射：官方 `train` 只做 dev，官方 `dev` 只做 calibration；官方 `test_normal` 按 template-family grouped hash 在 `adaptation:sentinel:candidate_audit=70:15:15` 间划分；官方 `test_challenge` 只作 sealed future holdout。若 pinned release 的 task 数不足以满足 power/fresh-sentinel 要求，必须在 test 解封前缩小 C2 claim，不能挪用 challenge tasks 做 Gate。

### 9.3 tau2：第二公共主环境

使用当前 [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)，不要基于已经提示 task outdated 的旧 `tau-bench`。当前仓库提供标准化 splits，并有 Gymnasium RL interface。

建议顺序：

1. `mock` domain 做 deterministic smoke；
2. `retail` 或 `airline` 做主实验；
3. user simulator 固定 model/version/prompt/temperature；
4. 优先使用 recorded/replayable user turns 做 paired branch；
5. 若在线 user LLM 无法做可靠 common-random-number coupling，则显式报告这一限制并增加 paired replications。

冻结角色映射：对 pinned `split_tasks.json` 中选定的官方 split，在每个 domain 内先按 template/goal family 分组，再以公开 salt 做 grouped hash partition：`dev/calibration/adaptation/sentinel/candidate_audit/future=20/15/30/10/10/15`。具体 task IDs、counts 与 content hashes 在 M0 写入 split registry；任一 role 少于 power contract 所需数量时，该 domain 不进入主 claim。tau2 当前任务与 split 由仓库版本决定，不能沿用旧 tau-bench 的陈旧 task 清单。

### 9.4 Optional，不阻塞主文

- BFCL：验证 syntax adaptation，但 stateful/long-horizon 较弱；
- WebArena：外部有效性强，但基础设施、网页漂移和成本高；
- StreamBench/EvaLearn：学习曲线评测补充；
- 安全 Agent benchmark：只用于 C2，不要扩成另一篇论文。

### 9.5 环境 Gate

每个环境正式运行前必须满足：

- reset 后 state hash 可复现；
- branch 不污染主 episode；
- task evaluator 与 online evidence API 物理/逻辑隔离；
- reward variance 与 advantage non-degeneracy 通过；
- 同任务至少存在成功与失败轨迹；
- action alternatives 不是 80% 以上 no-op；
- official split 版本固定；
- 一次 full audit 可重放。

---

## 10. Baseline 设计：三类强基线即可

### 10.1 Family A：Inference / context adaptation

- Frozen ReAct；
- Best-of-N / verifier selection，matched generated tokens；
- Reflexion/experience replay；
- ACE-style evolving context。

目的：证明梯度更新相对纯 test-time inference/context learning 是否必要。

### 10.2 Family B：Agent test-time adaptation

- GTTA Syntactic Alignment；
- GTTA Dynamics Grounding；
- OLIVIA-style contextual bandit；
- MemoPilot-style memory updater；若官方实现不能接入同一环境/模型，必须发布不可比性报告，不能静默省略；
- CausalFlow-style minimal trace repair，以及将同一 branch 数据转成 repaired demonstration/preference supervision 的 reuse control；
- session LoRA SFT on self-generated positive traces（若信号允许）。

目的：证明完整 LoRA RL 与 evidence credit 相比更轻适配策略是否值得。

### 10.3 Family C：Test-time RL

- naive terminal/proxy-reward LoRA-GRPO；
- self-consistency/majority-inspired TTRL；
- hard-verification weighted TTRL/T3RL-style；
- EGC-TTRL。

目的：隔离“RL 本身”与“proposed evidence/commit mechanism”。

### 10.4 Upper bounds 与 controls

不作为可部署 baseline：

- hidden-oracle reward update：仅在 dev/controlled env 给上限；
- all-turn exact branch：机制 upper bound；
- oracle commit：展示 Gate 理论余量；
- random labels / shuffled evidence：negative control。

### 10.5 公平预算

主表不再允许“至少一种口径有利”就过关。每个方法必须同时满足同一三维 hard cap：

1. `B_env`：所有 production、branch、restore 后 continuation、shadow/sentinel 的 charged environment transitions/tool calls；
2. `B_model`：所有 generation 与 scoring 的非 padding tokens；
3. `B_update`：进入 forward/backward 的 action tokens × optimizer steps。

每次操作只能由一个 canonical ledger event 计费；restore 本身另记 CPU seconds/bytes，但 restore 后执行的 transition 仍计入 `B_env`。verifier-model tokens 计入 `B_model`；规则 verifier 计 CPU time；shadow generation 同样计费。达到任一 cap 后，所有方法按预注册 resource-allocation rule 停止，未花完的某一资源不能兑换另一资源。wall-clock、GPU-hour、峰值显存与估算费用作为系统敏感性指标，不参与事后挑选 primary match。

若三维严格匹配导致某 baseline 无法运行，使用“同 cap 内最优配置”而不是给它额外预算；完整 raw ledger 和 success-cost Pareto surface 必须发布。

---

## 11. 主实验块：每个实验必须改变 reviewer belief

### Block 0：Sanity 与链路正确性

- Claim：结果不是 pipeline bug。
- 环境：ControlledToolShift tiny split。
- 系统：frozen、naive TTRL、EGC。
- 指标：exact oracle agreement、token identity、branch isolation、replay determinism、non-zero advantage。
- 成功条件：所有 correctness tests 通过；故障注入被 fail closed。
- 失败解释：停止 GPU 扩展，不讨论算法效果。
- 位置：Appendix + artifact report；属于 MUST-RUN。

### Block 1：主效果——未来任务学习效率

- Claim：C1。
- 环境：AppWorld + tau2；ControlledToolShift 仅机制辅助。
- 系统：每类最强 baseline + 预注册的 2×2 `reward∈{naive,EGC} × gate∈{always-commit,SafeCommit}`。
- 指标：`AUPC_prequential`、`sealed_future_holdout_score`、tail performance、三维 hard-cap ledger。
- Setup：两个 backbone families；seed 数在 blinded power analysis 后一次冻结；每个环境/模型/seed 使用 paired stream order。
- 成功条件：C1 用 `EGC×always-commit` 对 `naive×always-commit`；C2 用完整 stream-level factorial contrast；两个公共环境的 paired CI 方向一致，sealed holdout 同向，所有方法满足相同 hard caps。
- 失败解释：若只改善 within-task recovery，改为 recovery paper/portfolio，不保留 continual adaptation headline。
- 位置：Main Table 1、Figure 2；MUST-RUN。

### Block 2：创新隔离——为什么 counterfactual evidence 有效

- Claim：C1 机制。
- 对比：no branch、random branch、unpaired branch、hard-evidence only、terminal-only、equal-extra-rollout、CausalFlow-style repair、same-branch repaired-data update、full proposed。
- 指标：controlled exact credit correlation/rank、credit sign accuracy、gradient variance、sample efficiency、future performance。
- 成功条件：full proposed 在 mechanism metrics 上明确胜出，且最终 gain 不是纯额外 compute。
- 失败解释：若 random/equal-extra-rollout 或 CausalFlow-style matched control 持平，则只能声称额外 exploration/repair，不能声称新的 credit mechanism。
- 位置：Main Table 2、Figure 3；MUST-RUN。

### Block 3：SafeCommit 与污染压力测试

- Claim：C2。
- Streams：benign、soft-verifier drift、prompt/tool injection、abrupt domain shift。
- 对比：always commit、fixed threshold、periodic reset、risk-only no-learning、SafeCommit、oracle commit；并保留完整 `reward×gate` 2×2。
- 识别：纯 Gate 效应用同一批 immutable candidate archive 离线回放不同 Gate；真实部署流因 commit 后 policy 分叉，估计的是完整 gate-policy 的 stream-level effect，不能谎称“保持 candidate 完全相同”。
- 指标：catastrophic update rate、false commit、false rollback、worst anchor drop、`AUPC_prequential` retained、rollback latency。
- 成功条件：candidate-archive 与 factorial deployment stream 都降低 catastrophic update，同时保留预注册比例的 C1 gain；commit rate 不能接近 0。
- 失败解释：若只靠拒绝更新，C2 不成立。
- 位置：Main Figure 4、Table 3；MUST-RUN。

### Block 4：泛化、简单性与边界

- Claim：方法不是单环境 hack，且两个核心组件足够。
- 对比：第二模型家族、cross-domain transfer、不同 verifier coverage、不同 branch budget、overbuilt world-model variant。
- 指标：effect consistency、risk/coverage curve、Pareto frontier。
- 成功条件：至少两个模型家族方向一致；更复杂 variant 无稳定收益或成本明显更差。
- 失败解释：限定适用模型/coverage 范围；不隐藏异质性。
- 位置：Main Figure 5 + Appendix；主设置 MUST-RUN，overbuilt variant NICE-TO-HAVE。

### 明确 cut 的实验

- 十几个弱 benchmark 的 leaderboard sweep；
- full-parameter 7B/14B test-time update；
- 同时加入 DPO/PPO/GRPO/REINFORCE/ORPO 全家桶；
- 大量 LLM-as-a-Judge qualitative case；
- 无法 matched-cost 的 proprietary-model 对比；
- 为凑创新加入 multi-agent、RAG、MCTS 或 MoE。

---

## 12. 评测指标与统计协议

### 12.1 Primary endpoints

只预注册两个：

1. `AUPC_prequential`：每个 adaptation-stream task 在用于任何更新前取得的 first-attempt outcome 的归一化面积；
2. `catastrophic_update_rate`：candidate commit 后 hidden anchor 性能超过预注册下降阈值或触发 hard-policy violation 的比例。

对第 `i` 个 immutable candidate 定义：

\[
Z_i^{cat}=\mathbb 1\{\Delta H_i^{anchor}\le-\epsilon_{anchor}\ \lor\ V_i^{hard}=1\}.
\]

`ΔH_i^{anchor}` 是 candidate 相对 parent 在独立 `candidate_audit` 集上的 paired hidden score，`V_i^{hard}` 是预注册严重违规指示；`ε_anchor` 根据 dev variance/SESOI 在 test 解封前冻结。online Gate 看不到 `candidate_audit` 任务或分数。`false_commit=commit∧Z_cat`；`false_rollback=rollback∧(ΔH_future≥ε_gain)∧¬V_hard`。另报告 harm magnitude 和 worst-case drop，避免二值阈值掩盖严重度。

proxy SafeCommit 的统计保证不能外推为 `Z_cat` 的安全保证；C2 的措辞固定为“empirically reduces sealed-audit catastrophic updates under the registered streams”。

其余是 secondary/diagnostic，避免多重比较后挑显著。

### 12.2 Secondary metrics

- first-attempt task success；
- same-task retry success；
- sealed-future-holdout final success；
- pass^k / reliability；
- collateral damage；
- invalid tool calls；
- policy violation；
- generated tokens；
- tool calls/environment steps；
- branch restores；
- update tokens/FLOPs；
- rollout/update/shadow latency；
- GPU peak memory/utilization；
- adapter commit/rollback counts；
- reward/evidence variance；
- non-zero action advantage fraction；
- response/action entropy；
- KL to parent/base。

### 12.3 统计单位

不能把每个 token 或 branch 当独立样本。默认层级：

```text
seed
  └─ domain stream
       └─ task
            └─ rollout/branch
```

主 CI 使用 paired hierarchical bootstrap 或预注册 mixed-effects model。bootstrap 最外层至少按 seed/domain stream resample，再在 stream 内按 task family resample。

### 12.4 Seeds

- smoke：1 seed；
- decision pilot：3 seeds；
- primary table 的 `3 或 5 seeds` 由**不含 method label**的 calibration variance 与预注册 power target 决定，只能选择一次；
- 每个 seed/环境至少包含 4 个独立注册的 domain-stream instances；同一个 stream order 在所有方法间 paired；
- 高方差 user simulator 若 power analysis 要求超过 5 seeds，则在解封前增加，或预先缩小 claim breadth；
- seed 数、seed list、stream 数和 task 数在任何 test method result 前冻结，禁止“看到效果后 extend to 5”。

### 12.5 SESOI 与 sample size

用 dev/pilot variance 设定 smallest effect size of interest（SESOI）和 task 数。不得先看 test 再把 SESOI 降到刚好显著。

若样本量不足，应报告宽 CI 和 inconclusive，而不是用 rollout 数冒充独立样本数。

### 12.6 多重比较

- 两个 primary endpoints 预注册校正策略；
- 环境内主方法对最强 baseline 是 primary contrast；
- 其余 pairwise comparisons 标为 exploratory；
- ablation 关注 effect pattern，不堆 p-value 星号。

### 12.7 负结果保存

每个 run 将工程有效性与科学结论拆开输出：

```json
{
  "claim_id": "C1",
  "execution_status": "PLANNED|RUNNING|COMPLETED|CRASHED",
  "audit_status": "VALID|INVALID",
  "claim_outcome": "PASS|FAIL|INCONCLUSIVE|NOT_APPLICABLE",
  "protocol_sha256": "...",
  "code_commit": "...",
  "result_manifest_sha256": "...",
  "reason_codes": [],
  "allowed_claim": "...",
  "forbidden_claims": []
}
```

`audit_status=INVALID` 不能混入平均值；审计有效的 `FAIL` 必须进入预注册汇总并解释；`INCONCLUSIVE` 不等于负结果。

---

## 13. 关键消融矩阵

| ID | Variant | 检验什么 | 预期解释 |
|---|---|---|---|
| A00 | Frozen | 无 adaptation 基线 | 测静态能力 |
| A01 | Best-of-N | 纯更多 inference compute | 排除搜索收益 |
| A02 | ACE/Reflection memory | 非参数适应 | 检验 gradient 是否必要 |
| A02-CF | CausalFlow-style repair/reuse | 当前轨迹 minimal repair，或把相同 branch 数据转成 repaired supervision | 区分 repair/reuse 与 signed online action-credit RL |
| A03 | Naive LoRA-GRPO | RL 但无 EGC/SafeCommit | 基础 RL 增益 |
| A04 | Hard verifier only | 可验证性但无 branch credit | 检验 partial evidence 聚合 |
| A05 | Random branch | 同 branch 成本、错误 selection | 隔离 critical selector |
| A06 | Unpaired continuation | 无 matched coupling | 检验 variance/credit validity |
| A07 | Positive-only credit | 不惩罚 harmful action | 检验 signed negative credit |
| A08 | No reliability weight | 所有 branch 等权 | 检验 uncertainty gate |
| A09 | EGC only | 无 SafeCommit | C1 独立效果 |
| A10 | SafeCommit only | naive reward + SafeCommit | C2 是否独立 |
| A11 | Full EGC-TTRL | 两组件 | 最终方法 |
| A12 | Always rollback | 零学习安全控制 | 排除拒绝一切 |
| A13 | Oracle commit | hidden/dev oracle 上限 | SafeCommit 余量 |
| A14 | Shuffled evidence | 负控制 | 检查数据泄漏/伪机制 |
| A15 | Equal extra rollout | 把 branch 算力给普通 rollout | 排除更多采样解释 |

主文只保留 A00/A02/A02-CF/A03/A04/A09/A10/A11/A15；其余放 appendix，除非某个反例成为核心发现。

---

## 14. 故障注入与安全测试

### 14.0 Canonical pipeline fault manifest（F01–F12）

| ID | Injected fault | Expected decision / reason code |
|---|---|---|
| F01 | rollout adapter 落后 parent commit | `REJECT / POLICY_VERSION_MISMATCH` |
| F02 | current-model log-prob 冒充 behavior log-prob | `REJECT / OLD_LOGPROB_PROVENANCE` |
| F03 | trainer 从文本重新 tokenize | `REJECT / TOKEN_ARTIFACT_MISMATCH` |
| F04 | observation/prefix token mask 非零 | `REJECT / ACTION_MASK_SCOPE` |
| F05 | branch 写入 parent world | `INVALID / BRANCH_ISOLATION_BREACH` |
| F06 | online process import/call hidden evaluator | `INVALID+HALT / HIDDEN_CAPABILITY_BREACH` |
| F07 | split ID 不重叠但 template family 重叠 | `INVALID / TEMPLATE_LEAKAGE` |
| F08 | verifier timeout 被转成数值 0 | `ROLLBACK / EVIDENCE_MISSING` |
| F09 | late callback 覆盖较新 commit | `QUARANTINE+HALT / FENCING_EPOCH` |
| F10 | rollback 未恢复 optimizer/runtime | `HALT / NON_ATOMIC_ROLLBACK` |
| F11 | sentinel item 被主协议复用 | `INVALID_C2 / SENTINEL_REUSE` |
| F12 | ledger event 漏记或重复计费 | `INVALID_COST_CLAIM / LEDGER_CONSERVATION` |

`protocols/faults/F01_F12.yaml` 必须给出 injector、expected event sequence、terminal decision 和 golden hashes。B0/M1 所称“F01–F12 全过”专指本表；CTS-F01–F12 是另一个机制 fixture namespace。

### 14.1 Evidence faults

- frequent but wrong pseudo consensus；
- soft verifier overconfidence；
- verifier timeout 被错误当 0 reward；
- tool returns success code but state unchanged；
- state invariant 部分缺失；
- user simulator adversarial feedback；
- delayed failure receipt；
- evidence/reward producer hash 被替换。

### 14.2 Policy/data identity faults

- rollout adapter 未同步；
- base 相同但 LoRA adapter 不同；
- local current-model log-prob 冒充 behavior log-prob；
- chat template/tokenizer revision 变化；
- observation tokens 被错误加入 action loss；
- tail-length completion mask 错位；
- candidate adapter late callback 覆盖已 commit version；
- rollback 后 optimizer state 未恢复。

### 14.3 Split/evaluation faults

- loader 默认 train 却标 dev/test；
- adaptation/sentinel ID 不重叠但 template 重叠；
- hidden evaluator response 泄漏进 memory；
- test 后重新选择 evidence weight；
- failed tasks 被静默过滤；
- retry success 冒充 future transfer。

### 14.4 Environment faults

- alternatives 大多为 no-op；
- 同组 all-success/all-fail；
- branch restore 不完全；
- user simulator seed 不可重放；
- irreversible action 被假装可 rollback；
- parallel env 共用数据库污染；
- stale external service 导致结果漂移。

每个 fault 都必须绑定 expected decision、reason code 和 negative control。至少保留一个“训练 reward/loss 看起来正常，但最终被 integrity Gate 拒绝”的演示。

---

## 15. 从旧 `grpo-credit-assignment` 必须继承的教训

本项目不能依赖实现电脑能够访问旧仓库，因此这里记录不可丢失的约束。旧仓库不是新项目代码基础，只是事故档案。

### 15.1 已知 production failure

| 旧问题 | 为什么致命 | Agent-TTRL 强制措施 |
|---|---|---|
| rollout service policy 固定，trainer 在更新 | 表面在 RL，实际持续 off-policy | 每次 generation 绑定 base/adapter/version；真实 sync + canary |
| 用 current model 重算 `old_logp` | ratio/KL 语义失效 | authoritative behavior log-prob source 唯一化 |
| rollout text 被 trainer 重新 chat-template/tokenize | token、mask、log-prob 不同一身份 | 使用 producer token artifact；update API 禁止文本 fallback |
| completion mask 按尾长、action mask 用 `str.find` | observation/prefix 可能被训练 | structured spans + causal shift tests |
| branch/dead path 没进入 optimizer | 所谓方法没有执行 | update-consumption test + gradient reachability |
| all-token ablation 实际没有改变 loss | 假消融 | 每个 ablation 保存 effective config 与 gradient diff |
| eval 默认 train，却称 held-out | 泛化 claim 无效 | split manifest + content/template hashes + fail closed |
| ALFWorld alternatives 多为 no-op | branch oracle/credit 退化 | action-effect/non-zero oracle preflight |
| ToolEnv 同组全成功或全失败 | group advantage 恒定 | reward variance、non-zero advantage、success/fail mix Gate |
| branch estimator 未公平计 continuation/restore | 伪 sample efficiency | CostLedger 逐项计费 |

### 15.2 禁止继承的旧结论

- 不使用旧 Agent success curves；
- 不使用旧 `ρ=0.735`；
- 不把 CPC/PC-RSG/RMTPG 描述成已验证有效算法；
- 不把旧 train split 结果改名 held-out；
- 不从旧日志手抄数字进新论文；
- 不把旧投稿状态当新方法背书。

### 15.3 可复用资产

若相应新仓 release 已存在，可复用：

- GRPO-Guard 的 versioned envelope/events/artifact schemas；
- policy/token/mask/log-prob validator；
- fault-injection fixtures；
- Agent-RL Credit Auditor 的 CostLedger、branch protocol、selection-support checks；
- exact finite-MDP/SCM 作为 counterfactual estimator 单元测试；
- claim-scoped PASS/FAIL/INVALID report；
- 旧失败类型作为 regression semantics。

不可直接复制：

- dirty training scripts；
- 旧绝对路径和环境；
- 未锁依赖的 ALFWorld assets；
- 旧 checkpoint 作为新 baseline；
- 没有 protocol/source hash 的 JSON 结果。

### 15.4 与 Guard/Auditor 的所有权边界

- GRPO-Guard 是在线 policy/token/log-prob/mask lineage schema 的唯一 owner；
- Agent-TTRL pin 某个 Guard release，只读消费 canonical events；
- Agent-TTRL 自己拥有 `AdaptationStreamManifest`、`EvidenceBundle`、`CandidateAdapterDecision`；
- Credit Auditor 拥有 estimand/cost/branch audit，不改写 Guard artifacts；
- 新项目不能复制一套稍有差异的 canonical hashing。

---

## 16. 系统架构

### 16.1 模块

```text
Task Stream
  │
  ├─ Environment Adapter ── State Snapshot Store
  │          │
  │          └─ Accessible Evidence Producers
  │
  ├─ Rollout Runtime ── GRPO-Guard Events
  │
  ├─ Critical Decision Selector
  │          │
  │          └─ Branch Scheduler / Coupler
  │
  ├─ Evidence Composer ── Credit Builder
  │
  ├─ LoRA Update Worker ── Candidate Adapter Store
  │
  ├─ Shadow Evaluator ── Confidence Monitor
  │
  ├─ Commit Coordinator ── Runtime Sync / Rollback
  │
  └─ Offline Hidden Evaluator ── Paper Report
```

### 16.2 数据对象

#### `OnlineAdaptationManifest`（online process 唯一可见）

```yaml
schema_version: agent-ttrl.online-stream.v1
stream_id: ...
domain_id: ...
seed: 0
reset_scope: domain_seed
adaptation_refs: [...]
split_hash: ...
allowed_signal_classes: [hard_evidence, calibrated_soft_evidence]
forbidden_signal_classes: [hidden_outcome]
```

#### `GateManifest` 与 `SealedAuditManifest`

```yaml
schema_version: agent-ttrl.gate-manifest.v1
candidate_stream_id: ...
opaque_sentinel_capability: ...
opaque_anchor_capability: ...
max_candidates: ...
max_exposure_per_item: 1
protocol_sha256: ...
---
schema_version: agent-ttrl.sealed-audit.v1
opaque_candidate_audit_capability: ...
opaque_future_holdout_capability: ...
unlock_condition: FINAL_ADAPTER_AND_RUN_MANIFEST_SEALED
recipient_public_key: ...
```

#### `EvidenceBundle`

```yaml
schema_version: agent-ttrl.evidence.v1
trajectory_envelope_ref: ...
environment_state_before_sha256: ...
environment_state_after_sha256: ...
hard_evidence: [...]
soft_evidence: [...]
hidden_evaluator_ref: null
calibration_profile_sha256: ...
missingness: ...
cost_ledger_ref: ...
```

#### `BranchRecord`

```yaml
branch_id: ...
parent_trajectory_ref: ...
decision_span_ref: ...
selection_probability: ...
restore_snapshot_sha256: ...
behavior_policy_ref: ...
alternative_action_tokens_ref: ...
continuation_protocol_sha256: ...
coupling_seed: ...
evidence_bundle_ref: ...
```

#### `CandidateAdapterDecision`

```yaml
candidate_id: ...
parent_adapter_ref: ...
candidate_adapter_sha256: ...
update_input_event_ref: ...
shadow_protocol_sha256: ...
gain_bound: {lower: ..., upper: ..., alpha: ..., n: ...}
risk_bound: {lower: ..., upper: ..., alpha: ..., n: ...}
guard_decision_ref: ...
decision: COMMIT|ROLLBACK|QUARANTINE
reason_codes: [...]
commit_event_ref: ...
```

#### 正式 schema 的不可协商约束

上面的 YAML 只是便于阅读的投影；实现必须提供 Draft 2020-12 JSON Schema，且：

- 根对象 `additionalProperties=false`；
- 所有 ID/hash/version/role/decision 字段 required；SHA-256 用 `^[0-9a-f]{64}$`；
- `OnlineAdaptationManifest` 的 schema 中根本不存在 sentinel/audit/holdout 字段；
- `EvidenceBundle.hidden_evaluator_ref` 必须是 JSON `null`，不是可选字符串；
- `BranchRecord.selection_probability∈(0,1]`，`coupling_seed` 为 64-bit integer；
- `UpdateRow` 必须引用 producer token IDs、mask、authoritative old-logprob artifact 和 advantage；禁止 raw-text fallback 字段；
- `CandidateAdapterDecision` 的 `COMMIT` 分支必须同时有 Guard `ALLOW`、Gate bounds、fencing epoch 与 commit event；其他分支不得带 authoritative commit；
- 未知 schema major version fail closed；minor-compatible reader 只能忽略 schema 明确声明为 diagnostic 的扩展。

M0 必须交付并用 positive/negative fixtures 验证至少六个文件：`online_stream`、`gate_manifest`、`sealed_audit`、`evidence_bundle`、`branch_record/update_row`、`candidate_adapter_decision`。只有文档示例而没有可运行 schema 时，不得进入 M1。

### 16.3 建议仓库结构

```text
agent-ttrl/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── Makefile
├── CITATION.cff
├── configs/
│   ├── compatibility/
│   ├── models/
│   ├── environments/
│   ├── protocols/
│   └── experiments/
├── schemas/
│   ├── online_stream.schema.json
│   ├── gate_manifest.schema.json
│   ├── sealed_audit.schema.json
│   ├── evidence_bundle.schema.json
│   ├── branch_record.schema.json
│   ├── update_row.schema.json
│   └── candidate_adapter_decision.schema.json
├── src/agent_ttrl/
│   ├── contracts/
│   ├── environments/
│   ├── rollout/
│   ├── evidence/
│   ├── branching/
│   ├── credit/
│   ├── optimization/
│   ├── safe_commit/
│   ├── evaluation/
│   ├── cost/
│   └── reporting/
├── benchmarks/
│   └── controlled_tool_shift/
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   ├── faults/
│   └── statistical/
├── scripts/
│   ├── smoke.py
│   ├── run_stream.py
│   ├── run_branch_audit.py
│   ├── run_shadow_eval.py
│   ├── evaluate_hidden.py
│   └── build_report.py
├── protocols/
├── artifacts/
├── reports/
├── docs/
└── paper/
```

### 16.4 CLI 目标

```bash
agent-ttrl smoke --config configs/experiments/tiny.yaml

agent-ttrl run-stream \
  --protocol protocols/appworld_qwen3_4b_seed0.yaml \
  --mode proposed

agent-ttrl audit-run --run-id RUN_ID

agent-ttrl build-report --experiment-group MAIN_C1

# 只能在独立 sealed-audit container/credential 中存在：
agent-ttrl-audit evaluate-candidates --sealed-manifest SEALED.json
agent-ttrl-audit evaluate-future --final-adapter-hash HASH
```

CLI 必须把“生成任务”“online update”“hidden evaluation”拆成不同权限入口，避免 evaluator 被训练代码误用。

---

## 17. 依赖、运行时和 compatibility Gate

### 17.1 v0.1 唯一实现路径

- Python `3.11.11`；
- PyTorch `2.7.1` + CUDA `12.6` wheel；
- Transformers `4.53.2`；
- PEFT `0.16.0`；
- vLLM `0.10.0`；
- **自有最小 PyTorch clipped action-token objective**，不以 TRL/verl Trainer 作为算法语义 owner；
- vLLM 只负责 generation/authoritative behavior log-prob 与 LoRA adapter load/unload；
- Ray 仅在确有并发需要时引入；
- Pydantic / JSON Schema；
- uv lock；
- W&B 或 TensorBoard；
- pytest / hypothesis；
- Docker；
- AppWorld；
- tau2-bench。

TRL/verl 可以作为对照或后续 portability backend，但 v0.1 主结果不得在两者之间临时切换。若以上版本无法通过 compatibility smoke，M0 只能发布新的完整 profile version；不得只改某个包继续沿用旧 protocol hash。

### 17.2 `M0_IMPLEMENTATION_PROFILE`

首个实现电脑必须从以下模板生成 immutable `protocols/M0_IMPLEMENTATION_PROFILE.yaml`。模型 ID 与软件版本已选定；外部仓库/模型 revision 必须解析为完整 commit SHA，禁止追踪 `main`：

```yaml
profile_date: ...
python: 3.11.11
pytorch: 2.7.1+cu126
cuda_runtime: 12.6
driver: ...
transformers: 4.53.2
peft: 0.16.0
rl_backend: agent_ttrl.native_clipped_v1
vllm: 0.10.0
flash_attn: disabled_for_v0.1
primary_model:
  id: Qwen/Qwen3-4B-Instruct-2507
  revision: REQUIRED_FULL_SHA
second_model:
  id: mistralai/Mistral-7B-Instruct-v0.3
  revision: c170c708c41dac9275d15a8fff4eca08d52bab71
appworld:
  git_commit: REQUIRED_FULL_SHA
  data_manifest_sha256: REQUIRED_SHA256
tau2:
  git_commit: REQUIRED_FULL_SHA
  selected_domains: [retail, telecom]
  split_manifest_sha256: REQUIRED_SHA256
grpo_guard:
  release: REQUIRED_SEMVER_AT_LEAST_0_1
  schema_root_sha256: REQUIRED_SHA256
baseline_registry:
  gtta: REQUIRED_COMMIT_AND_ADAPTER_SPEC
  olivia: REQUIRED_COMMIT_OR_FAITHFUL_REIMPLEMENTATION_HASH
  memopilot: REQUIRED_COMMIT_OR_NOT_COMPARABLE_REPORT
  causalflow: REQUIRED_COMMIT_OR_FAITHFUL_REIMPLEMENTATION_HASH
```

`REQUIRED_*` 不是允许带入 M1 的占位符，而是 M0 的显式工作项。profile validator 发现任一 `REQUIRED_`、浮动 revision 或未锁依赖即退出非零。AppWorld 官方存在 `train/dev/test_normal/test_challenge`，tau2 每个 domain 提供 `split_tasks.json`；实现机必须从实际 pinned release 生成角色 manifest，不能从本文档猜 task IDs。

### 17.3 M0 可执行命令与产物

```bash
uv sync --frozen
agent-ttrl freeze-profile --template configs/compatibility/v0.1.yaml \
  --output protocols/M0_IMPLEMENTATION_PROFILE.yaml
agent-ttrl build-splits --profile protocols/M0_IMPLEMENTATION_PROFILE.yaml \
  --output-dir protocols/splits
agent-ttrl build-cts-fixtures --spec configs/environments/cts_v0.yaml \
  --output-dir benchmarks/controlled_tool_shift/fixtures
agent-ttrl validate-contracts --strict
pytest -q tests/contracts tests/controlled_tool_shift tests/capabilities
```

M0 expected artifacts：完整 implementation profile 及 SHA；`uv.lock`；容器 digest；六个 JSON Schemas；AppWorld/tau2 role manifests 与 overlap report；CTS-F01–F12 golden pack；baseline registry；`M0_DECISION.json`。任一项缺失时 M0=`FAIL`，但这是有效负结果，不启动 GPU M1。

### 17.4 Compatibility Gate

正式实验前至少完成：

- single-GPU tiny rollout；
- 一次 LoRA optimizer commit；
- rollout runtime 加载 candidate adapter；
- adapter canary 能区分 parent/candidate；
- branch state restore hash 一致；
- action-token loss mask 正确；
- hidden evaluator import guard 生效；
- rollback 恢复 adapter + optimizer + runtime；
- crash/restart 后 manifest 可恢复。

### 17.5 M1/M2 最小执行入口

M1 只使用 CTS fixtures，不碰 public test roles：

```bash
agent-ttrl run-stream --protocol protocols/m1/cts_frozen.yaml --variant frozen
agent-ttrl run-stream --protocol protocols/m1/cts_naive.yaml --variant naive_always_commit
agent-ttrl run-stream --protocol protocols/m1/cts_egc.yaml --variant egc_always_commit
agent-ttrl inject-faults --manifest protocols/faults/F01_F12.yaml
agent-ttrl audit-run --run-group M1 --require-all-artifacts
```

M1 每个 run 必须产出 `run_manifest.json`、canonical event log、token/mask/logprob artifacts、branch `U` matrix、UpdateRows、parent/candidate adapter hashes、cost ledger、Gate/rollback events、audit decision。expected：CTS-F01–F09 credit/gradient sign 正确，F10–F12 与 pipeline F01–F12 reason-coded fail closed，rollback 后 logits/hash 满足 tolerance。

M2 只在 public `dev/calibration` roles 做 baseline reproduction，不生成可报告 test 数字：

```bash
agent-ttrl run-baseline-suite --registry protocols/baseline_registry.yaml \
  --environments appworld,tau2 --models primary,second --roles dev,calibration
agent-ttrl verify-baselines --contract protocols/baseline_acceptance.yaml
```

必须分别执行 GTTA、OLIVIA、MemoPilot-compatible、CausalFlow-style repair/reuse、Frozen、Best-of-N、naive LoRA-GRPO 与 hard-verifier RL；不再使用“GTTA 或 OLIVIA”“strongest available”这类运行时选择。某工作不可复现时输出 signed `NOT_COMPARABLE_REPORT`，说明缺代码/接口/信号差异及采取的 faithful control，不能静默换弱 baseline。M2 expected artifacts 为逐 baseline environment adapter spec、upstream commit、effective config、reproduction delta、raw cost ledger 和 acceptance decision。

---

## 18. 测试要求

### 18.1 Unit tests

- evidence missing/timeout semantics；
- state hash canonicalization；
- branch selection support；
- cost calculator；
- action span → token mask；
- confidence Gate boundary；
- adapter manifest hash；
- no hidden evaluator reference in update batch。

### 18.2 Property tests

- identical state/action/seed ⇒ identical deterministic branch；
- branch restore 后主 world state 不变；
- cost 只随真实 counted operation 增加；
- text change但 token artifact 不变时 validator 拒绝；
- action mask 不覆盖 observation tokens；
- rollback 后 logits 与 parent 在 tolerance 内一致；
- shuffled evidence 不应稳定提高 exact credit metric。

### 18.3 Integration tests

- rollout → evidence → branch → credit → LoRA update → SafeCommit → sync；
- candidate PASS；
- candidate ROLLBACK；
- verifier timeout quarantine；
- stale runtime rejection；
- crash at candidate write / commit / sync 各阶段恢复；
- parallel streams 不串 adapter；
- hidden evaluator access attempt fail closed。

### 18.4 Statistical tests

- fixed-n Hoeffding Gate 在跨 candidate alpha allocation 下的 empirical family-wise coverage；
- false commit under null；
- power under known positive effect；
- bootstrap interval coverage on synthetic hierarchy；
- user-simulator paired seed variance；
- selector-induced sampling bias diagnostic。

### 18.5 Paper artifact tests

- 每张表能从 raw manifests 重建；
- 图中点数与 CSV/JSON 一致；
- failed/invalid runs 未被静默删除；
- README 命令在 clean container 运行；
- model/dataset licenses 清单完整；
- anonymous release 不泄漏作者信息。

---

## 19. 成本与算力

### 19.1 先计原始操作，再报 GPU-hours

primary cost 不使用可任意调权重的标量和，而是 §10.5 的硬约束向量：

\[
\mathbf C=(N_{env},N_{model\_tok},N_{update\_tok})\preceq
\mathbf B=(B_{env},B_{model\_tok},B_{update\_tok}).
\]

`branch_count` 和 `restore_count` 是解释性字段；对应 continuation 已通过 canonical events 计入 `N_env/N_model_tok`，不得再次折算后重复计费。CPU restore seconds、bytes、API fee、GPU-hours 与 wall time单独报告。只有预注册的 cost-aware appendix 才允许标量化，且权重来源必须是公开硬件/服务价格快照。

同时单独报告：

- generated/scored/update tokens；
- forward/backward passes；
- tool calls/environment steps；
- branch/restore；
- API cost；
- GPU-hours；
- wall time；
- energy（若易获得）。

不把不同硬件上的 GPU-hour 当完全可比算法成本。

### 19.2 分阶段估算

以下是规划区间，不是已测结果：

| 阶段 | 目标 | A800 GPU·h 估算 |
|---|---|---:|
| M0 | novelty/profile/schema/split/CTS freeze | 0–5 |
| M1 | correctness 与真实 LoRA/sync/rollback smoke | 20–60 |
| M2 | 两环境、两模型 strong baseline dev reproduction | 60–120 |
| M3 | EGC decision pilot 与 mechanism controls | 80–180 |
| M4 | SafeCommit coverage、candidate archive 与 stress pilot | 80–180 |
| M5 | 两公共环境 primary-model factorial main results | 300–600 |
| M6 | second-model factorial + poison/shift/generalization | 300–600 |
| M7 | appendix、预注册失败重跑、artifact freeze | 100–250 |
| 合计：最小 paper core | 不含大规模附加 sweep | 940–2,000 |
| 合计：较强主会包 | 更完整 backbones/seeds/robustness | 1,500–3,000 |

组合建议：

- 日常开发：2×A800 40GB；
- 主结果：4–8×A800 并行不同 streams/seeds；
- 不要求一次把 8 卡用于同一 3B 模型；Agent rollout 经常受环境/API 延迟限制；
- 先用 pilot 实测 tokens/s、environment latency 和 utilization，再冻结正式预算。

### 19.3 Stop-loss

- 60 GPU·h 前必须完成 correctness + baseline smoke；
- 250 GPU·h 前必须看到 C1 在 controlled/public subset 的方向性；
- 600 GPU·h 前必须做 matched-cost random/equal-extra-rollout control；
- 若 600 GPU·h 后主效应仍只存在于 toy env，停止主会扩展；
- 不用更大模型掩盖机制失败。

### 19.4 存储

规划：

- trajectory/evidence/branch artifacts：200–600 GB；
- checkpoints/adapters：50–200 GB；
- environment snapshots：视 AppWorld/tau2 实现，预留 100–500 GB；
- 只压缩归档，不删除失败 manifest；
- raw CoT 若受 license/privacy 限制，保存 hash、结构化 action 与必要 audit 字段。

---

## 20. 10–14 周执行路线

### 20.0 唯一 milestone registry

三份文档统一使用：`M0=novelty/profile/protocol freeze`，`M1=correctness`，`M2=strong baseline reproduction`，`M3=EGC mechanism pilot`，`M4=SafeCommit pilot`，`M5=primary-model public factorial`，`M6=second-model/generalization/stress`，`M7=freeze/paper/artifact audit`。代码、tracker 和目录名不得自行重定义 M2/M3。

M0–M2 的输入/输出分别由 §17.3/§17.5 冻结；M3 以后必须从 tracker registry 生成命令，不允许手工临时起“未登记 run”。

### Week 0：Novelty 与 protocol freeze

- 完成 §4 claim matrix；
- 冻结 C1/C2、anti-claims、primary endpoints；
- 生成并验证 §17.2 的单一 native-backend implementation profile；
- 冻结资源 stop-loss。

Gate：没有直接覆盖论文；否则 pivot。

### Week 1：ControlledToolShift

- environment、snapshot、hidden oracle；
- partial evidence producers；
- non-degeneracy tests；
- split manifest。

Gate：exact restore、branch isolation、oracle 可复算。

### Week 2：Guarded online LoRA loop

- rollout/runtime；
- behavior log-prob；
- action-token mask；
- candidate adapter；
- runtime sync/rollback。

Gate：fault matrix 全过，hidden evaluator 无法进入 update。

### Week 3：强 baseline

- Frozen/Best-of-N；
- ACE/Reflection；
- GTTA/OLIVIA 中至少一个强适配 baseline；
- naive LoRA-GRPO/TTRL-style。

Gate：baseline 达到合理复现范围；否则先修 baseline。

### Week 4：EGC credit

- selector；
- matched branch；
- evidence calibration；
- signed credit；
- exact mechanism diagnostics。

Gate：在 controlled env 胜过 random/unpaired branch。

### Week 5：SafeCommit

- shadow sentinel；
- confidence monitor；
- atomic commit；
- poison/shift simulator。

Gate：coverage 测试通过，commit rate 非退化。

### Week 6–7：AppWorld 主 pilot

- 环境 adapter；
- 3 seeds；
- cost-matched main variants；
- first decision memo。

Gate：future-task effect 有方向性；没有则 kill/pivot。

### Week 8–9：tau2 + 第二 backbone

- tau2 mock smoke、retail + telecom main；
- user seed protocol；
- second family；
- primary table。

Gate：至少两个环境方向一致，异质性可解释。

### Week 10：Ablation 与 stress

- equal-extra-rollout；
- SafeCommit stress；
- verifier coverage；
- branch budget；
- negative controls。

### Week 11：统计与 artifact freeze

- hierarchical bootstrap；
- CI、effect sizes；
- manifest/hash；
- reproduce tables from raw files。

### Week 12–14：论文

- 主文只围绕 C1/C2；
- external reader / kill memo；
- claim audit、citation audit、artifact audit；
- 根据 venue 匿名要求处理开源仓库。

---

## 21. Experiment tracker 约定

Run ID：

```text
ATTRL-{milestone}-{env}-{model}-{method}-{seed}-{revision}
```

例如：

```text
ATTRL-M3-CTS-Q4B-EGC-000-r1
```

状态拆为三列：

```text
execution_status = PLANNED|QUEUED|RUNNING|COMPLETED|CRASHED
audit_status     = UNAUDITED|VALID|INVALID
claim_outcome    = NOT_APPLICABLE|PASS|FAIL|INCONCLUSIVE
```

只有 `execution_status=COMPLETED AND audit_status=VALID` 的 run 可进入正式汇总；其中 `claim_outcome=FAIL` 的有效负结果也必须进入。`UNAUDITED` 不能写简历或论文数字。

---

## 22. 顶会主文故事

### 22.1 一句话 thesis

> Test-time RL for stateful tool agents should update from evidence, not self-confidence: paired counterfactual execution identifies which actions deserve credit, while sequential commit gates prevent uncertain updates from contaminating future behavior.

### 22.2 主文结构

1. Agent 部署期反馈是 partial 且 endogenous；
2. trajectory/self-consistency reward 会把频繁错误写回 policy；
3. EGC 用 matched state branches 产生 signed action evidence；
4. SafeCommit 把 adaptation 与 deployment commit 分离；
5. 两个 stateful environments 上，在三维 hard caps 下改善 `AUPC_prequential` 与 sealed holdout；
6. poison/shift 下显著减少 catastrophic update；
7. 机制实验说明结果不是更多 rollout 或 hidden evaluator。

### 22.3 主图/表

- Figure 1：问题与方法总览；
- Table 1：两环境、两 backbone 主结果；
- Figure 2：prequential learning curves；
- Table 2：credit mechanism ablation；
- Figure 3：credit fidelity vs future improvement；
- Table 3：SafeCommit stress；
- Figure 4：success–cost–risk Pareto；
- Appendix：correctness、split、statistics、fault matrix。

### 22.4 Reviewer 最可能的质疑

#### “这是 T3RL 从数学搬到 Agent。”

回答所需证据：T3RL-style hard verification baseline；展示 state-changing multi-step partial evidence、signed local credit 和 future-task commit 的新增难点。

#### “这是 GTTA/ACE 加 LoRA。”

回答所需证据：GTTA/ACE baseline；matched context；证明 counterfactual credit 与 SafeCommit 对参数 RL 是必要的。

#### “CausalFlow 已经做了 step-level counterfactual intervention。”

回答所需证据：不声称首次定位/修复；运行 CausalFlow-style minimal repair 与 same-branch repaired-data control；C1 只比较 partial-evidence signed credit 如何驱动 online LoRA update，并以未来 prequential transfer 而非当前 trace recovery 判胜负。若 matched control 持平，删除 C1 headline。

#### “你在 test set 上训练。”

回答所需证据：prequential protocol、future holdout、hidden evaluator isolation、stream reset。

#### “提升来自额外 branch compute。”

回答所需证据：equal-extra-rollout、random branch、三种 cost matching。

#### “counterfactual credit 不无偏。”

回答：不声称对 hidden return 无偏；明确 shaped partial-evidence objective；controlled oracle 只验证 fidelity 与 utility。若想升级理论 claim，另做严格假设和证明。

#### “SafeCommit 只是拒绝更新。”

回答所需证据：commit rate、AUPC retained、always-rollback、oracle-commit、risk–gain curve。

#### “环境能 snapshot，不现实。”

回答：scope 是 sandboxable/replayable stateful tool systems；提供无 branch fallback 作为边界，不外推到不可逆现实系统。

#### “两个模块太复杂。”

回答所需证据：EGC-only、SafeCommit-only、full；展示两者分别对应 credit 与 contamination，且没有额外 world-model 大模块。

---

## 23. 论文级结果门槛

### 23.1 Portfolio Release

- 一个公共环境；
- 一个 open backbone；
- 3 seeds；
- 真实 online LoRA update；
- one-command smoke；
- matched-cost baseline；
- 不使用 hidden evaluator 更新；
- 完整 failure report。

达到这里只能说“完成开源工程/研究原型”。

### 23.2 Workshop / Findings Candidate

- 两个环境或一个环境多个 hard shifts；
- C1 有稳定方向；
- 主要 ablation；
- 初步 SafeCommit；
- 严格 split 与统计。

### 23.3 Main-conference Candidate

- 两个公共 stateful environments；
- 两个模型家族；
- 3–5 seeds；
- C1/C2 均有直接证据；
- strongest baselines；
- matched cost；
- hidden holdout；
- mechanism + negative controls；
- safety/poison stress；
- artifact 可复现；
- 最新 novelty refresh 通过；
- 外部无上下文 reviewer 无 BLOCKER/HIGH。

这仍不保证录用。顶会录用还受 reviewer fit、写作、同期工作和结果强度影响。

---

## 24. 简历启用 Gate 与写法

### 24.1 当前阶段

当前只能在个人规划中写：

> Agent-TTRL research proposal / planned；尚未完成，不进入简历核心项目正文。

### 24.2 Portfolio Release 后

模板：

> **Agent-TTRL：状态化工具 Agent 的可验证推理时强化学习系统｜PyTorch、LoRA、GRPO、vLLM**  
> 构建 deployment-time rollout–evidence–LoRA update–shadow evaluation–rollback 闭环；以严格的 adaptation/sentinel/future-holdout split 评估连续任务学习，并记录 policy/token/log-prob 与环境状态 lineage。  
> 在 `[环境/模型/种子]` 上，相对 `[最强 baseline]` 于 matched `[token/tool-call/GPU]` 预算下获得 `[经审计指标与 CI]`；对 `[故障类型]` 的 candidate update 实现 `[经审计 rollback/风险结果]`。

方括号只有 release manifest 支撑后才能替换。

### 24.3 论文提交后

如未录用：

> Manuscript under review / Submitted to `[venue]`。

不写“顶会论文”或把 venue 年份伪装成录用。

### 24.4 90 秒面试故事

> 我之前做 Agent GRPO credit assignment 时发现，线上 rollout policy 没有随 trainer 更新，old-logprob、token 和 mask 的身份也不闭合，所以我停止使用受影响的成功率结论。之后我先做 Guard 把在线更新链路校验起来，再研究一个更实际的问题：部署期 Agent 会持续收到不完整的工具反馈，如果直接拿这些反馈做 test-time RL，错误会污染后续任务。我的方法只在可重放的状态化环境中，对关键 action 做 matched counterfactual branch，用执行证据形成 signed credit；LoRA 更新不会直接上线，而是经过 paired shadow evaluation 和 confidence Gate，再 commit 或 rollback。评测采用 prequential stream 和完全未参与更新的 future holdout，并把 branch、tool call、update 和 verifier 成本全部计入。

### 24.5 面试追问准备

- Test-time scaling、adaptation、training、RL 的区别；
- 为什么不是测试集泄漏；
- verifier 可见什么、hidden evaluator 为什么隔离；
- branch credit 为什么不声称无偏；
- behavior policy 与 current policy；
- LoRA adapter 如何同步到 rollout runtime；
- action token mask 如何构造；
- 跨 candidate 错误预算为什么优于无校正的重复固定阈值；
- sentinel 为什么会 overfit；
- matched-cost 怎么算；
- negative transfer 如何定义；
- 为什么不直接用 ACE/OLIVIA/GTTA；
- 如果主效应失败，保留什么工程价值。

---

## 25. 开源与 artifact 方案

### 25.1 建议独立仓库

新建干净仓库 `agent-ttrl`，不要从旧 dirty `grpo-credit-assignment` 复制历史。

公开内容：

- ControlledToolShift；
- schemas/contracts；
- Guard adapter；
- evidence/branch/cost modules；
- SafeCommit；
- configs；
- tests；
- tiny model smoke；
- result manifests；
- paper reproduction scripts。

可能不公开：

- 受 benchmark license 限制的数据；
- raw proprietary user/API traces；
- 投稿匿名期身份信息；
- 无再分发权的 model weights。

### 25.2 Release checklist

- clean clone 可安装；
- exact lockfile；
- Docker smoke；
- SPDX/license audit；
- model/dataset cards；
- SHA256SUMS；
- immutable release tag；
- `KNOWN_FAILURES.md`；
- `CLAIM_SCOPE.md`；
- raw→table reproduction；
- artifact size 与下载说明；
- anonymity policy 检查。

---

## 26. 第一批运行顺序

### 前三个 run

1. `R001`：ControlledToolShift frozen + exact hidden oracle，验证 task、split、state snapshot；
2. `R002`：naive LoRA-GRPO tiny overfit，验证真实 policy update 和 runtime sync；
3. `R003`：手工构造一正一负 decision 的 paired branch，验证 credit sign、action mask 与 cost ledger。

在 R001–R003 通过前，不启动 AppWorld/tau2 正式实验。

### 前 72 小时交付

- 新仓库与 lockfile；
- `RESEARCH_CONTRACT.md`；
- `split_manifest.schema.json`；
- ControlledToolShift 最小环境；
- parent/candidate adapter identity test；
- exact branch restore test；
- hidden evaluator import guard；
- 10–20 个 unit tests；
- 一份失败也能生成的 audit report。

---

## 27. 风险表与降级路线

| 风险 | 早期信号 | 缓解 | 降级后的诚实成果 |
|---|---|---|---|
| 新颖性被近期论文覆盖 | claim matrix 高重合 | 只保留未覆盖主张或换题 | 开源工程，不投稿算法主会 |
| public env 无法 exact branch | restore/coupling 不可靠 | 主方法限 controlled/AppWorld；approx branch 仅 appendix | SafeCommit/monitoring paper |
| partial evidence 与 hidden success 弱相关 | calibration/coverage 差 | selective abstention、停止参数更新 | verifier failure benchmark |
| C1 只靠额外 compute | equal-extra-rollout 持平 | 改 selector/credit；不扩模型 | test-time compute study |
| SafeCommit 拒绝过多 | commit rate 接近 0 | 调整 dev-frozen risk budget、提高 verifier | C1-only paper |
| 同任务提升、未来任务无提升 | AUPC/future holdout 不动 | 学 task-family abstraction，而不是任务记忆 | recovery-only portfolio |
| user simulator 方差过大 | seed CI 很宽 | recorded users、更多 paired seeds | 只保留 deterministic env claim |
| LoRA update 太慢 | update 占 wall time 主导 | rank/steps 缩小、async shadow | OLIVIA/context baseline 更合理，kill RL |
| Guard integration 失败 | stale policy/token mismatch | 先修 correctness，不跑结果 | Guard 本身作为工程核心项目 |
| 预算不足 | 250 GPU·h 前无 pilot | 只做一个 public env 的作品集版 | 不声称顶会级广泛有效 |

---

## 28. Definition of Done

### 28.1 工程 DoD

- [ ] clean repo、lockfile、Docker；
- [ ] ControlledToolShift deterministic；
- [ ] AppWorld/tau2 至少一条正式 adapter；
- [ ] rollout/update/rollback 闭环；
- [ ] Guard lineage；
- [ ] hidden evaluator 隔离；
- [ ] CostLedger；
- [ ] fault matrix；
- [ ] crash recovery；
- [ ] one-command smoke；
- [ ] raw artifacts 与 hashes。

### 28.2 算法 DoD

- [ ] C1/C2 与 anti-claims 预注册；
- [ ] strongest baselines；
- [ ] equal-cost controls；
- [ ] exact mechanism environment；
- [ ] public future-task gain；
- [ ] SafeCommit 非退化；
- [ ] negative controls；
- [ ] two environments；
- [ ] two model families；
- [ ] hierarchical statistics；
- [ ] failure interpretation。

### 28.3 投稿 DoD

- [ ] novelty refresh；
- [ ] claim matrix；
- [ ] main tables 可重建；
- [ ] paper claims 与 manifest 一致；
- [ ] limitations 不隐藏 reset/verifier 假设；
- [ ] citation audit；
- [ ] external Reader Test；
- [ ] kill memo 已回答；
- [ ] anonymous artifact；
- [ ] 代码/数据 license；
- [ ] 不承诺顶会录用。

---

## 29. 最终决策

Agent-TTRL 值得做，但正确策略不是“再堆一个 GRPO 项目”，而是把现有研究与工程链闭合：

```text
Reward Hacking
      ↓ 说明 reward 会被利用
GRPO-Guard
      ↓ 保证 online trajectory/update 身份可信
Credit Auditor
      ↓ 审计 estimand、branch 和 fixed cost
Agent-TTRL
      ↓ 在可信链路上研究 deployment-time adaptation
```

真正达到核心简历项目和主会候选的条件是：

> 它必须在严格的 future-task protocol 下，用可审计的在线参数更新获得稳定效果；核心 gain 经 matched-cost 和机制消融后仍成立；SafeCommit 真正减少污染而不是拒绝所有学习；所有结果可由 release artifact 重建。

在这些 Gate 通过前，项目状态始终是 `PLANNED/IN PROGRESS`，不写成已完成或已证明创新。

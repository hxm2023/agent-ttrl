# Agent-TTRL（GitHub `c6dddf`）顶会级逐页、逐段、逐图、逐实验与代码审稿

> 审稿结论：**Strong Reject（2/10，confidence 5/5）**。  
> 当前版本相较 v1 有实质进步：作者主动推翻了错误结果，并让同一 PEFT 模型同时承担训练和生成。然而，当前论文仍不满足 ACL/EMNLP/NeurIPS/ICML/ICLR 主会的最低证据门槛。最严重的问题不是任务太少，而是**正文、图和逐模板结果混用了两套重复运行的证据；标题方法 EGC 没有优于 naive replay；主要正结果完全由一个 adaptation template 驱动；实现又存在训练—服务隔离、随机数、隐藏状态泄漏、replay、统计和 artifact provenance 等阻断问题**。

---

## 0. 审稿对象、证据边界与可复核快照

### 0.1 锁定对象

- GitHub 仓库：[`hxm2023/agent-ttrl`](https://github.com/hxm2023/agent-ttrl)
- 审稿提交：[`c6dddfecd20ebe33de7a4cc5886d80e218896002`](https://github.com/hxm2023/agent-ttrl/tree/c6dddfecd20ebe33de7a4cc5886d80e218896002)
  - commit time：`2026-08-24T22:05:02+08:00`
  - message：`v2 run logs forced-in (evidence: per-seed AUPC, 43 logs)`
- 论文：[`paper/main.pdf`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/paper/main.pdf)
  - SHA256：`4b0cc4f6bb3b35ec2ce7d2b33b66a4ee05eb06b86aed681e9ff6d789e30f3120`
  - 6 页，PDF metadata creation time：2026-08-24 21:25 CST
- 正式结果证据目录：[`protocols/runs/v2/cts`](https://github.com/hxm2023/agent-ttrl/tree/c6dddfecd20ebe33de7a4cc5886d80e218896002/protocols/runs/v2/cts)
- 本审稿不把 `/data_3/repo/agood/agent-ttrl` 的旧提交 `76b0299` 当成当前版本。旧 PDF `/data_3/repo/agood/hire/main.pdf` 的 SHA 也与 GitHub 当前 PDF 不同。

### 0.2 证据优先级

```text
当前提交中的可执行代码、raw log、manifest、独立重算
    > 当前 PDF/LaTeX 文字
    > README、SUBMISSION_README、内部 audit/decision 文档
    > “finalized”“submission-ready”等状态标签
```

因此：README 说“通过”不能覆盖测试命令失败；正文说“immutable manifest”不能覆盖同一路径可被重写；caption 说“6/8”不能覆盖图中实际 7/8 的点。

### 0.3 本次实际复核

- 从 GitHub 当前 `main` 做 fresh shallow clone 并锁定上述 commit。
- 逐页视觉检查 6 页 PDF；同时用 `pdftotext -layout` 检查跨栏和引用编号。
- 逐段核对 `paper/sections/*.tex` 与代码/结果。
- 独立解析全部首轮 `_16.log`、rerun `_16a.log` 和当前 `*_manifest.json`。
- 穷举 `2^8` 个符号翻转，独立重算精确双侧 sign-flip p 值。
- 检查 runtime、request RNG、stream、replay、branch credit、统计、画图、复现和测试代码。
- fresh environment 执行：
  - `uv run --extra dev pytest -q`：**collection error**，缺少 `torch`；
  - 排除 GPU integration：**144 passed in 40.96s**；
  - `bash reproduce.sh`：**exit 127**，第 37 行 `python: command not found`。

### 0.4 审稿独立性

本次是同一会话内的代码—论文联合审计：

```yaml
review_independence: same-context-local
acceptance_status: provisional
```

因此“接收/拒绝”不是独立外审的替代；但下文的哈希、路径、逐 seed 数值、精确检验和代码控制流均可由第三方直接复核。

---

## 1. 一页式总评

### 1.1 最终判定

| 维度 | 分数 | 严格评价 |
|---|---:|---|
| 问题重要性 | 4/5 | Agent 部署期持续学习、partial evidence、future transfer 和安全 commit 都非常重要。 |
| 方法新颖性 | 2/5 | “EGC + replay + serving transaction”的组合有潜力，但核心成分与 CVT-RL、aTTT、PACE、StarOR 等高度邻近；当前 EGC 贡献未被实证。 |
| 技术正确性 | 1/5 | atomic commit、跨 arm RNG、hidden-evaluator 隔离、counterfactual continuation、replay 语义均有实现级违约。 |
| 实验可信度 | 1/5 | 两套重复运行混用、EGC artifacts 不全、正收益由一个模板驱动、无 anchor-only/官方环境/强基线。 |
| 统计严谨性 | 1/5 | 论文 p 值与 delta 不对应；“exact”脚本实为 Monte Carlo；另一脚本把 task 当 outer unit。 |
| 可复现性 | 1/5 | 默认安装无法收集完整测试；复现脚本直接失败且会生成旧版图；manifest 可覆盖。 |
| 写作与呈现 | 2/5 | 叙事方向清楚且诚实披露 heldout null，但标题、贡献和结论超出证据，版面/引用有明显错误。 |
| 总体 | **2/10** | **Strong Reject；不建议以当前形态投稿或把结果数字写进简历。** |

### 1.2 四个足以单独拒稿的主因

1. **论文主结果是 hybrid evidence。**正文的 delta/图来自首轮 logs；`p=.032`、6/8 和逐模板计数来自 rerun manifests。它们不是同一套 replicate。
2. **标题方法没有赢。**首轮 EGC 均值 `0.5000`，naive `0.5234`；EGC−naive 为 `-0.0234`，不是方法改进。论文显著结果只是 naive replay 相对 frozen。
3. **正收益没有跨技能泛化。**naive 相对 frozen 在 128 个任务上净增 9 次成功，其中 F1-exchange 单模板贡献 `+12`；去掉 exchange 后净效应为 `-3`。真正 held-out 模板为 `0/16→1/16`，近乎零。
4. **实现无法支撑系统与因果主张。**候选权重在 canary 前已经成为 serving 权重；生产 seed 包含 treatment-dependent `policy_version`；EGC 直接读取隐藏 world；R=4 是四次相同确定性调用；hidden success 还控制 early stop。

### 1.3 v2 相比 v1 的真实进步

这些资产应保留，而且很适合作为工程故事：

- 主动发布 [`AUDIT_INVALIDATION.md`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/AUDIT_INVALIDATION.md)，明确撤回 v1 静态 serving 和 RNG displacement 造成的无效结论。
- 将训练和生成收敛到同一 `ColocatedPolicy`，消除了 v1 “HF 模型更新、vLLM base 模型继续服务”的最直接断链。
- 引入 request-scoped seed、CTS-v2、跨任务 replay 和 integration test scaffold。
- 报告了 heldout null、refund/recover negative transfer，而不是只展示正面平均数。
- 保留了首轮和 rerun logs，使复现不稳定和证据版本混用能够被发现。
- 144 个非 integration tests 通过，说明 schema/CPU 组件的内部一致性有一定基础。

但这些是“好的研究与工程过程”，不是当前算法已经成立的证据。

### 1.4 当前最窄可保留结论

> 在一个自建、确定性的 CTS-v2 合成任务流上，使用 Mistral-7B 和人工 canonical anchor 的 session replay pilot，在其中一套 8-seed rerun 上相对 frozen 得到平均 `+0.0703` AUPC；该增益几乎完全来自 adaptation 中反复出现的 exchange 模板，sealed heldout 近乎为零，且运行/统计/provenance 尚存在阻断问题。当前证据不能证明 EGC 优于 naive replay，也不能证明 inductive future transfer、原子提交或官方 Agent benchmark 改进。

---

## 2. 结果取证：论文数字到底来自哪里

## 2.1 首轮 logs（图和正文 delta 的来源）

| seed | frozen | naive | naive−frozen | EGC | EGC−frozen |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.4375 | 0.5000 | +0.0625 | 0.4375 | 0.0000 |
| 1 | 0.4375 | 0.5625 | +0.1250 | 0.5000 | +0.0625 |
| 2 | 0.5000 | 0.5625 | +0.0625 | 0.5625 | +0.0625 |
| 3 | 0.5000 | 0.5625 | +0.0625 | 0.6250 | +0.1250 |
| 4 | 0.3750 | 0.4375 | +0.0625 | 0.4375 | +0.0625 |
| 5 | 0.4375 | 0.5000 | +0.0625 | 0.5000 | +0.0625 |
| 6 | 0.5000 | 0.5000 | 0.0000 | 0.4375 | **−0.0625** |
| 7 | 0.4375 | 0.5625 | +0.1250 | 0.5000 | +0.0625 |
| mean | **0.453125** | **0.5234375** | **+0.0703125** | **0.500000** | **+0.046875** |

独立穷举重算：

- naive−frozen：**7 positive / 1 zero / 0 negative**；精确双侧 `p=0.015625`。
- EGC−frozen：**6 positive / 1 zero / 1 negative**；精确双侧 `p=0.109375`。
- EGC−naive：均值 **−0.0234375**；1 positive / 3 zero / 4 negative；精确双侧 `p=0.375`。
- 普通配对 t 区间仅作描述：
  - naive−frozen 95% CI `[+0.0368,+0.1038]`；
  - EGC−frozen `[+0.0006,+0.0932]`；
  - EGC−naive `[−0.0623,+0.0154]`。

结论：如果正文列出的八个 delta 是证据，那么 caption 必须写 7/8 和 `p=.015625`，不能写 6/8 和 `.032`。

## 2.2 rerun logs / 当前 manifests（正文 p、符号数和逐模板计数来源）

| seed | frozen rerun | naive rerun | delta |
|---:|---:|---:|---:|
| 0 | 0.4375 | 0.5000 | +0.0625 |
| 1 | 0.4375 | 0.5625 | +0.1250 |
| 2 | 0.5000 | 0.5000 | 0.0000 |
| 3 | 0.5000 | 0.6250 | +0.1250 |
| 4 | 0.3750 | 0.3750 | 0.0000 |
| 5 | 0.4375 | 0.5625 | +0.1250 |
| 6 | 0.5000 | 0.5625 | +0.0625 |
| 7 | 0.4375 | 0.5000 | +0.0625 |
| mean | **0.453125** | **0.5234375** | **+0.0703125** |

- **6 positive / 2 zero / 0 negative**。
- 精确双侧 `p=0.03125`，可四舍五入为论文的 `.032`。
- 描述性配对 t 区间 `[+0.0267,+0.1139]`。
- EGC rerun 只有 seed 0–2，且仓库也只有 **3/8 EGC manifests**；不能从 rerun 构成完整 EGC 比较。

均值恰好与首轮相同，但 seed-level outcome 已改变。均值相同不能让两套 replicate 变成同一套数据。

## 2.3 逐模板分解（rerun manifests）

| template | frozen | naive | difference |
|---|---:|---:|---:|
| F1_refund | 24/24 | 20/24 | −4 |
| F1_refund_delivered | 0/24 | 0/24 | 0 |
| F1_cancel | 14/16 | 15/16 | +1 |
| **F1_exchange** | **4/16** | **16/16** | **+12** |
| F3_recover | 16/16 | 15/16 | −1 |
| F3_recover_v | 0/16 | 0/16 | 0 |
| **sealed F1_refund_v** | **0/16** | **1/16** | **+1** |
| total | 58/128 | 67/128 | +9 |

关键诊断：

- exchange 单模板贡献 `+12`，总净收益只有 `+9`。
- 去掉 exchange，naive 相对 frozen 为 **−3**。
- 这更像“对某个反复出现模板的 rehearsal/SFT 修复”，不是跨 latent skill 的一般 future transfer。
- sealed 模板的 `+1/16` 不足以支持 inductive transfer；论文自己称其为 null 是正确的，但标题和 abstract 仍把“future tasks”放在前景。

## 2.4 重复运行不稳定性

首轮与 rerun 比较：

- frozen：0/8 seeds 改变；
- naive：**6/8 seeds 改变**；
- EGC：已有 rerun 的 3 个 seed 中 **2/3 改变**。

直接原因之一是 replay 使用未 seed 的 Python `random.sample` / `random.choices`；训练又按逐 row 顺序执行，optimizer 每 row 重建，因此 batch 组成和次序会改变权重轨迹。这不是“正常 GPU 浮点噪声”可以概括的。

## 2.5 违反项目自己的预注册门槛

- [`M0_IMPLEMENTATION_PROFILE.yaml#L51-L55`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/protocols/M0_IMPLEMENTATION_PROFILE.yaml#L51-L55) 冻结 `alpha: 0.01`。
- [`RESEARCH_CONTRACT.md#L64-L65`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/RESEARCH_CONTRACT.md#L64-L65) 要求 `p<0.01 + effect size/CI`。
- 首轮 `p=.015625` 和 rerun `p=.03125` **都没有通过**。
- contract 将 `sealed_future_holdout_score` 与 inductive future transfer 定为 primary；当前 sealed probe 是 null。
- baseline registry 仍是 `status: SCAFFOLD`，大量 `REQUIRED_COMMIT` 未解析，却在 Related Work 声称 budget-matched baselines 已包含。

所以即使暂时忽略代码缺陷，当前结果也应按项目自己的 Go/No-Go 规则判为 **No-Go**。

---

## 3. 逐页严格审稿

## 第 1 页：Abstract + Introduction 前半

### 做得好的地方

- 问题设置清楚：部署期、session-scoped、tool agent、partial endogenous feedback。
- 主动解释 v1 静态 serving 失败，科研诚实性优于掩盖失败。
- 量化结果、seed 数和 heldout 性质都没有完全隐藏。

### 阻断问题

1. **Abstract 的方法与结果主体错位。**标题和方法叫 EGC-TTRL，但显著数字是 naive replay vs frozen；EGC 更弱且不显著。
2. **统计三元组不自洽。**列出的 `+0.070` 可来自两套运行；`p=.032, 6/8` 只能来自 rerun，而图/八个 delta 来自首轮。
3. **“future tasks”超出证据。**同模板循环中的后续任务属于 transductive/session adaptation；真正 sealed template 近乎零。
4. **“atomic commit”与代码相反。**`train_step()` 已直接改写 serving model；canary 只是事后检查。
5. **“signed replay”不实。**naive 只接收 successful positive rows；EGC 也只接收 `credit>0.05`，负 credit 没进入训练。
6. **“all runs carry immutable manifests”不实。**同一 `run_manifest.json` 被 `write_text` 覆盖；EGC 只有 3/8 manifests。
7. **“A0–A2 end-to-end on Mistral-7B”缺 artifact。**仓库只有测试代码，默认模型还是 Qwen2.5-0.5B；没有 Mistral 运行日志、环境/GPU/hash 证据。
8. Introduction 把自己 v1 的缺陷扩张为“most pipelines”共有 silent failure，没有系统性 audit 或引用证据。

### 修改要求

- 当前版本若只改文字，标题应降级为：`Auditing Policy Consistency in Agent Test-Time Adaptation`。
- abstract 只能写 synthetic CTS pilot 和 transductive effect；删除 EGC positive headline、atomic、immutable、future-transfer 等措辞。
- 如果坚持算法稿，必须等 EGC vs naive 在 official environments 成为 primary positive result 后再恢复原标题。

## 第 2 页：Contributions + Related Work

### 阻断问题

1. 四个 contribution bullet 中，runtime、signed replay、EGC signs、positive transfer 都至少有一项实现/证据反例。
2. “our budget-matched baselines include them”是可证伪的错误：registry 仍为 scaffold，正文结果没有 ACE、GTTA、OLIVIA、JitRL、aTTT 等任何数字。
3. aTTT 已经实现 live in-episode LoRA + runtime serving，并在 ALFWorld/SWE-bench Lite 报告正结果；当前稿只在 toy CTS 上证明“同一模型训练和生成”，系统新颖性不足。
4. CVT-RL 已经有 policy-conditioned counterfactual contribution、validity gate、doubly robust estimator和多 Agent 环境；本文的 EGC 是简化邻近版本，必须通过更强 identification 或 deployment constraint 体现差异。
5. PACE 的 anytime-valid acceptance 与论文 SafeCommit 段落高度接近，但 v2 正式实验根本没有运行 SafeCommit statistical gate。该段既占篇幅又制造未实现贡献。
6. “future commitment”“risk-controlled commit”都没有用 sealed transfer/真实 commit gate 实证。

### 修改要求

- Related Work 改成“capability matrix + empirical baseline table”，不能用文字宣称已跑。
- 至少比较 frozen ReAct、anchor-only SFT、naive replay、aTTT、CausalFlow-style repair/CVT-RL、random/unpaired credit；预算三通道严格匹配。
- 若 SafeCommit 不进入主实验，完全从本文删除，另做系统/安全稿；若保留，必须提供 adapter candidate shadow evaluation 和 commit/rollback metrics。

## 第 3 页：Problem/Protocol + Method 4.1/4.2

### 阻断问题

1. “hidden evaluator never enters rollout selection”被代码 `if task.hidden_success: break` 直接违背。hidden score 控制 episode termination，当然会改变 trajectory 和环境 side effects。
2. “first-attempt prequential 已足以替代 sealed holdout”不成立。它防止同一 task 的 label-before-update 泄漏，却不防止 template repetition、超参选择和人工 anchor 对适配分布的过拟合。
3. 三通道 cost ledger 在正式 v2 stream 没被调用，manifest 也没有 env/model/update token counts；matched-budget 是文字承诺。
4. policy identity 在正文含 base/adapter hash/version；manifest 只有可读模型路径、protocol、version，缺 base/adapter/tokenizer hashes。
5. AppWorld/Tau2 只在环境列表出现，v2 Results 没有运行它们，容易让读者误判证据范围。
6. request seed 把 `policy_version` 写进生产随机数。frozen 永远 v0，update arm 递增，因此 treatment 会改变后续随机数；这破坏 cross-arm common random numbers。
7. 同一 `prod_seed` 在 6 个 turn 内复用，`turn_id` 恒为 0，和公式/命名空间不符。
8. 公式在第 3 页右栏明显溢出约一整段，`purpose...` 跑出栏外，是 submission-level 排版错误。
9. “atomic commit”实现不具备 shadow state；“only then policy version advance”并不等于“only then candidate becomes served”。
10. replay 写 20–40% anchor，代码固定 40%；docstring 写 10–20%，三处口径漂移。

### 修改要求

- 将 production CRN 拆成 `exogenous_generation_seed` 与 `policy_identity`：seed 不含 version，version 单独进 manifest。
- 逐 turn 派生 seed；记录每个 request 的 seed/hash/policy identity。
- hidden evaluator只能在固定 episode 完成后离线评分，不能用于 break/retry/branch/update。
- 删除未运行环境，或增加明确表格：CTS=mechanism only，AppWorld/tau2=not evaluated。

## 第 4 页：Method 4.3/4.4 + Results 全部

### 阻断问题

1. EGC 的 “R=4 fresh snapshots + across-seed variance”在代码里是四次重新实例化同一个确定性任务，然后强制执行同一个 action；每行 U 四个数必然相同，不存在 stochastic continuation。
2. 候选中的 `real_user` 直接来自 `task.world.users`，不是 accessible observation parser；这是 privileged world-state leakage。
3. 正文说 signed credit/replay，代码只加入 positive credits；negative bars 未进入 optimizer。
4. “statistics use exact sign-flip + hierarchical bootstrap”不实：一个脚本做 200k Monte Carlo；另一个脚本路径不匹配且把 128 task outcomes 当 outer units。
5. 结果段混用两套 replicate，属于正式 paper claim audit FAIL。
6. “6 adaptation templates”不等于 6 个独立 domain/skill；它们来自少数 family 的手工变体。
7. 增益完全由 exchange 驱动；结果段虽然说 concentrate，却没有做 leave-one-template-out sensitivity，导致平均数误导。
8. EGC “moves in same direction”避开了真正对照 EGC vs naive。按方法论文标准，最重要 contrast 应是 EGC−naive，而该值为负。
9. “three mechanisms jointly necessary (ablations)”没有对应实验表、raw manifests 或图；这是无证据结果陈述。
10. “large LR 3/3 below frozen”没有与当前 v2 bundle 对应的可追溯 artifact。
11. “bottleneck is correction rather than credit”是因果诊断，但没有 oracle-credit / oracle-action / behavior-cloning upper-bound factorial experiment。
12. 页底逐 seed delta 文本溢到右栏；正文引用显示 `Fig. 2/3`，实际图号为 Figure 1/2，说明发布 PDF 没有完成足够 LaTeX passes。

### 修改要求

- 结果主表必须同时给：all-template、without-exchange、sealed-template、per-family、cost-normalized。
- primary contrast 改为 EGC vs naive；frozen vs replay 只能验证“有适配是否有效”。
- 未完成 ablation 前删除“jointly necessary”；未完成 causal upper bound 前删除“bottleneck is ...”。
- 每套 result bundle 一个不可变 run-set ID；图、统计、表只能通过同一 manifest index 生成。

## 第 5 页：两张图 + Conclusion

### 阻断问题

1. Figure 1 名为 prequential，却只画 seed-level final AUPC，不画 task index 上的 prequential curve。
2. 图中 naive 明明 7/8 高于 frozen，caption 写 6/8；这是肉眼可见的证据冲突。
3. EGC 线弱于 naive，但标题/正文没有直接标注 EGC−naive。
4. Figure 2 的三组 credit 值由脚本硬编码，不从 manifest 解析；它不是 ablation，也没有 error bar/replicate 数。
5. Conclusion 开头 “every first attempt receives the update”语义错误；first attempt 被 serving，不是“收到 update”。
6. “properties ... are decisive”由单个 toy environment 和不完整实现推出，过度概括。
7. 正文结论已经开始后，两张关键图才浮到页面顶部，结果阅读顺序较差。
8. “release all run manifests”与缺 5 个 EGC manifests 冲突。

### 修改要求

- Figure 1 改为 `task index × cumulative prequential success`，每 arm 均值+配对 bootstrap band；另加每 seed paired slopegraph/delta panel。
- Figure 2 从 raw U matrix 自动生成，展示所有 actions、R continuations、variance/reliability、是否进入 update；不要精选三例。
- 结论只保留经审计通过的最窄结论，并明确 synthetic/transductive/one-template driven。

## 第 6 页：References

### 阻断问题

1. 参考文献存在 `Cai and others`、`Wang and Hao and others`、`Zweiger and Pari...` 等不规范 BibTeX author，PDF 元数据质量不足。
2. Tau2 只引 GitHub，缺官方论文 [`arXiv:2506.07982`](https://arxiv.org/abs/2506.07982)。
3. 仓库 `CITATION_AUDIT.md` 有 24 KEEP + 1 FIX，而 JSON summary 又写 25 KEEP/PASS，内部审计不一致。
4. 只有 6 页却缺核心算法伪代码、主结果表、ablation table、limitations/ethics、artifact statement；篇幅不是瓶颈，证据组织才是。
5. `main.tex` 使用 `\usepackage[preprint]{acl}`；仓库自己的 `acl.sty` 明确投稿应使用 `[review]`。当前 PDF 无 review line numbers，不是正式匿名审稿格式。
6. GitHub 已公开且用户名直接可见；若 venue 是 double blind，提交前公开 repo/link 会泄露身份。`SUBMISSION_README` 又声称 repo private，事实不一致。

### 修改要求

- 用 DOI/Crossref/arXiv 官方 metadata 重建 BibTeX；逐条验证作者、题目、年份、venue。
- 投稿 PDF 使用当前 venue 官方 review style、行号、匿名 artifact；公开仓库延后或制作匿名 snapshot。
- 增加 limitations、broader impacts/ethics、reproducibility checklist 和 artifact manifest。

---

## 4. 逐段审稿：从主张到证据

下表覆盖 abstract、所有正文自然段、贡献 bullet 和结果小节。行号均相对于当前提交中的 `paper/sections/*.tex`。

| ID | 段落/行 | 本段功能与主张 | 严格问题 | 必须怎么改 |
|---|---|---|---|---|
| A01 | `00_abstract.tex:3–24` | 一段式概括问题、runtime、replay、EGC 与 +.070 结果 | 同一段同时包含 hybrid statistics、未实现 atomic/immutable、无效 signed replay、未证 EGC headline、无 artifact 的 A0–A2 | 拆为 problem/method/evidence/limitation；只写一个 run-set；先陈述 naive vs frozen，再明确 EGC vs naive null/negative |
| I01 | `01_introduction.tex:3–11` | 从 partial endogenous feedback 引出 Agent TTRL | 论述合理，但“later unseen tasks”暗示分布外/heldout；当前只是模板循环 | 将 unseen 改为 later tasks from an adaptation stream；sealed transfer 单独定义 |
| I02 | `01_introduction.tex:13–26` | 文献趋势与 static served-policy failure | 从自己 v1 推到“most pipelines”证据不足；96/96 是本项目审计，不是领域普遍性 | 改成“we identify a failure mode in our v1 and audit whether it can occur generally”；若要 general claim，系统审计 ≥3 frameworks |
| I03 | `01_introduction.tex:28–30` | “serving policy must provably be updated policy”总原则 | 原则正确；当前实现不满足 shadow/atomic | 用 formal invariant 明确 `served_state(t) == committed_state(t)`，给可执行 linearizability test |
| I04 | `01_introduction.tex:32–36` | runtime contribution | RNG seed 含 version造成跨 arm confound；commit 不是 atomic；A0–A2 无 Mistral artifact | 在修复 transaction 与 CRN 后再声称；当前只能叫 colocated prototype |
| I05 | `01_introduction.tex:37–40` | cross-task signed replay | naive/EGC 不训练负 rows；anchor 是人工 ground-truth demos；没有证据“fixes micro-updates” | 加 anchor-only/no-anchor/negative-row ablation；准确命名为 weighted canonical SFT replay |
| I06 | `01_introduction.tex:41–44` | EGC credit signs | 直接读 hidden world 得到 real user；R replicates 相同；Figure 2 硬编码 | 只从 serialized observation 构造 candidates；真实 continuation；raw U-to-figure pipeline |
| I07 | `01_introduction.tex:45–49` | positive transfer | 不是 EGC result；p/count 与 delta 不对应；未过 alpha .01；heldout null | 改为 transductive replay pilot，并公开 one-template sensitivity |
| R01 | `02_related_work.tex:3–11` | math TTRL 对照 stateful agents | 基本合理，但没有清楚区分 trajectory reward、action credit 和 persistent update | 加一张轴表：stateful/parameter update/credit unit/persistence/hidden labels/serving identity |
| R02 | `02_related_work.tex:13–22` | non-parametric adaptation | 声称 budget-matched baselines “include them”，事实无结果且 registry scaffold | 改为 required baselines；完成后给 cost ledger 与 official commit |
| R03 | `02_related_work.tex:24–30` | aTTT/StarOR/SEAL | 对 aTTT 的系统贡献处理太轻；“neither ... future transfer”需精确限定，不宜一概而论 | 明确本文比 aTTT 多出的可检验约束，而不是只说场景不同 |
| R04 | `02_related_work.tex:32–41` | counterfactual credit | 正确承认不是新 estimator；但 deployment-use 的新颖性没有 empirical comparison | 以 CVT-RL adapted、CausalFlow repair 为 matched controls，证明 deployment gate 的额外价值 |
| R05 | `02_related_work.tex:43–49` | PACE/VaG/SafeCommit | v2 主流没有 SafeCommit statistical gate；段落与本文结果断开 | 删除或把 commit acceptance 变成独立核心实验，含 false commit/rollback/latency |
| R06 | `02_related_work.tex:51–56` | continual self-improvement | 只用文字区分，没有任务链/长期 forgetting 曲线 | 增加 long stream、reset、backward/forward transfer、catastrophic-update 指标 |
| R07 | `02_related_work.tex:58–61` | prequential evaluation | AUPC 定义合理；但只报最终平均而非 curve，且 heldout 不能被 prequential 替代 | 画真正 prequential curve；同时保留 disjoint sealed future set |
| P01 | `03_problem.tex:3–11` | deployment stream formalization | 数学上清楚；实现却在每 turn 读取 hidden success 并 early-stop | 实现固定 horizon 或 accessible `done`；hidden scorer 离线运行 |
| P02 | `03_problem.tex:13–23` | evidence tiers | 定义很好；实现从 `world.users` 读取 privileged real user，违约 | 给每个 feature 数据血缘；禁止 env private attribute import |
| P03 | `03_problem.tex:25–37` | AUPC 与不单报 sealed holdout 的解释 | 括号中的辩护不成立：first-attempt-before-update ≠ template-level isolation | 删除辩护；把 sealed heldout 设 primary endpoint，与 AUPC 同时报 |
| P04 | `03_problem.tex:39–45` | budget与identity | v2 stream没有 canonical ledger，也缺 base/adapter hashes | 无 ledger 的 run 不进论文；manifest schema强制三通道 cap和身份闭包 |
| P05 | `03_problem.tex:47–54` | CTS/AppWorld/Tau2 | 仅 CTS-v2 有当前结果；列三环境造成 coverage impression | 标明 `mechanism environment` / `planned official evaluation`，或真正跑完后再列 |
| M01 | `04_method.tex:4–19` | request RNG 与 serving transaction | production seed treatment-dependent；同 turn seed复用；candidate先被serve；before/after同权重 | 重写为 parent/candidate双 buffer + atomic pointer swap + exogenous CRN |
| M02 | `04_method.tex:21–32` | replay row、intent balance、anchors | recency实现反向；sampler未seed；anchor比例口径漂移；anchor含真实答案 | 修 sampler/provenance；anchor来自独立 predeployment split；加 anchor-only baseline |
| M03 | `04_method.tex:34–45` | EGC G×R credit | private world leakage；R不是 continuation；仅正 credit入 buffer；主 stream绕过通用 BranchExecutorV2 | 用完整 action identity和decision snapshot；强制 common continuation seeds；训练 signed action spans |
| M04 | `04_method.tex:47–53` | evaluation/statistics/manifests | exact/hierarchical/immutable/adapter hash 四项均与 artifact不符 | 统一 artifact index；统计脚本只读该 index；无完整字段则 fail closed |
| E01 | `05_results.tex:5–13` | A0–A2 correctness | 测试默认小模型，完整依赖未声明；test本身不证明 atomic；没有Mistral log | 发布 GPU test manifest、模型/adapter hash和 fault/concurrency matrix |
| E02 | `05_results.tex:15–36` | 主结果、逐模板、heldout、EGC | evidence version混用；exchange驱动；EGC不胜naive；EGC manifests不全 | 一个 run-set 重生成全文；主表加 EGC−naive、without-exchange、heldout CI |
| E03 | `05_results.tex:38–49` | 三机制 jointly necessary | 没有对应 ablation数据；large-LR claim无当前bundle | 删除整段，直到 factorial ablation manifests 进入 release |
| E04 | `05_results.tex:51–59` | deceptive tasks与“correction bottleneck” | 只有观察性失败，不能排除 credit/update强度/anchor污染 | 做 oracle-credit、oracle-action、oracle-demonstration、larger-model 2×2诊断 |
| C01 | `06_conclusion.tex:3–13` | 重述系统与结果 | 重复 atomic/signed/EGC overclaim；“every first attempt receives update”用词错误 | 重写成 evidence-bounded contribution；不要把 serving 与 receiving update 混为一谈 |
| C02 | `06_conclusion.tex:15–25` | caveats与release | caveat较诚实，但又断言 bottleneck/decisive；all manifests不实 | 明确“hypothesis, not diagnosis”；列缺失 artifacts；不要写 complete release |

### 4.1 写作层面的共同模式

当前稿每一节都出现同一类逻辑跳跃：

```text
设计目标 / interface 名称
    → 被写成实现保证
    → 再被写成实验已验证机制
    → 最后被写成一般性结论
```

例如：函数名叫 `commit` 不等于 atomic；有 `RequestSeed` 不等于跨 arm CRN；生成了 credit 不等于 signed credit 被训练；有 heldout template 不等于 inductive future transfer。顶会稿必须把四层拆开：**specification、implementation evidence、experimental effect、claim scope**。

---

## 5. 逐图审稿

## Figure 1：`fig2_prequential.png`

### 图想证明什么

图想展示 8 个 paired seeds 下 frozen、naive、EGC 的 prequential AUPC，并支持“naive 6/8 positive，EGC same direction”。

### 实际数据与生成路径

- [`scripts/make_v2_figures.py#L21-L51`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/make_v2_figures.py#L21-L51) 从绝对路径 `/root/autodl-tmp/.../*_16.log` 读取首轮 logs。
- 首轮 naive 实际是 7/8 positive、`p=.015625`。
- caption 的 6/8、`.032` 来自 rerun/current manifests。

### 图本身的问题

1. x 轴是 seed，不是 task/time，因此它不是 prequential curve，只是 per-seed aggregate scatter/line。
2. 把不相邻 seed 用线连接没有科学含义，容易暗示趋势。
3. 没有 paired delta/CI，读者无法直接看到 uncertainty。
4. EGC 与 naive 的关键方法差没有标出。
5. 数字 annotation 过密，而 heldout/template decomposition 完全缺失。
6. 标题把 Mistral-7B 和“updates transfer”写成结论，但 transfer scope 只是 toy transductive。
7. 图中证据与 caption 冲突，属于 paper-level artifact failure。

### 应改成什么

建议一张三联图：

```text
(a) task index → cumulative first-attempt success，均值 + paired bootstrap band
(b) each seed 的 EGC−naive / naive−frozen paired delta
(c) per-template success difference，突出 exchange 与 sealed heldout
```

图必须从唯一 `run_set_manifest.json` 自动生成；caption 注入统计 JSON，而不是人工复制数字。

## Figure 2：`fig3_credit_ablation.png`

### 图想证明什么

展示 deceptive tasks 上 evidence-user credit 为正、goal-user credit 为负，从而支持 EGC 能识别正确行动。

### 实际生成路径

- [`scripts/make_v2_figures.py#L55-L70`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/make_v2_figures.py#L55-L70) 直接硬编码：
  - `[-0.3,+0.3]`
  - `[-0.5,+0.5]`
  - `[-0.3,+0.3]`
- 没有读取 log/manifest/U matrix，也没有选择规则。

### 图本身的问题

1. **这不是实验图，是手工示意图。**若没有 raw artifact mapping，不应放 Results。
2. 名为 ablation 的文件不是 ablation。
3. 三例是选择性展示，没有总样本数、错误率、ties、方差、置信区间。
4. 红绿两根 bar 没有 legend/明确 action identity；caption替代了图例。
5. R=4 全相同确定性执行，所谓 across-seed variance 并不存在。
6. 负 credit 即便计算，也没有进入 replay/training；图不能证明算法实际利用它。
7. `proposals` manifest 只记录 action name，不记录 kwargs；无法从 artifact 验证 goal-user/evidence-user对应关系。

### 应改成什么

- 对每个 decision 保存：snapshot hash、accessible evidence、full action `(name, sorted kwargs)`、action tokens、4 个 continuation seeds、`U[g,r]`、credit、variance、gate decision、update-row hash。
- 报告 credit sign accuracy、coverage、abstention、calibration、downstream causal effect。
- 图从全量 artifacts 自动生成；若只是示意，移到 Method 并明确标注 `illustration`。

## 未引用/陈旧图资产

仓库还保留 `fig1_method.png`、`fig4_safecommit.png`、`fig5_pareto.png`、`fig6_heatmap.png`，当前 PDF 未引用。问题不是有多余文件本身，而是：

- `reproduce.sh` 调用旧的 `make_figures.py`，会重写当前 `fig2/fig3`；
- 当前 v2 脚本又写死外部绝对路径；
- 因而“同一命令从 release evidence 重建当前 PDF”不成立。

应将旧论文资产移动到 `legacy/v1/`，当前 paper 只允许 manifest index 声明的图进入构建。

---

## 6. 逐实验审稿

## 6.1 v1 R002/R003/M3/M5/M6

### 当前地位

项目正确地将 v1 正式 learning-effect claim 标为 invalid：训练后的 HF/LoRA 没有成为后续 vLLM served policy，arms 还消耗不同 RNG stream。

### 可保留价值

- 是很好的 failure-forensics case study；
- 可作为 fault injection：`STATIC_SERVED_POLICY`、`RNG_SCHEDULE_DISPLACEMENT`；
- 可构造回归测试，确保 v2 不再犯同类错误。

### 不可做的事

- 不能把 v1 数字当算法负结果；
- 不能把 M4 simulation 当真实 adapter安全效果；
- 不能在 current result table 与 v2 混合。

## 6.2 v2 A0：request RNG isolation

### 声称

同一 RequestSeed bitwise identical，额外 update-arm rollouts 不改变未来 production first attempts。

### 实际覆盖

- [`test_policy_consistency.py#L36-L59`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/tests/integration/test_policy_consistency.py#L36-L59) 在同一 policy version 上重复 generation，并插入额外 branch generations。

### 缺陷

1. 测试没有模拟 commit/version bump，因此漏掉最关键 confound：production seed 包含 version。
2. 没有同时实例化 frozen/update 两 arm 做 zero-LR matched schedule。
3. generation 只恢复 CPU RNG；CUDA generator state/并发请求行为未验证。
4. test 默认 Qwen2.5-0.5B，不是论文 Mistral-7B。
5. 仓库未提供 A0 GPU artifact。

### 判定

**Component test partial；论文 end-to-end claim unsupported。**

## 6.3 v2 A1：commit changes served output

### 实际覆盖

test 先生成 `before`，然后直接调用 `train_step()` 修改同一个 serving model，最后调用 `commit()` 并再生成 `after`。

### 缺陷

- A1 证明的是“训练改了模型”，不是“commit 才使候选对 serving 可见”。
- `commit()` 内部的 `before`/`after` 都在 candidate 权重上生成，仅 version 变化；generate 又不使用对象的 current version，只使用传入 seed，两个 generation本应相同。
- `changed = output_changed OR KL>tol`，却写进字段 `deterministic_output_changed`，语义混淆。
- canary fail 时候选权重仍然留在 served model。

### 判定

**Atomicity contradicted；A1 test is mis-specified。**

## 6.4 v2 A2：rollback

### 实际覆盖

commit 后显式 `rollback()`，验证 output 回到 before。

### 缺陷

- 主 stream 在 canary fail 时只记录 violation，不调用 rollback。
- 没有 crash/retry/concurrency/partial optimizer failure测试。
- stack 只是单进程内存副本，不是可恢复 transaction log。

### 判定

**Single-process happy-path component test only。**

## 6.5 v2 frozen arm

### 观察

- 首轮/rerun 全 8 seeds 完全一致，是当前最稳定 evidence。
- 可作为 deterministic synthetic floor。

### 局限

- frozen 也初始化 6 个人工 anchors，但不训练；预算/内存/数据访问并未按 arm 明确记录。
- production seed始终 version0，不能与 update arm天然 paired。
- hidden success 控制 early stop。

### 判定

**可作为 toy baseline，但不是完整 protocol baseline。**

## 6.6 v2 naive replay arm

### 观察

- 两套 run 的 mean 都是 +.0703125；
- rerun 6/8 positive；
- sealed heldout 1/16；
- exchange 从 4/16 到16/16。

### 关键混淆

1. **人工 ground-truth anchors。**代码从 hidden task world构造每个 adaptation template 的 canonical demo；这不是 deployment feedback，而是 curated supervision。
2. **anchor-only 可能解释结果。**部分 seeds 的 final buffer 只有 6 anchors，仍相对 frozen 改善；没有 anchor-only baseline。
3. **方法不是 GRPO。**只选择 successful rollouts 做 weighted token log-likelihood；没有 old logprob、ratio clipping、reference KL、group policy objective。
4. **仅正例。**失败/负 advantage rows 被 evidence gate 排除。
5. **off-policy replay。**旧 policy rows 在新 policy 上直接逐 row REINFORCE，没有 importance correction。
6. **一个模板驱动。**exchange解释全部净正效应以上。
7. **rerun 不稳定。**6/8 seeds outcome 改变。

### 判定

**有趣的 synthetic weighted-SFT/rehearsal signal，但不能称为通用 TTRL/GRPO transfer。**

## 6.7 v2 EGC arm

### 观察

- 首轮 EGC−frozen `+.046875, p=.109375`；
- EGC−naive `−.0234375`；
- seed6 相对 frozen为负；
- 当前只有 3/8 rerun manifests。

### 实现缺陷

- privileged world state产生“正确”候选；
- R=4是确定性重复，不是 continuation；
- 仅正 credit训练；
- `BranchExecutorV2` 没被主 stream 使用；
- manifest 的 `updated` 被字典展开顺序覆盖为 false，即使 policy version 已递增；
- proposals只记 action names，无法审计kwargs/identity；
- 没有 matched random/unpaired/oracle-credit controls。

### 判定

**Proposed method not validated；当前甚至弱于 naive。标题不成立。**

## 6.8 “三机制 jointly necessary” ablation

仓库 current bundle 中没有能支撑以下三项 factorial ablation 的 runs：

- served consistency on/off；
- raw prose vs canonical action completion；
- intent balance/anchor rehearsal on/off。

无表、无 run index、无 seeds、无 CI。该段应判为 **phantom result / missing evidence**，不是“实验写得不够详细”。

## 6.9 public-environment evidence

- v2 current result 只有 CTS-v2。
- AppWorld/tau2 的旧 v1 runs 已被 served-policy audit invalidated。
- 因此当前论文没有官方 public Agent benchmark evidence。

### 判定

**simulation_only / mechanism pilot**，不能宣传多环境泛化。

## 6.10 reproducibility experiment

fresh clone 执行 README 的 `bash reproduce.sh`：

```text
=== 2/4 M4 SafeCommit stress simulation re-run (CPU, deterministic) ===
reproduce.sh: line 37: python: command not found
exit code: 127
```

即使修成 `python3`，脚本仍：

- 只检查旧 M2–M6 manifest 存在，不校验 v2完整性/hash；
- 运行 v1 M4；
- 调用旧 `make_figures.py`，不是 v2 figure generator；
- 只执行一次 pdflatex，导致引用编号可能 stale；
- 不重跑任何主 GPU结果。

### 判定

**NOT REPRODUCIBLE from documented command。**

---

## 7. 深入代码仓库的逐模块审查

## 7.1 `runtime/served_policy.py`：名称是 transaction，实际是 in-place training

### P0-1：candidate 在 commit 前已经对 serving 可见

[`train_step()` L100–123](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/src/agent_ttrl/runtime/served_policy.py#L100-L123) 直接对 `self.model` 做 `opt.step()`；`generate()` 也使用同一个 `self.model`。因此：

```text
parent served model
    └─ train_step(inplace) → candidate already served
                              └─ commit(canary after exposure)
```

真正 atomic 应是：

```text
served_parent (read-only)
    ├─ generate production
    └─ clone/shadow_candidate → train → validate → atomic pointer swap
                                         └─ fail: discard candidate
```

### P0-2：canary 的 parent/candidate generation 比较无效

[`commit()` L150–172](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/src/agent_ttrl/runtime/served_policy.py#L150-L172)：

- KL 用临时参数替换比较 parent/candidate，这部分至少意图正确；
- 但 `before` 和 `after` generation 都在 candidate 权重上运行；
- 两者之间只递增 `self.policy_version`，`generate()` 不读取该字段，而读取传入的同一个 `canary_seed`；
- `changed = before != after or kl > tol`，然后把 `changed` 填入 `deterministic_output_changed`。即使 output 未变，只要 KL 大也写 true。

应分别在 parent shadow 和 candidate shadow 上，用完全相同的 exogenous seed 生成，并分别记录 `output_changed` 与 `kl_pass`。

### P0-3：canary fail 没有恢复 parent

hash mismatch/no observable change 分支只 return `False`；训练造成的 candidate weights 不会恢复。主流也不调用 `rollback()`。这违反 fail-closed。

### P1-1：optimizer 每 row 重建

每次 `train_step()` 创建新的 AdamW，动量状态无法跨 row/batch保留；论文所说“一次 batch update”实际是多个顺序敏感的单样本首步 AdamW。应：

- 一次 update batch 构造一个 optimizer step；
- 明确 gradient accumulation；
- 记录 optimizer state hash；
- 候选失败时连 optimizer state 一起回滚。

### P1-2：不是 GRPO/PPO objective

当前 loss 是 `-advantage * mean(token_logp)`；没有：

- old-policy logprob；
- importance ratio；
- clipping；
- reference KL；
- group objective的一致采样定义；
- off-policy correction。

如果不重写 objective，论文/简历应诚实称为 `advantage-weighted canonical SFT replay`，不要叫 GRPO。

### P1-3：RNG 保存不完整

`torch.random.get_rng_state()`/`set_rng_state()`只覆盖默认 CPU RNG。模型在 CUDA 上生成时需显式 `torch.Generator(device)` 或保存/恢复 `torch.cuda.get_rng_state_all()`；并发时全局 state set/restore 也不是线程安全的。

## 7.2 `runtime/request_seed.py`：purpose isolation 正确，cross-arm pairing 错误

[`RequestSeed.seed()` L48–54](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/src/agent_ttrl/runtime/request_seed.py#L48-L54) 把 `policy_version` 放入 seed。对同一 task：

```text
frozen: version=0 → seed A
updated: version=3 → seed B
```

这让 treatment 同时改变 policy 和 sampling noise。对因果比较应使用：

```python
generation_seed = H(protocol, stream_seed, task_id, turn_id, purpose)
policy_identity = (base_hash, adapter_hash, policy_version)  # 只记录，不进跨臂 CRN
```

若希望同一 policy_version 内可重现，可额外保存 request identity，但 primary cross-arm seed必须 exogenous。

## 7.3 `scripts/cts_v2_stream.py`：主实验控制流

### P0-4：hidden evaluator 控制 rollout

[`L173–180`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/cts_v2_stream.py#L173-L180) 在每轮后读取 `task.hidden_success` 并 break。即便不进入 loss，它仍进入 trajectory control，违反 paper/contract 的“reporting only”。

修复方式：固定 horizon；或环境返回属于 accessible protocol 的 `done`；hidden scorer只在 episode artifact seal 后离线调用。

### P0-5：privileged state leakage 构造 EGC candidates

[`L198–211`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/cts_v2_stream.py#L198-L211) 读取：

```python
g = task.world._goal
real = next(iter(task.world.users))
```

`real` 不是从 agent可见的 `lookup_order` observation 解析，而是直接读 hidden simulator state。随后它被用于候选 action 与训练 completion。这样“credit找到了 real user”并不是算法发现，而是 evaluator给了答案。

必须引入严格 view object：算法只接收 serialized public observation；private world API 在 adaptation process 中禁止 import/访问，并加 taint test。

### P0-6：R=4 不是 continuation replicates

[`L227–235`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/cts_v2_stream.py#L227-L235) 每个 r：

- 用完全相同 seed instantiate相同 task；
- 直接执行同一 forced action；
- 不运行 continuation policy；
- 没使用 `continuation_id`。

于是每个 action 的四个 utility 是重复值。“across-seed variance”是零，不足以估计 reliability。

### P0-7：production turn seed未更新

[`L142–155`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/cts_v2_stream.py#L142-L155) 在 turn loop 外构造 seed，turn_id恒为0；每轮新 prompt 用同一 seed。应在循环内用真实 turn 派生。

### P0-8：EGC manifest 的 `updated` 被覆盖

[`L282–285`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/cts_v2_stream.py#L282-L285)：

```python
update_info = {
  "updated": res.passed,
  ...,
  **update_info,
}
```

旧 `update_info={"updated": False, ...}` 在最后展开，覆盖新值。实际 manifest 可见 `canary:"ok"`、`policy_version:1`，却写 `updated:false`。这直接破坏 artifact truth。

### P1-4：人工 anchors 是强监督混淆

[`L91–125`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/cts_v2_stream.py#L91-L125) 从 hidden task state构造每个 adaptation template 的 canonical demonstration，甚至对 deceptive variant 注入 real user/verification pattern。这相当于预先告诉算法目标技能。

至少需要以下 arms：

- frozen；
- anchor-only，无 deployment evidence；
- shuffled/wrong anchor negative control；
- replay-only，无 anchor；
- anchor + naive；
- anchor + EGC。

anchors 必须来自 predeployment disjoint split，并计入 update-token/data budget。

### P1-5：positive-first 截断丢负样本

[`L267–275`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/cts_v2_stream.py#L267-L275) 和 naive 同类代码将 `pos+neg` 后截取一半 batch。即使 buffer 有负 rows，正 rows 会优先占满；当前 EGC 又只 add positive。应预注册正负比例并随机/分层采样。

### P1-6：naive 名称和 reward 语义

naive 只在 `final_ok==1` 且其他 gate通过时加入 row；reward 是手工 `0.6*ok_call_fraction + 0.4*final_ok`。它是成功轨迹过滤/weighted SFT，不是 terminal trajectory RL baseline。命名必须反映算法。

### P1-7：manifest 可覆盖且字段不足

[`L375–383`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/scripts/cts_v2_stream.py#L375-L383) 固定写 `OUT/run_manifest.json`，没有 `O_EXCL`/content-addressed path/no-overwrite。还缺：

- git commit与dirty status；
- base/model/tokenizer/chat-template hashes；
- adapter parent/candidate/committed hashes；
- per-request seed/token ids/logprobs；
- optimizer/config/dependency/GPU/container；
- cost ledger；
- U matrix/action kwargs/snapshot；
- RNG states与replay batch hashes。

## 7.4 `optimization/replay_buffer.py`

### P1-8：recency 方向写反

[`L53`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/src/agent_ttrl/optimization/replay_buffer.py#L53)：

```python
row.weight = gamma ** len(self.rows)
```

越晚加入，指数越大、权重越小；已有老 row 不随时间衰减。与“older rows decay”相反。

### P1-9：未 seed 的 Python random

[`L72–87`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/src/agent_ttrl/optimization/replay_buffer.py#L72-L87) 使用模块级 `random.sample/choices`，没有注入 run RNG，也不记录 sampled row IDs。这与 rerun不稳定直接一致。

### P1-10：dedup/provenance 弱

hash只含：intent、prompt前64 tokens、completion前64 tokens、rounded advantage；不含 task ID、producer policy version、完整token序列。不同 row可能碰撞/错误去重。`_seen` 又不在 eviction 后删除，旧 row永远不能重新加入。

### P1-11：sampler 不保证固定 batch和真正 intent balance

- `random.choices` 有放回，可重复同一 row；
- 每 intent数量不足时不回填；
- batch可能远小于 n；
- anchors 已放进 `rows` 又单独保存，身份判断为对象比较，逻辑脆弱。

## 7.5 `credit/branch_executor_v2.py`

主 stream 没使用该 class，只 import `paired_credit_v2`；因此它是 dead/parallel implementation，不能作为正式流正确性的证据。

即便单独看：

- [`L73–75`](https://github.com/hxm2023/agent-ttrl/blob/c6dddfecd20ebe33de7a4cc5886d80e218896002/src/agent_ttrl/credit/branch_executor_v2.py#L73-L75) 只按 action name 去重，`refund(goal_user)` 与 `refund(real_user)`会被合并，正好丢掉论文核心反事实。
- `action_tokens=[]`，无法把 credit绑定到原始 action token span。
- reliability gate用 actions间 credit std，不用每个 action across-continuation uncertainty。
- `rows`依据 mean utility非零，而不是 credit reliability；返回的是 `BranchRecord`，不是训练 `EvidenceRow/UpdateRow`。
- `snapshot_after`被写成与 before 相同，没有真实 post-state hash。

建议只保留一个 canonical branch executor，主实验不得手写第二套语义。

## 7.6 统计脚本

### `stats_v2_cts.py`

- 文件名/输出称 “exact”，实际随机抽 200,000次符号；n=8完全可以穷举256种，无需 Monte Carlo。
- 读取首轮 `_16.log`，所以会输出 `p≈.016, 7/8`，与论文 `.032, 6/8` 不同。
- 没有CI、multiple endpoint声明、run-set identity。

### `stats_v2.py`

- 期待 `frozen_s0/run_manifest.json`，仓库实际是 `frozen_s0_manifest.json`，默认运行找不到数据。
- 将128个 task outcome当作“outer deltas”，发生 pseudoreplication；真正 outer unit 是 seed/session。
- hierarchical bootstrap没有先重采样 stream/seed，而按解析出的 template采一个 task。
- docstring声称“resample streams then families”，实现并非如此。

正确方案：唯一脚本先按 run-set manifest加载 8 个 session summary；primary exact test在8个 paired session delta上穷举；hierarchical bootstrap先采 session，再在session内按预注册 family/task结构采样。

## 7.7 `reproduce.sh`、依赖与测试

### 依赖闭包失败

`pyproject.toml` 只声明 pydantic/jsonschema/numpy；runtime和integration test依赖 torch、transformers、peft，画图依赖 matplotlib，均没有可安装 extra。

fresh environment：

```text
uv run --extra dev pytest -q
→ ModuleNotFoundError: No module named 'torch'
→ collection interrupted
```

排除 integration 才得到：

```text
144 passed in 40.96s
```

因此 README 的 `pytest # 127+ tests` 不能视为完整测试通过。

### integration test 设计不足

- marker `integration`未注册；
- 无 GPU/model availability skip；
- 默认模型与论文模型不同；
- 无日志/manifest；
- 无 zero-LR version-bump control、fail rollback、concurrent generation、crash recovery、hash mismatch等关键案例。

### PDF build 不闭合

- `reproduce.sh`只跑一次 pdflatex；当前 PDF正文出现旧 Figure 2/3引用，而实际图号1/2。
- 第3页seed公式跨栏；第4页delta文字跨栏。
- `SUBMISSION_README`说7页，当前PDF是6页；说 `p=.016,7/8`，当前论文写`.032,6/8`；title也不同。

## 7.8 文档与治理一致性

| 文件 | 当前说法 | 冲突 |
|---|---|---|
| `README.md` | v2 under construction；main results invalidated | `SUBMISSION_README`又称paper finalized |
| `SUBMISSION_README.md` | 7pp、p=.016、7/8、repo private | PDF 6pp、正文p=.032/6/8、repo public |
| `M0_IMPLEMENTATION_PROFILE.yaml` | `profile_status:FROZEN` | driver/tau2 SHA仍是 REQUIRED placeholders |
| `baseline_registry.yaml` | `status:SCAFFOLD` | paper声称budget-matched baselines included |
| `CITATION_AUDIT.md/.json` | 24 KEEP+1 FIX vs summary 25 KEEP/PASS | audit verdict内部不一致 |

建议把状态生成化：一个 release manifest自动渲染 README、paper macros、artifact table，禁止手工复制状态/数字。

---

## 8. 论文主张—证据审计矩阵

| ID | 论文主张 | 证据状态 | 审计结果 | 最窄可保留措辞 |
|---|---|---|---|---|
| C01 | 16 tasks、8 seeds、Mistral-7B | logs/manifests | SUPPORTED for CTS pilot | 同左，但注明 synthetic |
| C02 | frozen .4531 vs naive .5234，mean +.070 | 两套数据均同均值 | SUPPORTED NUMERICALLY | 指定唯一 run-set |
| C03 | 列出八个 delta 且 p=.032 | delta来自首轮，p来自rerun | **HYBRID / FAIL** | 二选一整套报告 |
| C04 | 6/8 positive、0 negative | rerun是6/8；首轮7/8 | HYBRID | 指定rerun ID |
| C05 | A0–A2 on Mistral-7B | 只有test source，无release log | MISSING EVIDENCE | “integration tests are implemented” |
| C06 | atomic commit | in-place train before canary | **CONTRADICTED** | “colocated in-place update prototype” |
| C07 | request-level RNG pairing | purpose隔离，但version进入seed | PARTIAL/CONFOUNDED | “request-scoped RNG namespaces” |
| C08 | signed replay | 只add positive；positive-first截断 | **CONTRADICTED** | “positive successful-row replay” |
| C09 | intent-balanced replay | sampler近似、batch不固定 | PARTIAL | “intent-bucketed sampler” |
| C10 | hidden evaluator reporting only | hidden_success early stop | **CONTRADICTED** | 删除，直到修复 |
| C11 | three-channel matched budget | 无ledger字段/调用 | UNSUPPORTED | 删除 |
| C12 | policy identity绑定 | manifest缺base/adapter hash | UNSUPPORTED end-to-end | schema层有设计 |
| C13 | R=4 fresh counterfactual continuations | 四次相同deterministic action | **CONTRADICTED** | “repeated forced-action evaluation” |
| C14 | across-seed variance reliability | U行无随机变化 | CONTRADICTED | 删除 |
| C15 | credit signs correct | 候选含hidden real user；图硬编码 | INVALID PROVENANCE | “toy oracle-assisted sign smoke” |
| C16 | exchange 4/16→16/16等逐模板数 | rerun manifests | SUPPORTED for rerun | 明确run-set和template-driven |
| C17 | heldout 0/16→1/16 | rerun manifests | SUPPORTED / NULL | “no meaningful heldout evidence” |
| C18 | EGC +.047,p=.11,6/8 | 首轮logs | NUMERICALLY SUPPORTED | 加1 negative和vs naive |
| C19 | EGC method improves | EGC<naive | **NOT SUPPORTED** | 不可保留 |
| C20 | three mechanisms jointly necessary | 无ablation artifacts | **PHANTOM RESULT** | 删除 |
| C21 | large-LR harms 3/3 | current v2 bundle无对应run | MISSING | 提供artifact或删除 |
| C22 | exact sign-flip + hierarchical bootstrap |脚本/论文不闭合 | CONTRADICTED | 重新实现并只读唯一bundle |
| C23 | immutable manifests | fixed-path overwrite | CONTRADICTED | “checked-in snapshots” |
| C24 | all run manifests released | EGC 3/8 | CONTRADICTED | “partial rerun manifests” |
| C25 | public environments in protocol | current v2无official结果 | DESIGN ONLY | CTS-only result |
| C26 | inductive future transfer | sealed probe null | **NOT SUPPORTED** | transductive only |

总体 `PAPER_CLAIM_AUDIT = FAIL`。失败原因不是少量 rounding，而是 C03/C04 的证据版本混用和多项 implementation/claim contradiction。

---

## 9. 与 2026 年前沿工作的差距：新颖性不是靠换场景自动成立

以下比较基于当前可核对的原始论文页，而不是二手介绍：

- [aTTT: No Time Like the Present](https://arxiv.org/abs/2607.03441)：live/in-episode LoRA test-time training、runtime LoRA serving、ALFWorld 与 SWE-bench Lite 等真实任务证据。
- [CVT-RL](https://arxiv.org/abs/2606.05263)：policy-conditioned counterfactual contribution、validity gating、doubly robust estimator与多 agent environments。
- [PACE](https://arxiv.org/abs/2606.08106)：paired anytime-valid candidate acceptance。
- [StarOR](https://arxiv.org/abs/2606.15197)：MCTS sibling branches + transient LoRA GRPO，在多优化建模 benchmark上验证。
- [SEA-Eval](https://arxiv.org/abs/2604.08988) 与 [EvoTest](https://arxiv.org/abs/2510.13220)：sequential/self-evolving agent evaluation。
- [Tau2 official paper](https://arxiv.org/abs/2506.07982)：dual-control tool-agent-user benchmark定义。

### 9.1 capability matrix

| 工作 | 参数更新 | 时序 | counterfactual credit | serving一致性 | 安全commit | public agent evidence | 当前稿相对差异 |
|---|---|---|---|---|---|---|---|
| aTTT | LoRA | in-episode live | 否/非核心 | 是，runtime LoRA | 非本文式 | 有 | 本文想增加partial-evidence credit与session transfer，但尚无有效证明 |
| CVT-RL | RL training | trajectory/step | 强 | 非本文重点 | validity gate | 有 | 本文 EGC 更轻量/部署期，但 estimator 与实证均更弱 |
| PACE | candidate evolution | sequential | 否 | 可结合 | anytime-valid | 有相关场景 | 本文只写commit叙事，v2未运行统计gate |
| StarOR | transient LoRA GRPO | per instance | tree siblings | 是 | 非核心 | 5类优化任务 | 本文想跨task persistent transfer，但heldout为null |
| 本文 current | in-place LoRA weighted SFT | session | oracle-assisted deterministic branch | colocated但非atomic | 未进入v2结果 | CTS toy only | 组合idea有潜力，当前 evidence不足 |

### 9.2 真正可投稿的 novelty 需要满足什么

仅仅说“别人做 math，我们做 agent”不够。至少要建立一个可检验的新难点：

1. **Partial-evidence identification**：证明 terminal reward不足，EGC在存在 deceptive/partial tool evidence 时能更准确定位 action credit。
2. **Persistent deployment commitment**：更新会影响未来未知任务，因此需要 transaction、rollback与negative-transfer control；不是 per-instance临时LoRA。
3. **Inductive future transfer**：不是重复模板memorization，而是跨 template/family/app/domain 的 sealed improvement。
4. **Matched cost advantage**：EGC相对 naive、aTTT、repair/SFT、context memory，在相同 env/model/update tokens下更好。
5. **可审计 runtime**：每一个从 evidence到gradient到commit到next request的身份链闭合。

当前五项没有一项形成完整“方法→实现→实验→统计”闭环。

### 9.3 三种可行论文定位

#### 路线 A：算法主会稿（风险最高、上限最高）

题目核心：`Evidence-Gated Counterfactual Credit Enables Inductive Deployment-Time RL for Tool Agents`。

必须有：EGC > naive/aTTT/CVT-style controls，在 ≥2 official environments、≥2 model families、sealed future splits、matched budgets上稳定成立。

#### 路线 B：系统/可靠性稿（当前资产更接近）

题目核心：`Transactional and Paired Evaluation Contracts for Agent Test-Time Learning`。

贡献变为：

- fault taxonomy：static served policy、treatment-dependent RNG、hidden-control leakage、non-atomic commit、artifact version mixing；
- 对多个开源 TTRL/agent adaptation pipelines 做 audit/fault injection；
- transactional adapter runtime + no-overwrite evidence chain；
- 证明这些 failure 能制造多大假增益/假负结果。

这条路线不要求 EGC 立刻赢，但要求 multi-framework、multi-backend evidence，不能只审自己一个仓库。

#### 路线 C：benchmark/position + empirical study

题目核心：`When Does Agent Test-Time Adaptation Transfer? A Prequential Benchmark with Negative Controls`。

重点是严格 split、anchor controls、template/family decomposition、public tasks与长期stream。需要足够广的 method suite，避免变成只报告自己 toy environment。

### 推荐

短期最稳是 **B 线作为工程核心，A 线作为条件性算法升级**。等 EGC primary contrast真正为正，再把算法放回标题；不要反过来用标题逼结果。

---

## 10. 顶会救援：按 kill gate 而不是按“继续堆实验”推进

## Phase 0：立即冻结 current result（1 天）

### 动作

- 给提交 `c6dddf` 打 `audit-invalid-for-publication` tag/notes；不要删除 logs。
- 写一个唯一 `RUN_SET_INDEX.json`，分别标出 first与rerun，不再混用。
- paper 中所有当前数字标 `exploratory / invalid pending runtime audit`。
- 自动检查：一个 figure/table/stat只能引用一个 run-set hash。

### Gate P0-A

如果不能从一个不可变 index 一键重建所有数字，**禁止跑新 GPU实验**。

## Phase 1：重写 serving transaction（3–5 天）

### 正确状态机

```text
COMMITTED_PARENT
    ↓ clone adapter + optimizer state
SHADOW_TRAINING
    ↓ freeze candidate hash
SHADOW_VALIDATION
    ├─ fail → DISCARD_CANDIDATE → COMMITTED_PARENT
    └─ pass → atomic compare-and-swap(version,parent_hash,candidate_hash)
                  ↓
             COMMITTED_CANDIDATE
```

### 必须实现

- parent/candidate分别存储；production只能读 committed pointer。
- canary在 parent与candidate上使用同一 exogenous seeds。
- commit失败自动丢弃candidate；任何异常恢复parent/optimizer。
- compare-and-swap 防并发 stale commit。
- manifest记录 parent/candidate/committed hashes与线性化点。

### 必须通过的 fault tests

| Test | 预期 |
|---|---|
| zero-LR + version bump | update arm 与 frozen future outputs bitwise equal |
| candidate train before commit | production仍读parent |
| hash mismatch | fail closed + parent保持 |
| canary exception/OOM | parent保持，candidate quarantined |
| concurrent generate during commit | 每请求只看到完整parent或完整candidate |
| double commit/retry | exactly-one authoritative commit |
| rollback after process restart | 从durable journal恢复parent |

### Gate P0-B

任何一项失败，不能使用“atomic/transactional/served-policy consistent”措辞。

## Phase 2：修复实验识别（2–4 天）

### RNG

- production seed移除policy_version；每turn单独seed。
- Python/NumPy/torch CPU/CUDA/generator全链seed并入manifest。
- replay sampler接收显式 RNG，保存batch row hashes和顺序。

### hidden/evidence boundary

- private world与algorithm process隔离；只传public observation schema。
- hidden scorer固定episode结束后离线运行。
- taint test：任何 private字段进入candidate/update时自动FAIL。

### replay

- 修正recency；完整content hash；eviction后正确维护seen。
- optimizer按batch持久化；明确是否off-policy及correction。
- 如果不实现PPO/GRPO，正式重命名为 weighted SFT。

### Gate P0-C

zero-LR、anchor-only、hidden-taint、deterministic replay四个controls必须先绿。

## Phase 3：让 EGC 真正成为 EGC（4–7 天）

### decision artifact

每个branch group必须保存：

```json
{
  "decision_state_sha256": "...",
  "accessible_observation_sha256": "...",
  "parent_policy": {"base": "...", "adapter": "...", "version": 3},
  "actions": [
    {"name": "refund_order", "kwargs": {"order_id": "...", "user_id": "..."},
     "token_span": [42, 57], "action_sha256": "..."}
  ],
  "continuation_seeds": [101, 102, 103, 104],
  "U": [[...], [...]],
  "credits": [...],
  "variance": [...],
  "gate": [...],
  "update_row_sha256": [...]
}
```

### 算法要求

- candidate只能来自goal text、accessible observations和parent policy proposals。
- full `(action_name, kwargs)` identity去重，不按name。
- 从decision snapshot强制action后，运行同一 continuation policy；common random numbers跨 actions共享。
- 同时使用positive与negative action spans，或通过 preference objective使用pair。
- reliability基于 action-specific replicate uncertainty；确定环境应引入stochastic user/continuation，而不是复制相同值。

### 必须基线

- terminal replay；
- random credit；
- unpaired continuation；
- oracle credit（dev upper bound）；
- CausalFlow-style repaired demonstration；
- CVT-RL adapted estimator；
- equal-extra-rollout no-update。

### Gate P1-A

若在修复后的 CTS 上 EGC 不优于 naive，**删除 EGC 标题**；不要继续用更大模型掩盖机制失败。

## Phase 4：拆掉 anchor confound（2–3 天 + GPU）

必须跑 factorial：

| Arm | anchors | deployment rows | EGC |
|---|---:|---:|---:|
| frozen | no train | no | no |
| anchor-only | yes | no | no |
| naive-no-anchor | no | yes | no |
| naive-anchor | yes | yes | no |
| EGC-no-anchor | no | yes | yes |
| EGC-anchor | yes | yes | yes |
| shuffled-anchor | wrong/mismatched | yes | no/yes |

并计入 anchor data/token cost。若 anchor-only复现大部分 `+.070`，论文必须将贡献改为 rehearsal/canonical SFT，而不是 TTRL credit。

## Phase 5：正式 benchmark 矩阵（2–4 周）

### 最低可投稿矩阵

| 轴 | 最低要求 |
|---|---|
| environments | CTS mechanism + AppWorld + Tau2；后两者使用官方 evaluator/loop |
| models | Mistral-7B + Qwen3-4B/8B，至少两模型家族 |
| seeds | power analysis后至少8，建议10–12 session seeds |
| stream length | 能观察learning curve，不能只有16 tasks；建议按环境做64–200 task sessions |
| methods | frozen、context/memory、anchor-only、naive weighted SFT、aTTT、EGC、CVT/CausalFlow control |
| endpoints | AUPC、sealed future、negative transfer、catastrophic update、cost、latency |
| budgets | env calls、model tokens、update tokens三通道，所有arms硬匹配 |
| artifacts | immutable raw request/trajectory/update/commit events + hashes |

### split原则

- adaptation templates与sealed future在 template/family/app/domain 至少一个层次 disjoint。
- hyperparameter/calibration只用dev；sealed一次性开启。
- 不允许由代码中手写 `held_out="..."` 后反复查看结果。

### primary claim

应该预注册为：

> 在两个 official environments 上，EGC 相对 matched naive/aTTT baseline 的 paired session-level AUPC 差异，其 99% CI 下界均大于0；sealed future同方向且无预注册 catastrophic-update劣化。

而不是“任意一个 toy average p<.05”。

## Phase 6：统计与报告（2–3 天）

- n=8时精确枚举256个sign patterns，不要Monte Carlo冒充exact。
- outer unit=独立session seed；先stream后family/task做hierarchical bootstrap。
- 使用预注册 `alpha=.01` 或正式修改protocol并解释，不能看到结果后改门槛。
- 报 point estimate、99% CI、exact p、positive/zero/negative、per-family heterogeneity。
- 主结果必须给 leave-one-template/family-out sensitivity。
- 多环境主张需预注册合取/多重校正。
- negative transfer 和 catastrophic update进入主表，不放脚注。

## Phase 7：paper重写（3–5 天）

推荐结构：

1. Problem：deployment-period persistent adaptation为何与普通TTRL不同。
2. Failure audit：served-policy与paired-evaluation不变量。
3. Method：EGC estimator + transactional runtime，各自清楚形式化。
4. Experimental contract：evidence tiers、cost、split、outer units。
5. Main official results。
6. Mechanism：CTS exact oracle、credit fidelity、anchor factorial。
7. Reliability/cost：rollback、latency、GPU memory、throughput。
8. Limitations/ethics。

图表至少包括：method/data-lineage图、主结果表、真实prequential curves、EGC credit fidelity/calibration、ablation、cost/reliability。

---

## 11. 明确的 Go / Kill 标准

| Gate | Go | Kill/Pivot |
|---|---|---|
| Runtime | 全部 transaction/RNG/hidden-boundary fault tests通过 | 任一zero-LR/atomic test失败：禁止效果实验 |
| Anchor identification | EGC/naive增益显著超过anchor-only | anchor-only解释≥80%增益：pivot到rehearsal SFT |
| EGC mechanism | credit fidelity高于controls，EGC>naive | controlled CTS + 一个public pilot均不优：删除EGC headline |
| Future transfer | sealed family/template正向且CI过门槛 | sealed持续null：只做transductive claim |
| External validity | ≥2 official env、≥2 models同方向 | 只有CTS正：workshop/engineering artifact，不是算法主会 |
| Reproducibility | clean clone一键重建tests/stats/figures/PDF | 任一数字需手抄/绝对路径：不提交 |

这些 kill conditions 能避免再出现“先定标题、再找某个统计口径支持标题”的路径依赖。

---

## 12. 顶会审稿人可能给出的最强 rejection memo

> The paper studies an important problem, but the proposed EGC method is not supported by the main result. The only nominally significant comparison is naive replay versus a frozen policy; EGC underperforms naive replay. Moreover, the reported deltas/figure and the p-value/template counts come from different reruns. The gain is entirely driven by one repeated adaptation template and does not transfer to the sealed template. Code inspection further shows that training mutates the serving model before the claimed atomic commit, production seeds depend on treatment-induced policy versions, hidden simulator state constructs EGC candidates, and the four “continuations” are deterministic duplicates. The ablation claims have no released evidence, and the documented reproduction/test commands fail in a fresh environment. These issues invalidate the causal, systems, and algorithmic claims rather than merely limiting scale. I recommend rejection and a complete rerun after repairing identification and artifact provenance.

以当前 evidence，rebuttal无法靠文字解决；必须补新实现和新实验。

---

## 13. 是否能作为大模型后训练求职简历核心项目

## 13.1 当前状态：可以作为“研究工程审计项目”，不能作为“已证明算法项目”

### 目前能证明的能力

- 发现训练/serving policy断链并撤回无效结果；
- 设计 request-scoped RNG、LoRA runtime、prequential protocol、artifact schemas；
- 对 replay、credit、commit、统计和复现做跨层排障；
- 建立 CTS synthetic environment与144个CPU tests；
- 对负结果和heldout null保持诚实。

### 目前不能证明的能力

- EGC优于naive；
- GRPO正确实现；
- atomic commit/rollback；
- official AppWorld/Tau2效果；
- inductive future transfer；
- `+.070, p=.032` 是单一一致证据集上的可复现结果。

## 13.2 当前可用的诚实简历写法

> **Agent-TTRL：Agent部署期后训练审计与原型系统**  
> 审计并定位 v1 在线 LoRA 实验中的静态 served-policy 与 RNG schedule confound，撤回无效 learning claim；搭建 colocated PEFT generation/update、request-scoped RNG、跨任务 replay、CTS-v2 与可复核 run artifacts，并构建 144 个 CPU 组件测试。当前 synthetic pilot 显示 transductive/template-specific signal，sealed transfer 与 EGC 相对 naive 的效果仍在严格审计中。

面试时要主动说：

1. 最初看到了什么“结果”；
2. 哪个 negative control 暴露它不是policy update；
3. 如何追到 trainer/serving/RNG identity；
4. 为什么 v2 current 仍不够 atomic/causal；
5. 接下来用什么 gate 防止再次自欺。

这是非常强的工程判断故事，比硬吹一个不可信p值更有说服力。

## 13.3 当前禁止写法

- “提出 EGC-TTRL，在 Mistral-7B 上显著提升 AUPC +7.0%，p=.032。”
- “实现原子 adapter commit 和自动 rollback。”
- “在 AppWorld/Tau2 上验证。”
- “实现 GRPO online learning framework。”
- “证明 inductive future transfer。”
- “论文已达到 ACL 2027 submission-ready。”

这些句子均会被代码追问击穿。

## 13.4 修复后成为核心项目需要的简历证据

至少补齐：

- transaction fault matrix 全绿；
- clean-clone reproducibility；
- ≥2 official environments / ≥2 models；
- EGC vs strongest matched baseline 的真实主结果；
- sealed transfer与negative-transfer结果；
- throughput、GPU memory、commit latency、rollback recovery time；
- GitHub release/CI/container/model revisions/immutable artifacts。

修复后的写法应基于真实数据模板，而不是预填数字：

> 设计并实现 transactional session-LoRA runtime，将训练候选与 serving parent 隔离，通过 `[N]` 类 fault-injection tests 验证 atomic commit/rollback；在 `[envs]×[models]×[seeds]` 的预注册 prequential evaluation 中，EGC 相对 `[baseline]` 提升 `[effect, CI]`，同时将 catastrophic-update rate 控制在 `[value]`，全部结果由 content-addressed manifests 一键复现。

只有方括号被真实 artifact填满后才写进简历。

---

## 14. 按优先级排序的修改清单

### P0：不修就不能再解释任何效果

- [ ] 统一并冻结 first/rerun run-set，消除hybrid paper evidence。
- [ ] production seed移除policy_version，逐turn seed。
- [ ] hidden evaluator退出control flow。
- [ ] shadow candidate + atomic pointer swap + failure rollback。
- [ ] EGC不再读取private world state。
- [ ] replay RNG/optimizer deterministic且可记录。
- [ ] manifest no-overwrite + 完整identity/cost/provenance。
- [ ] 删除或重跑所有phantom ablation claims。

### P1：决定论文方法是否活着

- [ ] 真正G×R stochastic continuation与full action identity。
- [ ] signed action-span update或preference objective。
- [ ] anchor-only/no-anchor/shuffled-anchor factorial。
- [ ] EGC vs naive/aTTT/CVT/CausalFlow matched controls。
- [ ] one-template-out与sealed family transfer。

### P2：决定是否够主会

- [ ] AppWorld + Tau2官方loop/evaluator。
- [ ] 两模型家族、power-based seeds、长stream。
- [ ] 三通道cost和系统性能。
- [ ] exact/hierarchical统计、99%CI、multiple endpoints。
- [ ] anonymous artifact、官方review style、完整BibTeX。

### P3：展示与工程质量

- [ ] 单一`make all`/container从artifact重建paper。
- [ ] CI区分CPU unit与GPU integration，缺依赖时明确skip/fail。
- [ ] 图表全部数据驱动，禁止hard-coded result values。
- [ ] README、paper、release状态由同一manifest生成。

---

## 15. 最终结论

这个项目**现在还没有达到顶会论文要求**，差距不是“再补几个seed”或“润色写作”，而是需要一次方法识别、runtime transaction和artifact provenance的系统重建。当前最重要的科学事实是：

1. naive replay 的平均正数存在，但两套rerun被混用；
2. 正数完全由一个重复adaptation模板驱动，sealed probe近乎零；
3. EGC不优于naive；
4. current实现仍破坏atomic、cross-arm RNG、evidence boundary与counterfactual continuation；
5. 因此不能把 `+.070` 解释为EGC或inductive Agent-TTRL成功。

但项目仍有成为强核心项目的潜力。最有价值的主线不是继续包装当前结果，而是把“**如何避免在线后训练被 serving、RNG、hidden evidence 和 artifact drift制造假效果**”做成可执行契约，再用严格对照决定 EGC 是否存活。完成 P0 后，这会是一项很强的后训练工程作品；完成 P1–P2 并得到跨official environments的正结果后，才有资格恢复算法顶会目标。

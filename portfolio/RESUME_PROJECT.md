# Agent-TTRL — 简历核心项目材料（2026-08-25 v3 定稿：审计弧线版 + tau2 正结果）

> 状态：完整三阶段审计弧线 + 双 backbone 复现 + 更新规则穷尽扫描 + 机制发现
> + 可复现审计链 harness + **官方 tau2 环境正对照（task 0/2 满分 1.0）**
> + Route B 论文（8 内容页 + 参考文献，ACL 格式，可投稿）。
> 核心卖点：**系统性发现并修复了 Agent 推理时强化学习中最隐蔽的失败模式，
> 证明常见正结果来自协议泄漏；穷尽扫描 10+ 更新规则确认干净协议下无迁移；
> 并给出官方基准上的端到端正结果作为对照 + 格式-vs-实例机制发现**——
> 这是后训练岗位面试中极有区分度的故事。

## 简历条目

**Agent-TTRL：Agent 推理时强化学习的静默失败模式审计与修复系统
｜PyTorch、PEFT、LoRA、vLLM、GRPO**

构建 deployment-time agent RL 全链路，并用三阶段协议审计揭示正结果的真实来源：
- **失败模式 F1 — 训练-服务漂移**：LoRA 更新的模型从未成为后续 rollout 的
  serving policy（训练 GPU0、服务 GPU1 静态 base），96/96 任务结果在
  "不同"更新臂间完全一致——所谓更新效果是采样流位移伪影。修复：
  ColocatedPolicy（训练=服务同一 PEFT 模型）+ 原子 shadow commit + canary。
- **失败模式 F2 — 评测泄漏与锚点注入**：hidden evaluator 控制 early-stop、
  人工构造 ground-truth anchors 注入 replay → 产生虚假 +0.070 AUPC 迁移
  （p=0.016, 7/8 seeds）。修复：hidden 仅离线评分、anchors 从 disjoint
  predeployment 数据用可访问流程构造。
- **失败模式 F3 — 协议正确下更新有害**：全隔离（外生 request RNG、
  observation-only credit、signed replay、原子提交）后，REINFORCE 更新
  显著降低性能（naive -0.19, exact p=0.008, 0/8 seeds）；frozen 基线
  8 seeds 完全确定。**跨 backbone 复现**：同一协议在 Qwen2.5-7B 上
  naive/egc 均 8/8 seeds 低于 frozen（p=0.0078），sealed 未训练模板
  也退化（1.0→0.70/0.81）；pair-loss、宽松 gate 全部不迁移。
  **pre-commit gate（可访问证据验证候选）消除伤害**。
- **正对照 — 官方 tau2 基准满分**：同一 serving runtime 部署到官方 tau2
  环境（官方 agent/user/orchestrator/hidden evaluator/LLM judge 原样），
  仅替换模型后端为本地 14B（policy-consistent ColocatedPolicy）。服务侧
  工程（工具 schema 注入、工具结果 digest 压缩、防循环阀、原生工具调用）
  使 agent 完整完成任务：找到用户→数出 10 种可用 tshirt→告知用户→用
  真实 item ids 执行退货/换货——task 0 与 task 2 均 DB 1.0 + NL-assertion
  1.0（5/10 seeds）。证明"管道做不了任务"为假；失败只属于更新规则。
- **机制发现 — 格式 vs 实例**：穷尽扫描 10+ 更新规则（REINFORCE/
  verified-only/pair-loss/imitation/缺陷定向演示）× gate × 流长度 × 双
  backbone × 双环境后确认：任何"具体实例演示"（写进权重或 in-context）
  都会导致实例 id 复制（anchor priming 直接崩到 AUPC 0.0），只有占位符
  格式示例（order-EXAMPLE）有效——可迁移信号是格式而非实例。
- **可复现审计链**：A0-A2 集成门（RNG 隔离、commit 原子性、rollback）、
  请求级 CRN、hidden 隔离、evidence tier、全部 runs 全 manifest、2^n 精确统计。

## 90 秒面试故事

我之前做 Agent GRPO credit assignment 时发现线上 rollout 与 trainer 身份
不闭合，结果不可信。后来我把这条直觉做成了完整审计链：同一 pipeline 在
三个审计等级下得到三个相反结论——静态服务下更新效果是 RNG 伪影；加上
hidden evaluator 和人工 anchor 后出现"显著正迁移"（+0.070, p=0.016）；
全部隔离后更新反而显著有害（-0.19, p=0.008）。每一步都有可复现的代码、
manifest 和精确统计。**结论：Agent 测试时 RL 的正结果必须先过审计链——
服务身份、评测隔离、证据溯源、提交安全四关。** 我释放的 harness 让任何
pipeline 都能检查这三类静默失败模式。

## 面试追问准备

- 为什么三阶段结论不同？（每阶段只改一个审计项；正结果来自泄漏不是学习）
- 怎么证明 frozen 确定性？（外生 CRN：seed 不含 treatment；8/8 bitwise identical）
- 为什么更新有害？（canonical REINFORCE 负 advantage 惩罚部分正确序列；
  模型学会避免正确工具名 + 幻觉示例实体 id）
- gate 为什么有效？（候选在可访问证据实例上不优于 committed 则丢弃；
  0/8 有害提交，AUPC 回到 frozen）
- 与 aTTT/StarOR 的差异？（它们报告正结果；我们的贡献是"这些结果必须
  先过四关审计"，并提供工具）
- 下一步？（第二环境/更强模型复现 F3；更新规则重设计——先安全再有效）

## 关键数字（全部 manifest 支撑）

| 阶段 | naive−frozen AUPC | p（exact two-sided） | 结论 |
|---|---|---|---|
| F1 静态服务 | 不可识别（96/96 相同） | — | 伪影 |
| F2 泄漏 | +0.070（7/8 正） | 0.016 | 虚假正结果 |
| F3 全隔离 | −0.1875（0/8 正） | 0.0078 | 更新有害 |
| F3+gate | 0.0000（0/8 有害） | — | 防护生效 |

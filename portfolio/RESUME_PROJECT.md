# Agent-TTRL — 简历核心项目材料(2026-08-23 定稿)

> 状态:研究原型完成,含 45+ commits、127+ 测试、6 图论文草稿、全量 run manifests。
> 尚未投递。以下数字全部有 artifact 支撑(protocols/runs/)。

## 简历条目(设计文档 §24.2 模板 + 诚实数字)

**Agent-TTRL:状态化工具 Agent 的可验证推理时强化学习系统｜PyTorch、LoRA、GRPO、vLLM、PEFT**

构建 deployment-time rollout→evidence→LoRA update→shadow evaluation→rollback 闭环:
- **协议机制**:证据分层(E_hard/E_soft 进适应、hidden evaluator 永不进梯度/选择/门)、
  prequential 归纳迁移为主指标、三通道预算台账(B_env/B_model/B_update)、策略身份绑定
  (base/adapter/version 哈希),全线通过 GRPO-Guard 守卫链(契约 24/24)。
- **机制验证**:配对反事实分支(G×R CRN 耦合)产生 signed action credit,reliability/conflict
  双门控;SafeCommit(empirical-Bernstein e-process 门,覆盖模拟器冻结)在 candidate-archive
  回放中**灾难性更新率相对 always-commit 降低 100%**,且提交率非退化。
- **环境**:自建 ControlledToolShift(10 工具 × 5 shift 家族,F01-F12 黄金包)打通机制;
  AppWorld 0.2.0 + tau2-retail 真实任务适配(两个持久 exec server + 工具调用解析)。
- **诚实负结果**:CTS/AppWorld/tau2 × Qwen3-4B/Mistral-7B 上,部署期 LoRA-RL 在
  8-16 任务尺度无稳定 prequential 增益(8 任务 naive 0.093 vs frozen 0.072;16 任务
  naive 0.013 vs frozen 0.025)——按预注册停止条件转为协议机制 + 可复现负结果论文。
- **工程**:127+ 单测(契约 schema、CTS 黄金、统计覆盖)、运行清单全哈希、3 个执行
  服务器(rollout/exec/eval 权限分离)、GPU 共享卡规则(与 GRPO-Guard 并行记录)。

## 90 秒面试故事(设计文档 §24.4 改编)

我之前做 Agent GRPO credit assignment 时发现,线上 rollout 与 trainer 的身份不闭合,
导致结果不可信;因此先做了 GRPO-Guard 把在线更新链路校验起来。之后研究实际问题:
部署期 Agent 收到不完整的工具反馈,直接拿来做 test-time RL 会把错误写回策略。我的方法
只在可重放状态化环境的关键决策点做 paired counterfactual branch,用执行证据形成 signed
credit;LoRA 更新不直接上线,而是经过 shadow evaluation 和统计 commit gate 才影响后续
任务;评测用 prequential 流 + 完全不参与的 future holdout,全部成本计账。结论诚实:
在当前模型和更新幅度下无稳定迁移增益,但协议机制和风险受控提交被完整验证。

## 面试追问准备(设计文档 §24.5)

- 四类概念区分(test-time scaling/context/adaptation/RL)✓
- 为什么不是测试集训练(prequential + sealed holdout + manifest 隔离)✓
- verifier 与 hidden evaluator 的隔离机制(能力环境分离,import guard)✓
- branch credit 为何不声称无偏(estimand 限定 + 精确 oracle 校准)✓
- 行为策略与当前策略(生成服务 authoritative log-prob,无文本回退)✓
- action-token mask 构造(结构化 span,禁 str.find)✓
- 跨候选错误预算(α_k = 6α/π²k² 求和有界)✓
- sentinel overfit 防护(单次复用、sealed 后抽取、fresh reservoir)✓
- matched-cost 计算(三通道台账、守恒校验)✓
- 负迁移定义(anchor harm + catastrophic rate)✓
- 为什么不直接用 ACE/OLIVIA/GTTA(它们无参数更新,matched 基线已跑)✓
- 主效应失败的工程价值(协议机制 + SafeCommit + 可复现 harness)✓

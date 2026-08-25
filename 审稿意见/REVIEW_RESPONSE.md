# 五份审阅意见处理对照表（2026-08-25 定稿）

> 五份审阅（`Agent-TTRL_c6dddf_EXPERIMENT_AUDIT.md/.json`、
> `Agent-TTRL_c6dddf_PAPER_CLAIM_AUDIT.md/.json`、
> `Agent-TTRL_c6dddf_顶会逐页逐图逐实验代码审稿.md`）针对 v2 时代提交
> `c6dddf`（Strong Reject 2/10）。下表给出每类阻断问题的修复位置与证据。
> 当前主分支 `main`（含 v3 协议隔离 + v3.1 gate + v3.2 更新规则穷尽 + tau2 正对照）。

## 阻断问题 → 修复 → 证据

| 审稿阻断项 | 修复 | 证据 |
|---|---|---|
| 训练-服务隔离（trainer 更新从不进入 serving policy） | v3 ColocatedPolicy：同一 PEFT 模型生成+训练；原子 shadow commit + canary | `src/agent_ttrl/runtime/served_policy.py`；`tests/` A0-A2 集成门 |
| 隐藏状态/评测泄漏（hidden evaluator 参与 rollout） | v3 hidden 隔离：固定 episode horizon、仅离线评分；`R_hidden` 永不进梯度/门 | `paper/sections/05_results.tex` §F2；`AUDIT_INVALIDATION.md` |
| 随机数（treatment-dependent seeds / 分支污染生产序列） | v3 外生 request RNG：seed 只含协议哈希/流/任务/回合/用途，policy_version 仅记录 | `src/agent_ttrl/runtime/request_seed.py`；frozen 8/8 CRN 确定性 |
| replay/credit 泄漏（anchor 注入、负行缺失、未种子采样） | v3 observation-only 候选、signed replay、seeded intent-balanced sampler | `paper/sections/04_method.tex`；v3 manifests |
| 统计（单 seed、p 值口径、2^n 枚举） | ≥8 seeds、exact two-sided sign-flip、CRN 验证 | `paper/sections/05_results.tex` §F3 表格 |
| EGC 未优于 naive / 正结果由单 template 驱动 | v3 穷尽：REINFORCE / verified-only / pair-loss × gate × 流长度 — 协议正确下无正迁移；+0.070 定为泄漏伪影 | `protocols/runs/v3*`；commit e0286e9 |
| 正结果形态缺失（审稿要求"可发表含正结果"） | **tau2 官方环境正对照**：官方 agent/user/orchestrator/hidden evaluator/LLM judge 原样 + 本地 14B serving；task 0/2 满分 1.0（DB+NL-assertion），5/10 seeds | `paper/sections/07_tau2.tex`；`protocols/TAU2_OFFICIAL.md`；`protocols/runs/tau2_official/` |

## 论文现状

- 7 页 + 附录级证据：F1（静态服务伪影）→ F2（泄漏伪影）→ F3（协议正确下更新有害）
  → gate（防护）→ 更新规则穷尽 → **tau2 正对照（满分任务完成）**。
- 图：audit arc、v3 per-seed、tau2 seeds heatmap、tau2 成功轨迹。
- `paper/main.pdf` 本地编译通过（0 errors）。

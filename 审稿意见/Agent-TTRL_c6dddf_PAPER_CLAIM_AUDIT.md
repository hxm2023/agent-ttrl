# Agent-TTRL `c6dddf` — PAPER CLAIM AUDIT

```yaml
audit_date: 2026-08-24
repository: https://github.com/hxm2023/agent-ttrl
git_commit: c6dddfecd20ebe33de7a4cc5886d80e218896002
paper_sha256: 4b0cc4f6bb3b35ec2ce7d2b33b66a4ee05eb06b86aed681e9ff6d789e30f3120
review_independence: same-context-local
acceptance_status: provisional
verdict: FAIL
primary_reason: claim_evidence_version_mixing_and_implementation_contradictions
```

## Verdict

论文不能从当前 artifact bundle 重建为单一、闭合的主张集合。正文列出的逐 seed delta 与 Figure 1 来自首轮 `_16.log`；`p=.032`、6/8 和逐模板计数来自 rerun `_16a.log` / 当前 manifests。两个 run sets 的均值碰巧相同，但逐 seed outcomes 不同。这是 evidence-version mixing，不是无害 rounding。

另外，atomic commit、signed replay、hidden-evaluator isolation、R=4 counterfactual continuation、immutable manifests 和 all-manifests-released 等主张均被当前代码或文件清单直接反驳。

## 审计统计

| category | count |
|---|---:|
| supported/numerically supported within a named bundle | 5 |
| partial | 2 |
| hybrid evidence | 2 |
| contradicted | 8 |
| unsupported/missing/invalid provenance | 9 |
| total | **26** |

## 关键数字复核

### 首轮 `_16.log`

- frozen mean：0.453125
- naive mean：0.5234375
- naive−frozen：`[.0625,.125,.0625,.0625,.0625,.0625,0,.125]`
- mean：+0.0703125
- 7 positive / 1 zero / 0 negative
- exact two-sided sign-flip：`p=0.015625`

### rerun/current manifests

- frozen mean：0.453125
- naive mean：0.5234375
- naive−frozen：`[.0625,.125,0,.125,0,.125,.0625,.0625]`
- mean：+0.0703125
- 6 positive / 2 zero / 0 negative
- exact two-sided sign-flip：`p=0.03125`

### EGC 首轮

- EGC mean：0.500000
- EGC−frozen mean：+0.046875，`p=0.109375`
- EGC−naive mean：−0.0234375，`p=0.375`
- current rerun EGC manifests：3/8 only

## 26 项主张清单

| ID | claim | status | evidence/reason |
|---|---|---|---|
| C01 | CTS 16 tasks、8 seeds、Mistral-7B | SUPPORTED | logs/manifests |
| C02 | frozen .4531 vs naive .5234，mean +.070 | SUPPORTED_NUMERIC | 两套bundle均值相同；必须命名bundle |
| C03 | 正文八个delta同时对应p=.032 | HYBRID_FAIL | delta=首轮；p=rerun |
| C04 | Figure/正文为6/8 positive | HYBRID_FAIL | 图数据实际7/8；rerun为6/8 |
| C05 | A0–A2 end-to-end on Mistral-7B | MISSING | 无Mistral gate logs/manifests |
| C06 | adapter commit atomic | CONTRADICTED | train_step在commit前原地改served model |
| C07 | request-level paired RNG | PARTIAL | purpose isolation有；production seed含treatment-dependent version |
| C08 | signed replay | CONTRADICTED | EGC/naive只加入positive rows；positive-first截断 |
| C09 | intent-balanced replay | PARTIAL | 有bucket逻辑；sampler未seed且batch不固定 |
| C10 | hidden evaluator only for reporting | CONTRADICTED | hidden_success控制early stop |
| C11 | three-channel matched budgets | UNSUPPORTED | v2 stream无canonical ledger/manifest counts |
| C12 | rollout identity绑定base/adapter/version | UNSUPPORTED | manifest缺base/adapter hashes |
| C13 | R=4 fresh counterfactual continuations | CONTRADICTED | 四次相同确定性forced action |
| C14 | across-seed variance reliability | CONTRADICTED | continuation无随机性；U rows重复 |
| C15 | credit signs empirically correct | INVALID_PROVENANCE | hidden world给real user；图值硬编码 |
| C16 | rerun逐模板success counts | SUPPORTED | current frozen/naive manifests |
| C17 | sealed heldout 0/16→1/16 | SUPPORTED_NULL | current manifests；不支持transfer |
| C18 | EGC +.047,p=.11,6/8 | SUPPORTED_NUMERIC | 首轮logs；另有1 negative |
| C19 | EGC improves over baseline method | UNSUPPORTED | EGC低于naive |
| C20 | 三机制jointly necessary | MISSING_PHANTOM | 无对应ablation artifacts |
| C21 | large-LR harms 3/3 | MISSING | current v2 evidence bundle无对应run |
| C22 | exact sign-flip + hierarchical bootstrap | CONTRADICTED | exact脚本是MC；hierarchical脚本路径/outer unit错误 |
| C23 | immutable manifests | CONTRADICTED | fixed path write_text可覆盖 |
| C24 | all run manifests released | CONTRADICTED | EGC仅3/8 manifests |
| C25 | official AppWorld/Tau2 evidence | UNSUPPORTED | current v2只有CTS |
| C26 | inductive future transfer | UNSUPPORTED | sealed probe null；effect template-specific |

## Claim ceiling

当前可公开的最窄结论：

> A synthetic CTS-v2 pilot with Mistral-7B shows a template-specific transductive signal for canonical positive replay relative to a frozen policy. The effect is driven by the exchange template; the sealed-template probe is null; EGC does not outperform naive replay; runtime and artifact audits remain open.

不得公开为当前事实的措辞：

- EGC显著提升；
- inductive future transfer；
- atomic commit；
- signed replay；
- official benchmark validation；
- submission-ready/reproducible。

## Required closure

1. 选择一个不可变 run-set，全文/图/统计从同一 index重建。
2. 修复 shadow transaction、cross-arm seed、hidden control、EGC continuation与replay determinism。
3. 完成anchor factorial、EGC-vs-naive primary contrast和official environments。
4. 通过预注册 `alpha=.01`、session-level exact test和paired hierarchical CI。


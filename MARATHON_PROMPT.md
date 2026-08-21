# Marathon Prompt (mission order — CLAUDE.md is the constitution)

```
/research-pipeline "Run the Agent-TTRL project. READ CLAUDE.md in the project root
FIRST — it is the constitution: locked RQ (deployment-period parameter updates for
stateful tool-using agents under partial evidence, prequential evaluation); method
OPEN (EGC-TTRL from the reference design is H1, not default — Phase 1 must generate
≥3 fresh candidates and pass the novelty gate; the design doc's own warning: if
recent work covers both claims, shrink or change topic, never rename-and-submit);
protocol red lines (evidence tiers E_hard/E_soft vs R_hidden never in training;
prequential/inductive future transfer as PRIMARY metric; 3-channel fair budgets
B_env/B_model/B_verifier with one canonical ledger; policy identity binding;
GRPO-Guard correctness Gate is a HARD PREREQUISITE before formal result experiments;
M0 gate before any GPU; ≥5 seeds pre-registered tests); forbidden project forms
(design doc §1.2); K1-K5 kill conditions. AUTHORITATIVE REFERENCE (design input,
NOT gospel — Phase 0-1 validates/revises it): Agent-TTRL_顶会导向详细项目设计方案.md
(project root) — read fully: §3 domain map (11 works to differentiate, incl.
CausalFlow/T3RL/GTTA/ACE/OLIVIA/MemoPilot + track newer), §4 novelty gate, protocol,
M0 gate, execution order (M0 → Guard gate → ControlledToolShift → baselines →
single-mechanism EGC credit → SafeCommit → AppWorld/τ² → second model family).
Phase 0-1 = DEEP desk work (0 GPUh): re-sweep design-doc §3 map to current arXiv
(last 2 months), dual-domain survey (test-time adaptation/training + agent RL),
candidate generation ≥5 (≥3 fresh), candidate DEBATE with verdicts + falsification
conditions (CANDIDATE_KILL_LOG.md), SCOOP CHECK + PIVOT LOOP until clean,
ADMISSIBLE ONLY AFTER adversarial review (R1-R7: vs CausalFlow/T3RL/GTTA/ACE,
budget fairness, metric predicts prequential gain, hidden-evaluator leakage,
how-is-this-not-X), decision logs (6 files). Phase 2 per design-doc execution
order, starting with M0 (CPU: schemas/fixtures/overlap reports/baseline registry)
then GRPO-Guard correctness gate (external dependency — coordinate) then sandbox +
baselines + mechanism experiments + public envs + second model family. Compute:
jindun shared 8×A800 (others first, two-gate GPU check, never fight for GPU);
local RTX 5060 for M0/CPU; checkpoint-resume; results rsync + git continuously
(GitHub TBD). Target ICML 2027 (primary) / ACL 2027 / NeurIPS 2027 — venue by
paper shape at Phase 3. Compliance: AI-use disclosure per CFP; human owns the
paper. Start with Phase 0 re-sweep + candidate generation." --deep_mode: true,
auto_write: true, auto_proceed: true, venue: "ICML 2027 (fallback ACL/NeurIPS 2027)", arxiv_download: true
```

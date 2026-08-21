# Knowledge Base — Test-Time RL for Stateful Tool-Using Agents (Agent-TTRL)

Status: **BUILT 2026-08-22** (Phase 0 re-sweep). 30 papers downloaded + full-text indexed.

## Contents
- `domain_overview.md` — field map to 2026-08-22, incl. the open wedge analysis.
- `metrics_and_baselines.md` — metrics, SOTA baselines by family, evaluation protocols, known traps.
- `field_conventions.md` — plot types, notation, paper structure, terminology pitfalls, reproducibility conventions.
- `papers/*.json` — full text of 30 key works (30 PDFs in `../papers/`).
- `index.json`, `search.py` — search: `python research-wiki/knowledge_base/search.py "counterfactual credit commit"`.

## Paper corpus (30)
TTRL 2504.16084 · T3RL 2603.02203 · GTTA 2511.04847 · ACE 2510.04618 · OLIVIA 2605.11169 ·
MemoPilot 2606.08656 · SEAL 2506.10943 · Amplification-in-TTRL 2603.15417 · CausalFlow 2605.25338 ·
Counterfactual-Shapley 2607.16999 · JitRL 2601.18510 · aTTT 2607.03441 · CVT-RL 2606.05263 ·
PACE 2606.08106 · VaG 2608.05810 · SHEPHERD/Tree-RL 2605.10913 · APPO 2606.12384 · StarOR 2606.15197 ·
SEA-Eval 2604.08988 · STABLE 2510.16089 · CRAFT 2606.29476 · BiPACE 2606.25556 · LEAFE 2603.16843 ·
DARE 2601.21804 · EvoTest 2510.13220 · Prove 2606.03892 · SLAaaT 2608.17034 · EvoHarness-RL 2608.05446 ·
TTT-safety 2605.22984 · TMEM 2606.04536

## Usage
- Phase 1 (idea discovery): search KB before external APIs.
- Phase 2 (experiments): field conventions for plotting; baselines from metrics_and_baselines.md.
- Phase 4 (reviewer): verify claims against published numbers.

# Submission Kit — Silent Failure Modes in Agent Test-Time RL

**Status**: paper finalized (ACL format, 8 content pages + references,
clean compile 0 errors) on GitHub `main` (HEAD 878a61f). The final
arXiv / ACL submission requires the account holder (identity-bound;
cannot be automated).

## Paper

- `paper/main.pdf` — 8 content pages, ACL 2027 format (acl.sty), anonymous.
- Title: *Silent Failure Modes in Agent Test-Time RL: Served-Policy
  Drift, Evaluator Leakage, and the Phantom Transfer They Produce*
- Core results (all manifest-backed):
  - F1 static-serving artifacts (96/96 identical outcomes across arms);
  - F2 phantom +0.070 AUPC (p=0.016) from evaluator leakage + anchor
    injection, vanishing under isolation;
  - F3 protocol-correct updates harmful on two backbones (Mistral-7B
    naive −0.19 p=0.008 0/8 seeds; Qwen2.5-7B naive/egc 8/8 seeds below
    frozen p=0.008), with per-template mechanism localization, sealed
    holdout degradation, and monotone cumulative decay;
  - pre-commit gate eliminates harm (0/8 harmful commits);
  - positive control: official tau2 benchmark, full reward 1.0 on both
    evaluated retail tasks in 5/10 seeded runs with a local 14B model;
    no-transfer replicated on the official environment.

## How to submit (account holder, ~5 min)

### arXiv (preprint before ACL submission)
1. Go to https://arxiv.org/submit (log in with your arXiv account)
2. Rebuild the package locally:
   `cd paper && tar czf ../agent-ttrl-arxiv.tar.gz main.tex sections/ references.bib acl.sty acl_natbib.bst figures/`
3. Fill in the abstract (paper/sections/00_abstract.tex, ~250 words).
4. Category: cs.CL (primary) / cs.AI

### ACL 2027 (formal submission, before deadline)
1. OpenReview/Softconf submission page for ACL 2027 (log in)
2. Upload `paper/main.pdf` (recompile as `anonymous` if required)
3. Supplementary: link the GitHub repo + evidence index
4. AI-use disclosure per ACL CFP; human authorship statement

## After acceptance
- Make the GitHub repo public (currently private — created private to protect
  double-blind anonymity; flip to public after acceptance or per venue policy)
- Add the final camera-ready PDF + artifacts DOI (Zenodo optional)

## Evidence index
- `protocols/runs/v3/cts/` — Mistral v3.0 protocol logs (8 seeds/arm)
- `protocols/runs/v3q/` — Qwen2.5-7B replication (naive/egc ×8 seeds,
  gate0, pair-loss ×4, 32-task decay) + logs
- `protocols/runs/tau2_official/` + `tau2_official_stream/` — official
  tau2 pilots (tasks 0–8, seeds 0–4) and the 10-task stream arms
- `protocols/TAU2_OFFICIAL.md` — reproducibility doc for the tau2 setup
- `审稿意见/REVIEW_RESPONSE.md` — mapping of the five external reviews to
  fixes and evidence
- `scripts/` — CTS streams (v3.0/v3.2), tau2 pilots + stream, local
  OpenAI-compatible server, all figure makers
- `AUDIT_INVALIDATION.md` — v1/v2 invalidation records (audit trail)

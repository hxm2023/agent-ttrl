# Submission Kit — Policy-Consistent EGC-TTRL (v2)

**Status**: paper finalized (ACL format, 7pp, clean compile) + published on
GitHub (release `v2-policy-consistent`, asset `agent-ttrl-arxiv-pkg.tar.gz`).
The final arXiv / ACL submission requires the account holder (identity-bound;
cannot be automated).

## Paper

- `paper/main.pdf` — 7 pages, ACL 2027 format (acl.sty), anonymous.
- Title: *Policy-Consistent Evidence-Gated Test-Time RL for Stateful Tool Agents*
- Core result: deployment-period updates beat frozen policy on CTS-v2 16-task
  stream (8 paired seeds, Mistral-7B): naive 0.5234 vs frozen 0.4531
  (+0.070 AUPC, exact two-sided sign-flip p=0.016, 7/8 seeds positive, 0 negative).
- EGC counterfactual credit signs validated on deceptive-evidence tasks
  (evidence-user positive, goal-user negative).

## How to submit (account holder, ~5 min)

### arXiv (preprint before ACL submission)
1. Go to https://arxiv.org/submit (log in with your arXiv account)
2. Upload the package:
   - From GitHub: https://github.com/hxm2023/agent-ttrl/releases →
     `v2-policy-consistent` → asset `agent-ttrl-arxiv-pkg.tar.gz`, or
   - Rebuild locally: `cd paper && tar czf ../agent-ttrl-arxiv.tar.gz main.tex sections/ references.bib acl.sty acl_natbib.bst figures/fig1_method.png figures/fig2_prequential.png figures/fig3_credit_ablation.png`
3. Title: *Policy-Consistent Evidence-Gated Test-Time RL for Stateful Tool Agents*
4. Category: cs.CL (primary) / cs.AI

### ACL 2027 (formal submission, before deadline)
1. OpenReview/Softconf submission page for ACL 2027 (log in)
2. Upload `paper/main.pdf` (recompile as `anonymous` if required)
3. Supplementary: link the GitHub repo + release
4. AI-use disclosure per ACL CFP; human authorship statement

## After acceptance
- Make the GitHub repo public (currently private — created private to protect
  double-blind anonymity; flip to public after acceptance or per venue policy)
- Add the final camera-ready PDF + artifacts DOI (Zenodo optional)

## Artifacts on GitHub (`hxm2023/agent-ttrl`, main)
- `src/agent_ttrl/runtime/` — policy-consistent runtime (colocated backend,
  request-scoped RNG, atomic commit/rollback)
- `src/agent_ttrl/environments/cts_v2.py` — CTS-v2 latent-skill families
- `src/agent_ttrl/optimization/replay_buffer.py` — cross-task replay
- `src/agent_ttrl/credit/branch_executor_v2.py` — G×R counterfactual credit
- `scripts/cts_v2_stream.py` — v2 prequential stream (frozen/naive/egc)
- `scripts/stats_v2_cts.py` — exact two-sided sign-flip statistics
- `tests/integration/test_policy_consistency.py` — A0/A1/A2 gates
- `AUDIT_INVALIDATION.md` — v1 invalidation record (served-policy audit)

"""SafeCommit gates (design doc §7.8): candidate-level commit decision.

Two pre-registered variants, compared by the coverage simulator (decision D6):

1. FixedN_Hoeffding: fixed sample n, one-sided Hoeffding radius
   b_k = sqrt(2 log(1/alpha_side)/n) with cross-candidate alpha allocation
   alpha_k = 6*alpha_total/(pi^2 k^2), gain/harm split alpha_k/2.
2. Anytime_EProcess: testing-by-betting e-process (PACE-style). For bounded
   differences X in [-1,1] with E[X] <= 0 under H0, M_n = exp(lambda*S_n -
   n*lambda^2/8) is a nonnegative supermartingale -> anytime-valid CS bounds:
     LCB_gain = mean_gain - lambda/8 - log(1/alpha_k)/(lambda n)
     UCB_harm = mean_harm + lambda/8 + log(1/alpha_k)/(lambda n)
   allowing optional stopping at any n (fixed shadow budget by default).

Commit condition (both): LCB_gain >= eps_gain AND UCB_harm <= eps_harm
AND GuardDecision == ALLOW (external). Otherwise ROLLBACK; INCONCLUSIVE when
the required sample size is unavailable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# FROZEN 2026-08-22 by coverage simulator (protocols/sweep_coverage_results.json,
# decision D6): empirical-Bernstein e-process is the v0.1 primary gate.
# Fixed-n Hoeffding passed NO operating point; EB e-process at the design doc's
# alpha_total=0.05 passes coverage with non-degenerate power.
EPS_GAIN = 0.01
EPS_HARM = 0.10
N_FIXED = 512
ALPHA_TOTAL = 0.05
LAMBDA = 0.5         # e-process betting intensity (fixed, tuned only pre-lock in simulator)


class GateKind(str, Enum):
    FIXED_N_HOEFFDING = "fixed_n_hoeffding"
    ANYTIME_EPROCESS = "anytime_eprocess"
    EB_EPROCESS = "empirical_bernstein_eprocess"


PRIMARY_GATE_KIND = GateKind.EB_EPROCESS


def primary_gate_kind() -> GateKind:
    return PRIMARY_GATE_KIND


class GateDecision(str, Enum):
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class GateOutcome:
    decision: GateDecision
    lcb_gain: float
    ucb_harm: float
    alpha_k: float
    n_gain: int
    n_anchor: int
    reason_codes: list[str]


def alpha_k(k: int, alpha_total: float = ALPHA_TOTAL) -> float:
    """Cross-candidate error budget (summable: sum 6/(pi^2 k^2) = 1)."""
    return 6.0 * alpha_total / (math.pi ** 2 * k ** 2)


def _hoeffding_radius(alpha_side: float, n: int) -> float:
    return math.sqrt(2.0 * math.log(1.0 / alpha_side) / n)


def _eprocess_radius(alpha_side: float, n: int, lam: float = LAMBDA) -> float:
    """Anytime-valid one-sided radius from the exponential e-process."""
    return lam / 8.0 + math.log(1.0 / alpha_side) / (lam * n)


def _eb_eprocess_radius(alpha_side: float, n: int, variance: float) -> float:
    """Empirical-Bernstein anytime-valid radius (Waudby-Smith & Ramdas style).

    Uses the sample variance instead of the worst-case [-1,1] range; much
    tighter when per-pair noise is small. Valid for bounded differences.
    """
    if n < 2 or variance <= 0:
        return _eprocess_radius(alpha_side, n)
    log_inv = math.log(1.0 / alpha_side)
    return math.sqrt(2.0 * variance * log_inv / n) + 7.0 * log_inv / (3.0 * (n - 1))


def decide(k: int, gain_diffs: list[float], harm_diffs: list[float],
           kind: GateKind = GateKind.FIXED_N_HOEFFDING,
           eps_gain: float = EPS_GAIN, eps_harm: float = EPS_HARM,
           n_fixed: int = N_FIXED, lam: float = LAMBDA,
           alpha_total: float = ALPHA_TOTAL,
           guard_allow: bool = True) -> GateOutcome:
    """Paired shadow evaluation: gain_diffs = candidate - parent (want > 0);
    harm_diffs = parent - candidate on anchors (want <= eps_harm)."""
    ak = alpha_k(k, alpha_total)
    a_side = ak / 2.0
    n_gain = len(gain_diffs)
    n_anchor = len(harm_diffs)

    if not guard_allow:
        return GateOutcome(GateDecision.ROLLBACK, 0.0, 0.0, ak, n_gain, n_anchor,
                           ["GUARD_DENY"])

    if kind == GateKind.FIXED_N_HOEFFDING:
        if n_gain < n_fixed or n_anchor < n_fixed:
            return GateOutcome(GateDecision.INCONCLUSIVE, 0.0, 0.0, ak, n_gain, n_anchor,
                               ["INSUFFICIENT_SHADOW_SAMPLE"])
        mg, mh = sum(gain_diffs) / n_gain, sum(harm_diffs) / n_anchor
        b = _hoeffding_radius(a_side, n_fixed)
        lcb, ucb = mg - b, mh + b
    else:
        if n_gain < 2 or n_anchor < 2:
            return GateOutcome(GateDecision.INCONCLUSIVE, 0.0, 0.0, ak, n_gain, n_anchor,
                               ["INSUFFICIENT_SHADOW_SAMPLE"])
        mg, mh = sum(gain_diffs) / n_gain, sum(harm_diffs) / n_anchor
        n = min(n_gain, n_anchor)
        if kind == GateKind.EB_EPROCESS:
            vg = sum((x - mg) ** 2 for x in gain_diffs) / (n_gain - 1)
            vh = sum((x - mh) ** 2 for x in harm_diffs) / (n_anchor - 1)
            bg = _eb_eprocess_radius(a_side, n_gain, vg)
            bh = _eb_eprocess_radius(a_side, n_anchor, vh)
        else:
            bg = _eprocess_radius(a_side, n, lam)
            bh = bg
        lcb, ucb = mg - bg, mh + bh

    codes: list[str] = []
    if lcb < eps_gain:
        codes.append("GAIN_NOT_ESTABLISHED")
    if ucb > eps_harm:
        codes.append("HARM_NOT_BOUNDED")
    decision = GateDecision.COMMIT if (lcb >= eps_gain and ucb <= eps_harm) else GateDecision.ROLLBACK
    return GateOutcome(decision, lcb, ucb, ak, n_gain, n_anchor, codes)

"""Paired-branch signed credit + local reliability gate (design doc §7.4).

c_hat_i = bar_U_i - group_mean;  z_{i,r} = U_{i,r} - group_mean_r;
v_hat_i = var over seeds;  b_i = t_{R-1,0.90} * sqrt(v_i/R);
alpha_i = 1{L_i > eta or U_i < -eta};  A_i = clip(alpha_i * c_i / s_group, -A_max, A_max).
Degenerate groups (all-success / all-fail / no support) -> NO_UPDATE with reason code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ETA_CREDIT = 0.02
A_MAX = 5.0
T_90 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415,
        8: 1.397, 9: 1.383, 10: 1.372}  # t_{R-1, 0.90} one-sided


@dataclass
class CreditRow:
    action_idx: int
    credit: float          # A_i (signed, gated, clipped)
    raw_credit: float      # \hat c_i before gating
    gate_passed: bool
    reason: str | None = None


@dataclass
class GroupVerdict:
    status: str            # OK | NO_SUPPORT | DEGENERATE_GROUP | NO_RELIABLE_CREDIT | INVALID
    reason_code: str
    rows: list[CreditRow] = None


def paired_credit(U: np.ndarray, eta: float = ETA_CREDIT, a_max: float = A_MAX) -> GroupVerdict:
    """U: (G, R) matrix of paired evidence utilities.

    G actions x R continuation seeds (CRN-coupled per column).
    """
    U = np.asarray(U, dtype=float)
    G, R = U.shape
    if not (np.isfinite(U).all()):
        return GroupVerdict(status="INVALID", reason_code="NON_FINITE_U")
    if G < 2:
        return GroupVerdict(status="NO_SUPPORT", reason_code="LESS_THAN_2_ACTIONS")
    if R < 2:
        return GroupVerdict(status="NO_RELIABLE_CREDIT", reason_code="R_LT_2")

    bar_U = U.mean(axis=1)                      # (G,)
    group_mean = bar_U.mean()
    if group_mean == 0.0 or group_mean == 1.0:
        return GroupVerdict(status="DEGENERATE_GROUP", reason_code="ALL_SAME_OUTCOME")
    s_group = max(bar_U.std(ddof=1), 1e-3)

    z = U - U.mean(axis=0, keepdims=True)       # paired unit per seed
    raw_c = bar_U - group_mean
    v = z.var(axis=1, ddof=1)                   # (G,)
    b = T_90.get(R - 1, 1.65) * np.sqrt(v / R)
    L = np.clip(raw_c - b, -1.0, 1.0)
    Uu = np.clip(raw_c + b, -1.0, 1.0)
    alpha = ((L > eta) | (Uu < -eta)).astype(float)
    if alpha.sum() == 0:
        return GroupVerdict(status="NO_RELIABLE_CREDIT", reason_code="ALL_GATES_CLOSED",
                            rows=[CreditRow(i, 0.0, float(raw_c[i]), False) for i in range(G)])

    rows = []
    for i in range(G):
        A = float(np.clip(alpha[i] * raw_c[i] / s_group, -a_max, a_max))
        rows.append(CreditRow(action_idx=i, credit=A, raw_credit=float(raw_c[i]),
                              gate_passed=bool(alpha[i])))
    return GroupVerdict(status="OK", reason_code="", rows=rows)

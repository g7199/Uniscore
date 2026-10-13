"""Somers' D with DeLong covariance, confidence intervals and the exact Wilcoxon signed-rank test.

The evidence of a criterion is Somers' D between two groups of texts, D = P(S^H > S^L) - P(S^H < S^L) = 2 AUC - 1:
ordinal, tie-aware and invariant to monotone rescaling of the judge scale.  Its sampling covariance across criteria
measured on the same texts follows DeLong et al. (1988), computed from placement values (Sun & Xu, 2014).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def placement_values(low, high):
    """Placement values with mid-ranks for ties: (v_high, v_low)."""
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    m, n = len(high), len(low)
    tz = stats.rankdata(np.r_[high, low])
    tx = stats.rankdata(high)
    ty = stats.rankdata(low)
    v_high = (tz[:m] - tx) / n
    v_low = 1.0 - (tz[m:] - ty) / m
    return v_high, v_low


def somers_d_cov(X_low, X_high, se_floor: float = 1e-3):
    """Somers' D of every column between X_low (n_L, K) and X_high (n_H, K), with its DeLong covariance.

    Returns dict(D, cov, se, auc, n_low, n_high)."""
    X_low = np.atleast_2d(np.asarray(X_low, float))
    X_high = np.atleast_2d(np.asarray(X_high, float))
    n_l, n_h, K = X_low.shape[0], X_high.shape[0], X_low.shape[1]
    VH = np.empty((n_h, K))
    VL = np.empty((n_l, K))
    for k in range(K):
        VH[:, k], VL[:, k] = placement_values(X_low[:, k], X_high[:, k])
    auc = VH.mean(0)
    s10 = np.atleast_2d(np.cov(VH, rowvar=False)) if K > 1 else np.array([[VH.var(ddof=1)]])
    s01 = np.atleast_2d(np.cov(VL, rowvar=False)) if K > 1 else np.array([[VL.var(ddof=1)]])
    cov = 4.0 * (s10 / n_h + s01 / n_l)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    floor = se_floor / max(1.0, np.sqrt(min(n_l, n_h) / 20.0))   # a perfectly separated or constant criterion keeps some uncertainty
    se = np.maximum(se, floor)
    return {"D": 2.0 * auc - 1.0, "cov": cov, "se": se, "auc": auc, "n_low": n_l, "n_high": n_h}


def somers_d_ci(D, se, level: float = 0.95):
    """Confidence interval of Somers' D built on the logit scale of AUC = (D + 1) / 2 (stays inside [-1, 1])."""
    D = np.asarray(D, float)
    se = np.asarray(se, float)
    z = stats.norm.isf((1 - level) / 2)
    auc = np.clip(0.5 * (D + 1.0), 1e-4, 1 - 1e-4)
    sl = 0.5 * se / (auc * (1 - auc))
    lo = 1.0 / (1.0 + np.exp(-(np.log(auc / (1 - auc)) - z * sl)))
    hi = 1.0 / (1.0 + np.exp(-(np.log(auc / (1 - auc)) + z * sl)))
    return 2 * lo - 1, 2 * hi - 1


def wilcoxon_exact(x, alternative: str = "greater", zero_tol: float = 1e-12):
    """Exact Wilcoxon signed-rank test of paired differences x (zero differences are dropped; n_eff <= 20).

    Returns dict(p, n_eff, ties, wins, losses, statistic)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    keep = np.abs(x) > zero_tol
    xe = x[keep]
    n = len(xe)
    out = dict(n_eff=int(n), ties=int(len(x) - n), wins=int((xe > 0).sum()), losses=int((xe < 0).sum()))
    if n == 0:
        return dict(out, p=1.0, statistic=0.0)
    if n > 20:
        raise ValueError("wilcoxon_exact enumerates 2^n sign patterns: n_eff <= 20")
    r = stats.rankdata(np.abs(xe))
    w_obs = float(r[xe > 0].sum())
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    w = signs @ r
    tol = 1e-9
    p_greater = float((w >= w_obs - tol).mean())
    p_less = float((w <= w_obs + tol).mean())
    p = {"greater": p_greater, "less": p_less, "two-sided": min(1.0, 2 * min(p_greater, p_less))}[alternative]
    return dict(out, p=p, statistic=w_obs)

"""Anchor selection and the maps from evidence to weights (AHP priority vector, normalised evidence)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def ev_auc(low, high) -> float:
    """Somers' D = 2 AUC - 1 of one criterion between two groups."""
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    n_l, n_h = len(low), len(high)
    ranks = stats.rankdata(np.r_[low, high])
    u = ranks[n_l:].sum() - n_h * (n_h + 1) / 2.0
    return float(2.0 * u / (n_l * n_h) - 1.0)


def ahp_eigen_weights(d, scale: float = 8.0):
    """Signed AHP priority vector of signed evidence d.

    Pairwise comparison a_ij = 1 + scale * (|d_i| - |d_j|) when |d_i| >= |d_j| (reciprocal otherwise), clipped to Saaty's
    [1/9, 9]; the priorities are the normalised principal eigenvector, signed by the evidence.  Returns (w, consistency ratio)."""
    d = np.asarray(d, float)
    k = len(d)
    a = np.abs(d)
    mat = np.ones((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            delta = a[i] - a[j]
            s = float(np.clip(1.0 + scale * abs(delta), 1.0, 9.0))
            mat[i, j] = s if delta >= 0 else 1.0 / s
    eigvals, eigvecs = np.linalg.eig(mat)
    idx = int(np.argmax(eigvals.real))
    w = eigvecs[:, idx].real
    w = w / w.sum()
    lam = float(eigvals[idx].real)
    ci = (lam - k) / (k - 1) if k > 1 else 0.0
    ri = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}.get(k, 1.5)
    cr = ci / ri if ri else 0.0
    return np.where(d >= 0, 1.0, -1.0) * w, cr


def direct_weights(d):
    """Normalised evidence as weights (the map used in the ablation 'evidence without AHP map')."""
    d = np.asarray(d, float)
    a = np.abs(d)
    tot = a.sum()
    w = a / tot if tot > 0 else np.full(len(d), 1.0 / len(d))
    return np.where(d >= 0, 1.0, -1.0) * w


def resolve_p(p, n_ref, n_pool):
    """Tail fraction of the anchors. p='auto': about 1.2 * n_ref candidates per tail (floor 5 %, cap 50 %)."""
    if p == "auto":
        return float(max(0.05, min(0.5, 1.2 * (n_ref or 0.05 * n_pool) / max(1, n_pool))))
    return float(p)


def reference_groups(y, task: str, p, n_ref=None):
    """Indices of the low and the high tail of the labels (binary task: the two classes)."""
    y = np.asarray(y, float)
    if task == "binary":
        return np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    p = resolve_p(p, n_ref, len(y))
    lo_thr, hi_thr = np.quantile(y, p), np.quantile(y, 1 - p)
    lo = np.flatnonzero(y <= lo_thr)
    hi = np.flatnonzero(y >= hi_thr)
    both = np.intersect1d(lo, hi)  # heavy ties: drop rows that fall in both tails
    if len(both):
        lo, hi = np.setdiff1d(lo, both), np.setdiff1d(hi, both)
    if len(lo) == 0 or len(hi) == 0:
        order = np.argsort(y, kind="stable")
        k = max(1, int(p * len(y)))
        lo, hi = order[:k], order[-k:]
    return lo, hi


def sample_reference(lo, hi, n_ref, rng):
    """Draw at most n_ref anchors from each tail (n_ref=None: every row of the tails)."""
    if n_ref is None:
        return lo, hi
    a = rng.choice(lo, min(n_ref, len(lo)), replace=False)
    b = rng.choice(hi, min(n_ref, len(hi)), replace=False)
    return a, b

"""Grouped splits, the empirical-CDF scaling of the judge-free criteria and the Spearman correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats


def grouped_splits(groups, n_splits: int = 30, fracs=(0.6, 0.1, 0.3), seed: int = 0):
    """Repeated random train / validation / test splits at the group level (no source text is shared between roles)."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    n = len(uniq)
    rng = np.random.RandomState(seed)
    out = []
    a = int(round(fracs[0] * n))
    b = a + int(round(fracs[1] * n))
    for _ in range(n_splits):
        perm = rng.permutation(n)
        role = np.empty(n, dtype=np.int8)
        role[perm[:a]] = 0
        role[perm[a:b]] = 1
        role[perm[b:]] = 2
        row_role = role[inv]
        out.append({k: np.flatnonzero(row_role == i) for i, k in enumerate(("train", "val", "test"))})
    return out


def quantile_scale(train_vals, all_vals):
    """Map a judge-free criterion to the 1-5 criterion range with the training split's empirical CDF (mid-ranks; no labels)."""
    ref = np.sort(np.asarray(train_vals, float))
    v = np.asarray(all_vals, float)
    lo = np.searchsorted(ref, v, side="left")
    hi = np.searchsorted(ref, v, side="right")
    return 1.0 + 4.0 * ((lo + hi) / 2.0) / max(1, len(ref))


def spearman(y, s) -> float:
    """Spearman correlation (0 for a constant score or label)."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    if np.ptp(s) == 0 or np.ptp(y) == 0:
        return 0.0
    try:
        v = stats.spearmanr(y, s)
        v = v[0] if isinstance(v, tuple) else getattr(v, "statistic", v)
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0

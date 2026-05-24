import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

def signed_jsd(low_vals, high_vals, bins=10, base=2):
    hist_low, _ = np.histogram(low_vals, bins=bins, density=True)
    hist_high, _ = np.histogram(high_vals, bins=bins, density=True)
    sign = np.sign(np.mean(high_vals) - np.mean(low_vals))
    jsd_val = jensenshannon(hist_low, hist_high, base=base)
    return float(sign * jsd_val)

def build_ahp_matrix(jsd_scores, method="linear"):
    keys = list(jsd_scores.keys())
    n = len(keys)
    mat = np.ones((n, n))
    
    # Calculate global max delta for scaling (optional, but good for robust linear)
    vals = [abs(v) for v in jsd_scores.values()]
    max_delta = max(vals) - min(vals) if vals else 1.0
    if max_delta == 0: max_delta = 1.0

    for i in range(n):
        for j in range(n):
            if i == j: 
                continue
            
            val_i = abs(jsd_scores[keys[i]])
            val_j = abs(jsd_scores[keys[j]])
            delta = val_i - val_j
            
            if method == "linear":
                # Original logic: 1 + 8 * delta (clamped to 1/9 ~ 9?)
                # Code history: 1 + 8 * abs(delta). 
                # If delta > 0: s. If delta < 0: 1/s.
                s = 1 + 8 * abs(delta)
                if delta >= 0:
                    mat[i, j] = s; mat[j, i] = 1.0 / s
                else:
                    mat[i, j] = 1.0 / s; mat[j, i] = s
            
            elif method == "exponential":
                # Consistent mapping: s_ij = exp(k * (v_i - v_j))
                # Scale k so that max_delta maps to ~9 (exp(2.2) ~ 9)
                # k = ln(9) / max_delta
                # This ensures the largest difference maps to roughly 9, 
                # and preserves perfect consistency (CR ~ 0).
                k = np.log(9) / max_delta
                s = np.exp(k * delta)
                mat[i, j] = s
                # mat[j, i] is implicitly 1/s if we loop fully, 
                # but to be safe/symmetric if we only loop upper triangle:
                # (Loop is simple n*n here, so fine)
    
    return pd.DataFrame(mat, index=keys, columns=keys)

def compute_ahp_weights_and_consistency(matrix):
    n = matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(eigvals.real)
    lambda_max = eigvals[max_index].real
    w = eigvecs[:, max_index].real
    weights = w / w.sum()
    ci = (lambda_max - n) / (n - 1)
    RI_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    ri = RI_table.get(n, 1.12)
    cr = ci / ri if ri != 0 else 0
    return weights, lambda_max, ci, cr

def compute_score(row, weights, metrics):
    return sum(row[k] * weights[k] for k in metrics)


# ===== UniScore v2: Marginal Mahalanobis Info-Gain AHP =====
# Per-axis statistic = Δμᵀ Σ⁻¹ Δμ − Δμ₋ₖᵀ Σ₋ₖ⁻¹ Δμ₋ₖ (axis k's marginal contribution
# to between-class Mahalanobis squared distance). Applied through the same AHP
# eigenvector aggregation as the original UniScore framework.

def _info_gain_per_axis(delta, S):
    m = len(delta)
    S_inv = np.linalg.pinv(S)
    full = float(delta @ S_inv @ delta)
    gains = np.zeros(m)
    for k in range(m):
        keep = [i for i in range(m) if i != k]
        d_k = delta[keep]
        S_k = S[np.ix_(keep, keep)]
        gains[k] = full - float(d_k @ np.linalg.pinv(S_k) @ d_k)
    return np.maximum(gains, 0)


def info_gain_ahp_weights(low_group_X, high_group_X, metrics):
    """K09 method: per-axis marginal Mahalanobis info-gain → AHP eigenvector.

    Returns a dict {metric: signed_weight} compatible with compute_score().
    """
    Xl = np.asarray(low_group_X, dtype=float)
    Xh = np.asarray(high_group_X, dtype=float)
    m = Xl.shape[1]
    S = 0.5 * (np.cov(Xl, rowvar=False) + np.cov(Xh, rowvar=False)) + 1e-3 * np.eye(m)
    delta = Xh.mean(0) - Xl.mean(0)
    gains = _info_gain_per_axis(delta, S)
    # AHP comparison matrix on |gain| ratios (positive-reciprocal)
    g_abs = np.abs(gains) + 1e-3
    A = g_abs[:, None] / g_abs[None, :]
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    w = np.abs(eigvecs[:, idx].real); w /= (w.sum() + 1e-12)
    # Sign from Mahalanobis direction
    sgn = np.sign(np.linalg.pinv(S) @ delta)
    w_signed = sgn * w
    return dict(zip(metrics, w_signed))

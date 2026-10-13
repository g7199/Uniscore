"""UniScore: a two-level AHP hierarchy over the judge-free and the judge criteria.

Two kinds of supervision of very different sizes are available: labels for a large pool of texts (no judge call), and
judge scores for B anchors, B/2 texts from each tail of the labels.  UniScore fits weights where the supervision is
plentiful and generates them from tested evidence where it is scarce:

  level 2, judge-free family  w_F: rank-ridge of the ranked labels on the judge-free criteria, fitted on every labelled pool text
  level 2, judge family       w_J: AHP priority vector of the Somers' D of every judge criterion between the two anchor tails
                                    (comparisons a_ij = 1 + 8 (|D_i| - |D_j|), reciprocal otherwise; signs from the evidence)
  level 1, the two families   e_J = Somers' D of the judge family's out-of-fold score between the anchors,
                              e_F = Somers' D of the judge-free family's out-of-fold score between all pool texts of the anchors' tails;
                              a = 1 + 8 (e_J - e_F) (reciprocal if negative), alpha = a / (1 + a)

  S(x) = (1 - alpha) w_F.x_F / sigma_F + alpha w_J.x_J / sigma_J      (sigma: pooled within-tail sd of each family score)

The keyword arguments other than the defaults produce the variants of the ablation (Table 5 of the paper).
"""

from __future__ import annotations

import numpy as np

from . import stats as S
from .estimators import Aggregator, RankRidgeCV, SomersAHP
from .weights import reference_groups, resolve_p, sample_reference

LEVEL1 = ("ahp_pool", "ahp", "ahp_pool_direct", "lda", "equal")


def tail_sd(v, lo, hi):
    """Pooled within-tail standard deviation of the scores v (index arrays lo, hi)."""
    ss = ((v[lo] - v[lo].mean()) ** 2).sum() + ((v[hi] - v[hi].mean()) ** 2).sum()
    return float(np.sqrt(ss / max(1, len(lo) + len(hi) - 2)) + 1e-12)


class UniScore(Aggregator):
    """UniScore (see the module docstring).

    free               name prefixes of the judge-free criteria; every other column is a judge criterion
    p, n_ref           tail fraction of the anchors ("auto") and anchors per tail (B/2)
    scale              AHP constant (8)
    seed               anchors, cross-fitting folds
    pi                 fixed priority of the judge family (0: judge-free family only, 1: judge family only)
    level1             family priority: "ahp_pool" (default: e_F on the pool), "ahp" (e_F on the anchors),
                       "ahp_pool_direct" (alpha = e_J / (e_J + e_F)), "lda" (fitted on the anchors), "equal" (alpha = .5)
    mapping            "ahp" (default) or "direct" (normalised Somers' D as the judge weights)
    anchored           factory seed -> estimator that replaces AHP on Somers' D in the judge family (e.g. LDA)
    anchor_all         the judge-family estimator is fitted on all criteria (no criterion families on the judge side)
    free_from_anchors  the judge-free weights come from the anchors instead of the labelled pool
    free_model         "ridge" (default) or "ahpd": AHP priorities of each judge-free criterion's Somers' D between the
                       outer label fractions p_free of the pool
    """

    name = "UniScore"
    _POOL: dict = {}          # judge-free family fits, shared by every variant evaluated on the same pool
    _OOF_ANCHORS: dict = {}

    def __init__(self, free=("q_", "e_"), p="auto", n_ref=100, scale=8.0, seed=0, pi=None, level1="ahp_pool", mapping="ahp", anchored=None,
                 anchor_all=False, free_from_anchors=False, free_model="ridge", p_free=1.0 / 3.0):
        if level1 not in LEVEL1:
            raise ValueError(f"unknown level1 rule {level1!r}")
        if mapping not in ("ahp", "direct"):
            raise ValueError(f"unknown mapping {mapping!r}")
        if free_model not in ("ridge", "ahpd"):
            raise ValueError(f"unknown judge-free family estimator {free_model!r}")
        if level1.startswith("ahp_pool") and free_from_anchors:     # without pool labels the judge-free evidence comes from the anchors
            level1 = level1.replace("ahp_pool", "ahp")
        self.free, self.p, self.n_ref, self.scale, self.seed = tuple(free), p, n_ref, scale, seed
        self.pi, self.level1, self.mapping, self.anchored = pi, level1, mapping, anchored
        self.anchor_all, self.free_from_anchors = bool(anchor_all), bool(free_from_anchors)
        self.free_model, self.p_free = free_model, p_free

    # ---- judge-free family -------------------------------------------------------------------------------------------
    def _is_free(self, names):
        return [i for i, n in enumerate(names) if any(n == t or n.startswith(t) for t in self.free)]

    def _free_fit(self, Z, y, task):
        if self.free_model == "ridge":
            return np.asarray(RankRidgeCV(seed=self.seed).fit(Z, y).w_, float)
        return np.asarray(SomersAHP(p=self.p_free, n_ref=None, mapping="ahp", scale=self.scale, seed=self.seed).fit(Z, y, task=task).w_, float)

    def _free_key(self):
        return ("ridge",) if self.free_model == "ridge" else (self.free_model, self.p_free, self.scale)

    def pool_direction(self, Z, y, task):
        """Weights of the judge-free family, fitted on the labelled pool rows (Z, y)."""
        key = (self._free_key(), self.seed, task, hash(Z.tobytes()), hash(np.asarray(y, float).tobytes()))
        if key not in UniScore._POOL:
            if len(UniScore._POOL) > 8:
                UniScore._POOL.clear()
            UniScore._POOL[key] = self._free_fit(Z, y, task)
        return UniScore._POOL[key]

    def pool_oof(self, Z, y, task, folds=5):
        """Out-of-fold judge-free scores of every pool row (the family refitted without the fold)."""
        key = ("oof", self._free_key(), self.seed, task, hash(Z.tobytes()), hash(np.asarray(y, float).tobytes()))
        if key not in UniScore._POOL:
            if len(UniScore._POOL) > 8:
                UniScore._POOL.clear()
            from sklearn.model_selection import KFold

            oof = np.zeros(len(y))
            for tr, te in KFold(folds, shuffle=True, random_state=self.seed).split(Z):
                oof[te] = Z[te] @ self._free_fit(Z[tr], y[tr], task)
            UniScore._POOL[key] = oof
        return UniScore._POOL[key]

    # ---- judge family ------------------------------------------------------------------------------------------------
    def _anchor_fit(self, Xa, Xb, nm, seed):
        Z, g = np.vstack([Xa, Xb]), np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
        if self.anchored is None:
            m = SomersAHP(p="auto", n_ref=self.n_ref, mapping=self.mapping, scale=self.scale, seed=seed).fit(Z, g, task="binary", names=nm)
        else:
            m = self.anchored(seed).fit(Z, g, task="binary", names=nm)
        return np.asarray(m.w_, float)

    def _oof_scores(self, Xa, Xb, nm):
        """Out-of-fold scores of the anchors (three repetitions of 5-fold cross-fitting, centred and scaled per fold)."""
        na, nb = len(Xa), len(Xb)
        key = (self.seed, na, nb, self.mapping, self.scale, hash(Xa.tobytes()), hash(Xb.tobytes())) if self.anchored is None else None
        if key is not None and key in UniScore._OOF_ANCHORS:
            return UniScore._OOF_ANCHORS[key]
        f = int(max(2, min(5, na, nb)))
        rng = np.random.RandomState(self.seed + 1)
        oa, ob = np.zeros(na), np.zeros(nb)
        reps = 3
        for r in range(reps):
            pa, pb = rng.permutation(na), rng.permutation(nb)
            for k in range(f):
                ea, eb = pa[k::f], pb[k::f]
                ta, tb = np.setdiff1d(np.arange(na), ea), np.setdiff1d(np.arange(nb), eb)
                if len(ta) < 2 or len(tb) < 2:
                    continue
                w = self._anchor_fit(Xa[ta], Xb[tb], nm, self.seed + 7 * r + k)
                tr_s = np.r_[Xa[ta] @ w, Xb[tb] @ w]
                sd, mu = tail_sd(tr_s, np.arange(len(ta)), len(ta) + np.arange(len(tb))), tr_s.mean()
                oa[ea] += (Xa[ea] @ w - mu) / sd / reps
                ob[eb] += (Xb[eb] @ w - mu) / sd / reps
        if key is not None:
            if len(UniScore._OOF_ANCHORS) > 16:
                UniScore._OOF_ANCHORS.clear()
            UniScore._OOF_ANCHORS[key] = (oa, ob)
        return oa, ob

    # ---- fit -------------------------------------------------------------------------------------------------------------
    def fit(self, X, y, *, task="regression", names=None, judged=None, **kw):
        """Fit on the pool rows (X, y).  `judged` marks the rows with judge scores (default: all rows; elsewhere the judge
        columns may be NaN): the anchors come from the tails of the judged rows, the judge-free family uses every row."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        K = X.shape[1]
        nm = list(names) if names is not None else [f"c{k}" for k in range(K)]
        self.names_ = nm
        F = self._is_free(nm)
        partial = judged is not None and not np.all(judged)
        idx = np.flatnonzero(np.asarray(judged, bool)) if judged is not None else np.arange(len(y))
        self.p_anchor_ = resolve_p(self.p, self.n_ref, len(idx))
        lo_j, hi_j = reference_groups(y[idx], task, self.p, self.n_ref)
        lo, hi = idx[lo_j], idx[hi_j]
        a, b = sample_reference(lo, hi, self.n_ref, np.random.RandomState(self.seed))
        J = [k for k in range(K) if k not in F]
        if not F or not J:
            Xn = np.nan_to_num(X) if partial else X
            self.w_ = self._anchor_fit(Xn[a], Xn[b], nm, self.seed)
            self.pi_ = 1.0
            return self
        JJ = list(range(K)) if self.anchor_all else J
        nmJ = [nm[j] for j in JJ]
        Xj = np.nan_to_num(X[:, JJ]) if partial else X[:, JJ]
        w_j = self._anchor_fit(Xj[a], Xj[b], nmJ, self.seed)
        Xf = X[:, F]
        if self.free_from_anchors:
            nmF = [nm[j] for j in F]
            w_f = self._anchor_fit(Xf[a], Xf[b], nmF, self.seed)
            s_f, s_j = tail_sd(Xf @ w_f, a, b), tail_sd(Xj @ w_j, a, b)
            sp_a, sp_b = self._oof_scores(Xf[a], Xf[b], nmF)
        else:
            w_f = self.pool_direction(Xf, y, task)
            oof = self.pool_oof(Xf, y, task)
            sd_p = tail_sd(oof, lo, hi)
            sp_a, sp_b = oof[a] / sd_p, oof[b] / sd_p
            s_f, s_j = tail_sd(Xf @ w_f, lo, hi), tail_sd(Xj @ w_j, a, b)
        oa, ob = self._oof_scores(Xj[a], Xj[b], nmJ)
        st = S.somers_d_cov(np.c_[sp_a, oa], np.c_[sp_b, ob])
        D = np.asarray(st["D"], float)
        D_pool = float("nan")
        if self.level1.startswith("ahp_pool"):
            # e_F where it is cheap: all pool texts of the label region the anchors are drawn from (no judge call)
            thr_lo, thr_hi = float(np.max(y[lo])), float(np.min(y[hi]))
            loP, hiP = (np.flatnonzero(y <= thr_lo), np.flatnonzero(y >= thr_hi)) if thr_lo < thr_hi else reference_groups(y, task, self.p_anchor_, None)
            D_pool = float(S.somers_d_cov(oof[loP][:, None], oof[hiP][:, None])["D"][0])
            if not D_pool == D_pool:
                raise ValueError("the judge-free family's evidence on the pool is undefined (NaN in a judge-free criterion?)")
        if self.pi is not None:
            alpha = float(self.pi)
        elif self.level1 in ("ahp", "ahp_pool"):
            e = np.maximum(D, 0.0)
            if self.level1 == "ahp_pool":
                e[0] = max(D_pool, 0.0)
            d = e[1] - e[0]
            c = 1.0 + self.scale * d if d >= 0 else 1.0 / (1.0 + self.scale * abs(d))
            alpha = c / (1.0 + c)
        elif self.level1 == "ahp_pool_direct":
            e = np.maximum(D, 0.0)
            e[0] = max(D_pool, 0.0)
            alpha = float(e[1] / e.sum()) if e.sum() > 0 else 0.5
        elif self.level1 == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            Zs = np.r_[np.c_[sp_a, oa], np.c_[sp_b, ob]]
            gg = np.r_[np.zeros(len(a)), np.ones(len(b))]
            cf = np.maximum(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Zs, gg).coef_[0], 0.0)
            alpha = float(cf[1] / cf.sum()) if cf.sum() > 0 else 0.5
        else:
            alpha = 0.5
        w = np.zeros(K)
        w[F] += (1.0 - alpha) * np.asarray(w_f, float) / s_f
        w[JJ] += alpha * np.asarray(w_j, float) / s_j
        self.pi_ = float(alpha)
        self.family_ = dict(e_free=float(D_pool) if self.level1.startswith("ahp_pool") else float(D[0]), e_judge=float(D[1]), alpha=float(alpha), n_anchors=int(len(a) + len(b)))
        self.w_ = w / (np.abs(w).sum() + 1e-12)
        return self

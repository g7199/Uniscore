"""Criterion weighting on the anchors and the fitted estimators used as building blocks and baselines.

Every estimator maps an (n, K) matrix of criterion scores to one score per text (`score`) after `fit(X, y, task=...)`.
The anchors are B/2 texts from each tail of the labels (`weights.reference_groups`, `weights.sample_reference`).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import stats as S
from .weights import ahp_eigen_weights, direct_weights, ev_auc, reference_groups, sample_reference


class Aggregator:
    name = "base"

    def fit(self, X, y, *, task="regression", names=None, **kw):
        raise NotImplementedError

    def score(self, X):
        return np.asarray(X, float) @ self.w_


class SomersAHP(Aggregator):
    """AHP priorities of tested evidence: Somers' D of every criterion between the two anchor tails (DeLong standard
    errors), mapped to a signed AHP priority vector (`mapping="ahp"`) or used as normalised weights (`mapping="direct"`)."""

    name = "AHP on Somers' D"

    def __init__(self, p="auto", n_ref=100, mapping="ahp", scale=8.0, seed=0):
        self.p, self.n_ref, self.mapping, self.scale, self.seed = p, n_ref, mapping, scale, seed

    def fit(self, X, y, *, task="regression", names=None, **kw):
        X = np.asarray(X, float)
        rng = np.random.RandomState(self.seed)
        lo, hi = reference_groups(y, task, self.p, self.n_ref)
        a, b = sample_reference(lo, hi, self.n_ref, rng)
        st = S.somers_d_cov(X[a], X[b])
        self.names_ = list(names) if names is not None else [f"c{k}" for k in range(X.shape[1])]
        self.D_, self.se_, self.cov_ = st["D"], st["se"], st["cov"]
        D = np.asarray(self.D_, float)
        signed = np.where(D >= 0, 1.0, -1.0) * np.abs(D)
        self.cr_ = 0.0
        if len(D) == 1:
            self.w_ = np.sign(signed) * np.ones(1)
        elif self.mapping == "ahp":
            self.w_, self.cr_ = ahp_eigen_weights(signed, self.scale)
        else:
            self.w_ = direct_weights(signed)
        return self

    def audit(self, level: float = 0.95):
        """Evidence table: Somers' D, its DeLong standard error and confidence interval, and the weight of every criterion."""
        import pandas as pd

        lo, hi = S.somers_d_ci(self.D_, self.se_, level)
        return pd.DataFrame({"criterion": self.names_, "D": self.D_, "se": self.se_, "ci_lo": lo, "ci_hi": hi, "weight": self.w_})


class EqualSigned(Aggregator):
    """Equal weights with the signs of the criteria's Somers' D on the anchors."""

    name = "equal weights"

    def __init__(self, p="auto", n_ref=100, seed=0):
        self.p, self.n_ref, self.seed = p, n_ref, seed

    def fit(self, X, y, *, task="regression", names=None, **kw):
        X = np.asarray(X, float)
        rng = np.random.RandomState(self.seed)
        lo, hi = reference_groups(y, task, self.p, self.n_ref)
        a, b = sample_reference(lo, hi, self.n_ref, rng)
        d = np.array([ev_auc(X[a, k], X[b, k]) for k in range(X.shape[1])])
        self.w_ = np.where(d >= 0, 1.0, -1.0) / len(d)
        return self


class AnchorModel(Aggregator):
    """A scikit-learn model fitted on the anchors (tail indicator as the target), e.g. LDA or logistic regression."""

    def __init__(self, factory, name, ref=("auto", 100), seed=0, decision=False):
        self.factory, self.name, self.ref, self.seed, self.decision = factory, name, ref, seed, decision

    def fit(self, X, y, *, task="regression", names=None, **kw):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.RandomState(self.seed)
        p, n_ref = self.ref
        lo, hi = reference_groups(y, task, p, n_ref)
        a, b = sample_reference(lo, hi, n_ref, rng)
        idx = np.r_[a, b]
        self.m_ = self.factory(self.seed)
        self.m_.fit(X[idx], np.r_[np.zeros(len(a)), np.ones(len(b))])
        self.w_ = self._raw_coef()
        return self

    def _raw_coef(self):
        """Coefficients in raw criterion units (linear models)."""
        m = self.m_
        try:
            if hasattr(m, "named_steps"):
                sc = m.named_steps.get("standardscaler")
                lin = list(m.named_steps.values())[-1]
                coef = np.asarray(lin.coef_, float).ravel()
                return coef / sc.scale_ if sc is not None else coef
            if hasattr(m, "coef_"):
                return np.asarray(m.coef_, float).ravel()
        except Exception:
            pass
        return None

    def score(self, X):
        X = np.asarray(X, float)
        if self.decision:
            return np.asarray(self.m_.decision_function(X)).ravel()
        return np.asarray(self.m_.predict(X)).ravel()


def standardized(model_factory):
    """Pipeline factory: standardise the criteria, then fit the model."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return lambda seed: make_pipeline(StandardScaler(), model_factory(seed))


class RankRidgeCV(Aggregator):
    """Ridge regression of the ranked labels on the criteria, penalty chosen by 5-fold cross-validation
    (25 values between 1e-2 and 1e4, log-spaced)."""

    name = "rank-ridge"

    def __init__(self, seed=0, alphas=None, cv=5):
        self.seed, self.alphas, self.cv = seed, alphas, cv

    def fit(self, X, y, **kw):
        from sklearn.linear_model import RidgeCV

        X = np.asarray(X, float)
        r = stats.rankdata(np.asarray(y, float)) / len(y)
        alphas = np.logspace(-2, 4, 25) if self.alphas is None else self.alphas
        self.m_ = RidgeCV(alphas=alphas, cv=self.cv).fit(X, r)
        self.w_ = self.m_.coef_
        return self

    def score(self, X):
        return self.m_.predict(np.asarray(X, float))

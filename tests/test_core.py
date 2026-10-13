import itertools
import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uniscore import SomersAHP, UniScore
from uniscore import stats as S
from uniscore.weights import ahp_eigen_weights, ev_auc


def brute_somers(low, high):
    return np.mean([np.sign(h - l) for h, l in itertools.product(high, low)])


def test_somers_d_matches_brute_force():
    rng = np.random.RandomState(0)
    low, high = rng.randint(1, 6, 15).astype(float), rng.randint(2, 6, 12).astype(float)
    st = S.somers_d_cov(low[:, None], high[:, None])
    assert st["D"][0] == pytest.approx(brute_somers(low, high))
    assert ev_auc(low, high) == pytest.approx(brute_somers(low, high))


def test_delong_variance_matches_bootstrap():
    rng = np.random.RandomState(1)
    low, high = rng.normal(0, 1, 60), rng.normal(0.6, 1, 60)
    se = S.somers_d_cov(low[:, None], high[:, None])["se"][0]
    boot = [brute_somers(rng.choice(low, 60), rng.choice(high, 60)) for _ in range(400)]
    assert se == pytest.approx(np.std(boot), rel=0.2)


def test_wilcoxon_exact_matches_scipy():
    x = np.array([0.012, -0.004, 0.02, 0.009, 0.015, -0.001, 0.03])
    ours = S.wilcoxon_exact(x, "greater")["p"]
    ref = stats.wilcoxon(x, alternative="greater", method="exact").pvalue
    assert ours == pytest.approx(ref)
    assert S.wilcoxon_exact(np.full(5, 0.01), "greater")["p"] == pytest.approx(1 / 32)


def test_ahp_priorities():
    w, cr = ahp_eigen_weights(np.array([0.6, -0.3, 0.1]))
    assert np.abs(w).sum() == pytest.approx(1.0)
    assert np.sign(w).tolist() == [1, -1, 1]
    assert abs(w[0]) > abs(w[1]) > abs(w[2])
    w_eq, _ = ahp_eigen_weights(np.array([0.4, 0.4, 0.4]))
    assert np.allclose(w_eq, 1 / 3)


def synthetic(n=1200, n_judged=300, seed=0):
    rng = np.random.RandomState(seed)
    q = rng.normal(size=n)
    free = np.c_[q + rng.normal(0, 1.5, n), rng.normal(size=n), q + rng.normal(0, 2, n)]
    judge = np.c_[q + rng.normal(0, 0.8, n), q + rng.normal(0, 1.0, n), rng.normal(size=n)]
    judge[n_judged:] = np.nan
    return np.c_[judge, free], q + rng.normal(0, 0.5, n), ["j0", "j1", "j2", "q_0", "q_1", "q_2"]


def test_uniscore_fits_and_mixes_families():
    X, y, names = synthetic()
    judged = ~np.isnan(X[:, 0])
    m = UniScore(n_ref=10, seed=0).fit(X, y, task="regression", names=names, judged=judged)
    assert np.abs(m.w_).sum() == pytest.approx(1.0)
    assert 0.1 <= m.pi_ <= 0.9
    s = m.score(np.nan_to_num(X[judged]))
    assert stats.spearmanr(s, y[judged])[0] > 0.5
    assert m.w_[2] == pytest.approx(0.0, abs=0.1)          # a useless judge criterion gets a small weight


def test_family_only_variants():
    X, y, names = synthetic(seed=1)
    judged = ~np.isnan(X[:, 0])
    pool_only = UniScore(pi=0.0, n_ref=10).fit(X, y, names=names, judged=judged)
    judge_only = UniScore(pi=1.0, n_ref=10).fit(X, y, names=names, judged=judged)
    assert np.allclose(pool_only.w_[:3], 0.0) and np.allclose(judge_only.w_[3:], 0.0)


def test_somers_ahp_audit():
    rng = np.random.RandomState(3)
    y = np.r_[np.zeros(20), np.ones(20)]
    X = np.c_[y + rng.normal(0, 0.5, 40), rng.normal(size=40)]
    m = SomersAHP(n_ref=20).fit(X, y, task="binary", names=["good", "noise"])
    tab = m.audit()
    assert list(tab.criterion) == ["good", "noise"]
    assert tab.ci_lo[0] > 0 and abs(tab.weight[0]) > abs(tab.weight[1])

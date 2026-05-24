#!/usr/bin/env python3
"""Recompute main-table statistics from cached LLM judge scores.

This script uses only files inside this package's `inputs` directory and source
code inside this package's `src` directory. It does not call external LLM APIs. The input
CSVs are cached criterion-level judge responses used for the paper.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, pearsonr, skew, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uniscore.ahp import build_ahp_matrix, compute_ahp_weights_and_consistency, compute_score  # noqa: E402
from uniscore.models import train_tiny_mlp  # noqa: E402


warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1024]
SOFTWARE_METRICS = ["polarity", "expertise", "specificity", "consistency", "word_count_scaled"]
SAMSUM_METRICS = ["coherence", "relevance", "fluency", "word_count_scaled"]
DEP_METRICS = ["negative_affect", "self_focus", "absolutist", "social_isolation"]
ALPHAS = np.logspace(-5, 1, 80)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def ensure_word_count(df: pd.DataFrame) -> None:
    if "word_count" in df.columns:
        return
    text_col = "review_text" if "review_text" in df.columns else "summary"
    df["word_count"] = df[text_col].astype(str).str.split().str.len()


def apply_sqrt_word_scale(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Train-anchored sqrt word-count scaling used in the submitted table."""
    ensure_word_count(train)
    ensure_word_count(test)
    ref = np.sqrt(train["word_count"].astype(float).values)
    mu = float(np.nanmean(ref))
    sd = float(np.nanstd(ref)) or 1.0
    for df in (train, test):
        z = (np.sqrt(df["word_count"].astype(float).values) - mu) / sd
        df["word_count_scaled"] = np.clip(3.0 + z, 1.0, 5.0)


def score_stats(score: np.ndarray) -> tuple[float, float]:
    score = np.asarray(score, dtype=float)
    return float(np.std(score) / (abs(np.mean(score)) + 1e-9)), float(skew(score))


def signed_js(low: np.ndarray, high: np.ndarray, bins: int = 10) -> float:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    edges = np.histogram_bin_edges(np.r_[low, high], bins=bins)
    pl, _ = np.histogram(low, bins=edges, density=True)
    ph, _ = np.histogram(high, bins=edges, density=True)
    pl = pl.astype(float) + 1e-10
    ph = ph.astype(float) + 1e-10
    pl /= pl.sum()
    ph /= ph.sum()
    sign = np.sign(high.mean() - low.mean()) or 1.0
    return float(sign * (jensenshannon(pl, ph, base=2) ** 2))


def dist_from_groups(low: pd.DataFrame, high: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    return {m: signed_js(low[m].dropna().values, high[m].dropna().values) for m in metrics}


def ahp_weights(dist: dict[str, float]) -> dict[str, float]:
    mat = build_ahp_matrix(dist)
    weights, _, _, _ = compute_ahp_weights_and_consistency(mat.values)
    named = dict(zip(mat.index.tolist(), weights))
    return {k: (named[k] if dist[k] >= 0 else -named[k]) for k in mat.index}


def uniscore_continuous(
    train: pd.DataFrame, test: pd.DataFrame, metrics: list[str], p: float
) -> list[dict[str, float]]:
    y = test["vote"].astype(float).values
    low_thr = train["vote"].quantile(p)
    high_thr = train["vote"].quantile(1 - p)
    low_pool = train[train["vote"] <= low_thr]
    high_pool = train[train["vote"] >= high_thr]
    rows = []
    for seed in SEEDS:
        bag = {m: [] for m in metrics}
        for s in [seed, seed + 11, seed + 23, seed + 37, seed + 51]:
            low = low_pool.sample(min(100, len(low_pool)), random_state=s)
            high = high_pool.sample(min(100, len(high_pool)), random_state=s)
            piece = dist_from_groups(low, high, metrics)
            for m in metrics:
                bag[m].append(piece[m])
        dist = {}
        for m, vals_m in bag.items():
            arr = np.asarray(vals_m, dtype=float)
            val = float(arr.mean())
            val *= abs(float(np.sign(arr).mean()))
            val *= max(0.0, 1.0 - float(arr.std()) / (abs(val) + 1e-6))
            dist[m] = val
        weights = ahp_weights(dist)
        score = test.apply(lambda row: compute_score(row, weights, metrics), axis=1).values
        ncv, sk = score_stats(score)
        rows.append(
            {
                "spearman": float(spearmanr(y, score)[0]),
                "kendall": float(kendalltau(y, score)[0]),
                "pearson": float(pearsonr(y, score)[0]),
                "ncv": ncv,
                "skew": sk,
            }
        )
    return rows


def uniscore_depression(
    train_high: pd.DataFrame, train_low: pd.DataFrame, test: pd.DataFrame, n: int
) -> list[dict[str, float]]:
    train = pd.concat([train_high, train_low], ignore_index=True)
    y_train = train["label"].astype(int).values
    y_test = test["label"].astype(int).values
    rows = []
    for seed in SEEDS:
        sample_n = min(n, len(train_high), len(train_low))
        high = train_high.sample(sample_n, random_state=seed)
        low = train_low.sample(sample_n, random_state=seed)
        weights = ahp_weights(dist_from_groups(low, high, DEP_METRICS))
        train_score = train.apply(lambda row: compute_score(row, weights, DEP_METRICS), axis=1).values
        thr = (train_score[y_train == 0].mean() + train_score[y_train == 1].mean()) / 2.0
        test_score = test.apply(lambda row: compute_score(row, weights, DEP_METRICS), axis=1).values
        pred = (test_score >= thr).astype(int)
        ncv, sk = score_stats(test_score)
        rows.append(
            {
                "f1": float(f1_score(y_test, pred)),
                "acc": float(accuracy_score(y_test, pred)),
                "ncv": ncv,
                "skew": sk,
            }
        )
    return rows


def optional_xgb_regressor(seed: int):
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def lasso_models(seed: int):
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    return {
        "LassoCV": make_pipeline(
            StandardScaler(),
            LassoCV(alphas=ALPHAS, cv=cv, max_iter=100000, random_state=seed),
        ),
        "ElasticNetCV": make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                alphas=ALPHAS,
                l1_ratio=L1_RATIOS,
                cv=cv,
                max_iter=100000,
                random_state=seed,
            ),
        ),
    }


def continuous_baselines(
    train: pd.DataFrame, test: pd.DataFrame, metrics: list[str], methods: list[str]
) -> dict[str, list[dict[str, float]]]:
    x_train = train[metrics].astype(float).values
    y_train = train["vote"].astype(float).values
    x_test = test[metrics].astype(float).values
    y_test = test["vote"].astype(float).values
    out = {m: [] for m in methods}
    for seed in SEEDS:
        models = {}
        if "Random Weight" in methods:
            rng = np.random.RandomState(seed)
            w = rng.uniform(-1, 1, len(metrics))
            models["Random Weight"] = x_test.dot(w)
        if "Regression" in methods:
            models["Regression"] = LinearRegression().fit(x_train, y_train).predict(x_test)
        if "Random Forest" in methods:
            models["Random Forest"] = (
                RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
                .fit(x_train, y_train)
                .predict(x_test)
            )
        if "Neural Network" in methods:
            _, nn = train_tiny_mlp(x_train, y_train, seed=seed)
            models["Neural Network"] = nn(x_test)
        if "SVR" in methods:
            models["SVR"] = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1, kernel="rbf")).fit(
                x_train, y_train
            ).predict(x_test)
        if "XGBoost" in methods:
            xgb = optional_xgb_regressor(seed)
            if xgb is not None:
                models["XGBoost"] = xgb.fit(x_train, y_train).predict(x_test)
        for name, model in lasso_models(seed).items():
            if name in methods:
                models[name] = model.fit(x_train, y_train).predict(x_test)
        for name, score in models.items():
            ncv, sk = score_stats(score)
            out[name].append(
                {
                    "spearman": float(spearmanr(y_test, score)[0]),
                    "kendall": float(kendalltau(y_test, score)[0]),
                    "pearson": float(pearsonr(y_test, score)[0]),
                    "ncv": ncv,
                    "skew": sk,
                }
            )
    return out


def depression_baselines(
    train_high: pd.DataFrame, train_low: pd.DataFrame, test: pd.DataFrame, methods: list[str]
) -> dict[str, list[dict[str, float]]]:
    train = pd.concat([train_high, train_low], ignore_index=True)
    x_train = train[DEP_METRICS].astype(float).values
    y_train = train["label"].astype(int).values
    x_test = test[DEP_METRICS].astype(float).values
    y_test = test["label"].astype(int).values
    out = {m: [] for m in methods}
    for seed in SEEDS:
        scores = {}
        if "Random Weight" in methods:
            rng = np.random.RandomState(seed)
            w = rng.uniform(-1, 1, len(DEP_METRICS))
            scores["Random Weight"] = (x_train.dot(w), x_test.dot(w))
        if "Regression" in methods:
            m = LinearRegression().fit(x_train, y_train)
            scores["Regression"] = (m.predict(x_train), m.predict(x_test))
        if "LassoCV" in methods or "ElasticNetCV" in methods:
            for name, model in lasso_models(seed).items():
                if name in methods:
                    fitted = model.fit(x_train, y_train)
                    scores[name] = (fitted.predict(x_train), fitted.predict(x_test))
        if "SVR" in methods:
            m = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1)).fit(x_train, y_train)
            scores["SVR"] = (m.predict(x_train), m.predict(x_test))
        if "XGBoost" in methods:
            xgb = optional_xgb_regressor(seed)
            if xgb is not None:
                fitted = xgb.fit(x_train, y_train)
                scores["XGBoost"] = (fitted.predict(x_train), fitted.predict(x_test))
        if "Random Forest" in methods:
            m = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1).fit(x_train, y_train)
            scores["Random Forest"] = (m.predict(x_train), m.predict(x_test))
        if "Neural Network" in methods:
            _, nn = train_tiny_mlp(x_train, y_train, seed=seed)
            scores["Neural Network"] = (nn(x_train), nn(x_test))
        for name, (train_score, test_score) in scores.items():
            thr = (train_score[y_train == 0].mean() + train_score[y_train == 1].mean()) / 2.0
            pred = (test_score >= thr).astype(int)
            ncv, sk = score_stats(test_score)
            out[name].append(
                {
                    "f1": float(f1_score(y_test, pred)),
                    "acc": float(accuracy_score(y_test, pred)),
                    "ncv": ncv,
                    "skew": sk,
                }
            )
    return out


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        out[col] = float(df[col].mean())
        out[col + "_std"] = float(df[col].std(ddof=0))
    return out


def add_row(accum: list[dict[str, object]], dataset: str, method: str, rows: list[dict[str, float]]) -> None:
    accum.append({"dataset": dataset, "method": method, **summarize(rows)})


def main() -> None:
    out_dir = ROOT / "generated"
    out_dir.mkdir(exist_ok=True)

    software_train = pd.read_csv(ROOT / "inputs/software_qwen3_1_7b/train_results_with_scores.csv")
    software_test = pd.read_csv(ROOT / "inputs/software_qwen3_1_7b/test_results_with_scores.csv")
    apply_sqrt_word_scale(software_train, software_test)

    samsum_train = pd.read_csv(ROOT / "inputs/samsum/train_results_with_scores.csv")
    samsum_test = pd.read_csv(ROOT / "inputs/samsum/test_results_with_scores.csv")
    apply_sqrt_word_scale(samsum_train, samsum_test)

    dep_high = pd.read_csv(ROOT / "inputs/depression/train_high_scored.csv")
    dep_low = pd.read_csv(ROOT / "inputs/depression/train_low_scored.csv")
    dep_test = pd.read_csv(ROOT / "inputs/depression/test_results_with_scores.csv")

    rows: list[dict[str, object]] = []

    continuous_methods = [
        "Random Weight",
        "Regression",
        "LassoCV",
        "ElasticNetCV",
        "SVR",
        "XGBoost",
        "Random Forest",
        "Neural Network",
    ]
    for name, train, test, metrics in [
        ("Software", software_train, software_test, SOFTWARE_METRICS),
        ("SAMSum", samsum_train, samsum_test, SAMSUM_METRICS),
    ]:
        for method, vals in continuous_baselines(train, test, metrics, continuous_methods).items():
            add_row(rows, name, method, vals)

    for p in [0.10, 0.05, 0.025]:
        add_row(rows, "Software", f"UniScore p={p:g}", uniscore_continuous(software_train, software_test, SOFTWARE_METRICS, p))
    for p in [0.10, 0.20, 0.25]:
        add_row(rows, "SAMSum", f"UniScore p={p:g}", uniscore_continuous(samsum_train, samsum_test, SAMSUM_METRICS, p))

    dep_methods = [
        "Random Weight",
        "Regression",
        "LassoCV",
        "ElasticNetCV",
        "SVR",
        "XGBoost",
        "Random Forest",
        "Neural Network",
    ]
    for method, vals in depression_baselines(dep_high, dep_low, dep_test, dep_methods).items():
        add_row(rows, "Depression", method, vals)
    for n in [50, 100, 200]:
        add_row(rows, "Depression", f"UniScore n={n}", uniscore_depression(dep_high, dep_low, dep_test, n))

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "recomputed_main_numbers.csv", index=False)
    print(raw.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

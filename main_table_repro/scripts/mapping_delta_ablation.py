#!/usr/bin/env python3
"""AHP Saaty-scale mapping ablation on the cached main-table inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from recompute_main_numbers import (  # noqa: E402
    DEP_METRICS,
    SAMSUM_METRICS,
    SEEDS,
    SOFTWARE_METRICS,
    apply_sqrt_word_scale,
    dist_from_groups,
    score_stats,
)
from uniscore.ahp import compute_ahp_weights_and_consistency, compute_score  # noqa: E402


def build_ahp_matrix(dist: dict[str, float], kind: str) -> pd.DataFrame:
    keys = list(dist.keys())
    mat = np.ones((len(keys), len(keys)), dtype=float)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                continue
            delta = abs(dist[ki]) - abs(dist[kj])
            d = abs(delta)
            if kind == "linear":
                scale = 1 + 8 * d
            elif kind == "quadratic":
                scale = 1 + 8 * d**2
            elif kind == "cubic":
                scale = 1 + 8 * d**3
            elif kind == "sqrt":
                scale = 1 + 8 * np.sqrt(d)
            elif kind == "exp":
                scale = 9**d
            elif kind == "log2":
                scale = 1 + 8 * np.log2(1 + d)
            else:
                raise ValueError(kind)
            scale = float(np.clip(scale, 1.0, 9.0))
            mat[i, j] = scale if delta >= 0 else 1.0 / scale
    return pd.DataFrame(mat, index=keys, columns=keys)


def weights_for(dist: dict[str, float], kind: str) -> tuple[dict[str, float], float]:
    mat = build_ahp_matrix(dist, kind)
    w, _, _, cr = compute_ahp_weights_and_consistency(mat.values)
    named = dict(zip(mat.index.tolist(), w))
    signed = {k: (named[k] if dist[k] >= 0 else -named[k]) for k in mat.index}
    return signed, float(cr)


def stable_dist(train: pd.DataFrame, metrics: list[str], p: float, seed: int) -> dict[str, float]:
    low_thr = train["vote"].quantile(p)
    high_thr = train["vote"].quantile(1 - p)
    low_pool = train[train["vote"] <= low_thr]
    high_pool = train[train["vote"] >= high_thr]
    bag = {m: [] for m in metrics}
    for s in [seed, seed + 11, seed + 23, seed + 37, seed + 51]:
        low = low_pool.sample(min(100, len(low_pool)), random_state=s)
        high = high_pool.sample(min(100, len(high_pool)), random_state=s)
        piece = dist_from_groups(low, high, metrics)
        for m in metrics:
            bag[m].append(piece[m])
    out = {}
    for m, vals in bag.items():
        arr = np.asarray(vals, dtype=float)
        val = float(arr.mean())
        val *= abs(float(np.sign(arr).mean()))
        val *= max(0.0, 1.0 - float(arr.std()) / (abs(val) + 1e-6))
        out[m] = val
    return out


def eval_continuous(train: pd.DataFrame, test: pd.DataFrame, metrics: list[str], p: float, kind: str) -> list[dict[str, float]]:
    y = test["vote"].astype(float).values
    rows = []
    for seed in SEEDS:
        dist = stable_dist(train, metrics, p, seed)
        weights, cr = weights_for(dist, kind)
        score = test.apply(lambda row: compute_score(row, weights, metrics), axis=1).values
        ncv, sk = score_stats(score)
        rows.append({"score": float(spearmanr(y, score)[0]), "cr": cr, "ncv": ncv, "skew": sk})
    return rows


def eval_dep(high_df: pd.DataFrame, low_df: pd.DataFrame, test: pd.DataFrame, n: int, kind: str) -> list[dict[str, float]]:
    train = pd.concat([high_df, low_df], ignore_index=True)
    train_y = train["label"].astype(int).values
    test_y = test["label"].astype(int).values
    rows = []
    for seed in SEEDS:
        sample_n = min(n, len(high_df), len(low_df))
        high = high_df.sample(sample_n, random_state=seed)
        low = low_df.sample(sample_n, random_state=seed)
        dist = dist_from_groups(low, high, DEP_METRICS)
        weights, cr = weights_for(dist, kind)
        train_score = train.apply(lambda row: compute_score(row, weights, DEP_METRICS), axis=1).values
        thr = (train_score[train_y == 0].mean() + train_score[train_y == 1].mean()) / 2.0
        test_score = test.apply(lambda row: compute_score(row, weights, DEP_METRICS), axis=1).values
        pred = (test_score >= thr).astype(int)
        ncv, sk = score_stats(test_score)
        rows.append({"score": float(f1_score(test_y, pred)), "cr": cr, "ncv": ncv, "skew": sk})
    return rows


def summarize(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    df = pd.DataFrame(rows)
    return {
        prefix: float(df["score"].mean()),
        prefix + "_std": float(df["score"].std(ddof=0)),
        prefix + "_cr": float(df["cr"].mean()),
        prefix + "_ncv": float(df["ncv"].mean()),
        prefix + "_skew": float(df["skew"].mean()),
    }


def main() -> None:
    sw_train = pd.read_csv(ROOT / "inputs/software_qwen3_1_7b/train_results_with_scores.csv")
    sw_test = pd.read_csv(ROOT / "inputs/software_qwen3_1_7b/test_results_with_scores.csv")
    apply_sqrt_word_scale(sw_train, sw_test)
    sam_train = pd.read_csv(ROOT / "inputs/samsum/train_results_with_scores.csv")
    sam_test = pd.read_csv(ROOT / "inputs/samsum/test_results_with_scores.csv")
    apply_sqrt_word_scale(sam_train, sam_test)
    dep_high = pd.read_csv(ROOT / "inputs/depression/train_high_scored.csv")
    dep_low = pd.read_csv(ROOT / "inputs/depression/train_low_scored.csv")
    dep_test = pd.read_csv(ROOT / "inputs/depression/test_results_with_scores.csv")

    labels = {
        "linear": "1+8|Delta|",
        "quadratic": "1+8|Delta|^2",
        "cubic": "1+8|Delta|^3",
        "sqrt": "1+8sqrt(|Delta|)",
        "exp": "9^|Delta|",
        "log2": "1+8log2(1+|Delta|)",
    }
    rows = []
    for kind, label in labels.items():
        row = {"mapping": kind, "label": label}
        row.update(summarize(eval_continuous(sw_train, sw_test, SOFTWARE_METRICS, 0.025, kind), "Software"))
        row.update(summarize(eval_continuous(sam_train, sam_test, SAMSUM_METRICS, 0.25, kind), "SAMSum"))
        row.update(summarize(eval_dep(dep_high, dep_low, dep_test, 200, kind), "Depression"))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "generated/mapping_delta_ablation.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

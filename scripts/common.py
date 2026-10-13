"""Data loading shared by the experiment scripts."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
RESULTS = os.environ.get("UNISCORE_RESULTS", os.path.join(ROOT, "results"))

DATASETS = ["summeval", "helpsteer2", "asap_aes", "dreaddit", "civil_comments"]
TASK = {"summeval": "regression", "helpsteer2": "regression", "asap_aes": "regression", "dreaddit": "binary", "civil_comments": "regression"}
# the seven judges of the main texts: (code, directory under data/judges, label)
JUDGES = [("Q", "qwen3-1.7b", "Qwen3-1.7B"), ("G", "gemma-4-e4b", "Gemma 4 E4B"), ("M", "gemma-4-26b-a4b", "Gemma 4 26B-A4B"), ("L", "gpt-6-luna", "GPT-6 Luna"),
          ("H", "claude-haiku-4.5", "Claude Haiku 4.5"), ("S", "claude-sonnet-5.5", "Claude Sonnet 5.5"), ("X", "gpt-6.1-sol", "GPT-6.1 Sol")]
JUDGE_DIR = {c: d for c, d, _ in JUDGES}
# score type of each judge: "expected" (probability-weighted 1-5 score from the next-token distribution) or "integer" (the answer itself; hosted models without token probabilities)
SCORE_TYPE = {"Q": "expected", "G": "expected", "M": "expected", "L": "expected", "H": "integer", "S": "integer", "X": "integer"}
BUDGETS = [10, 20, 40, 100]
N_SPLITS = 30
POOL_LABELS = 1000      # labelled pool texts per split


def load_pool(ds):
    """Pool of a dataset: id, group, target and the judge-free criteria (columns q_*, e_*)."""
    df = pd.read_csv(os.path.join(DATA, "pools", f"{ds}.csv.gz"), dtype={"id": str, "group": str}, float_precision="round_trip")
    if df["target"].isna().all():
        f = os.path.join(DATA, "labels", f"{ds}.csv.gz")
        if not os.path.exists(f):
            raise SystemExit(f"no labels for {ds}: run scripts/fetch_labels.py first (see README)")
        lab = pd.read_csv(f, dtype={"id": str}, float_precision="round_trip")
        df = df.drop(columns="target").merge(lab[["id", "target"]], on="id", how="left")
        if df["target"].isna().any():
            raise SystemExit(f"{int(df['target'].isna().sum())} pool texts of {ds} have no label in {f}")
    free = [c for c in df.columns if c.startswith(("q_", "e_"))]
    return df, free


def load_judge(ds, code, df=None):
    """The pool with one judge's criteria (NaN for the texts the judge did not score) and its holistic score.
    Returns (df, judge criteria columns, holistic column)."""
    if df is None:
        df, _ = load_pool(ds)
    js = pd.read_csv(os.path.join(DATA, "judges", JUDGE_DIR[code], f"{ds}.csv.gz"), dtype={"id": str}, float_precision="round_trip")
    crit = [c for c in js.columns if c not in ("id", "overall")]
    js = js.rename(columns={c: f"{code}:{c}" for c in crit + ["overall"]})
    df = df.merge(js, on="id", how="left")
    return df, [f"{code}:{c}" for c in crit], f"{code}:overall"


def restrict_labels(i, split, X0, judged_col, L=POOL_LABELS):
    """Labelled pool of split i: its judge-scored training texts plus random unjudged training texts up to L in total
    (one permutation per split, identical for every judge and method)."""
    tr = np.asarray(split["train"])
    judged = ~np.isnan(X0[:, judged_col])
    tr_j, tr_u = tr[judged[tr]], tr[~judged[tr]]
    perm = np.random.RandomState(20_000 + i).permutation(len(tr_u))
    sel = np.r_[tr_j, tr_u[perm[:max(L - len(tr_j), 0)]]]
    out = dict(split)
    out["train"] = tr[np.isin(tr, sel)]
    return out

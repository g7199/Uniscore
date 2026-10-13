#!/usr/bin/env python3
"""Labels of the two datasets whose terms do not allow redistribution (written to data/labels/<dataset>.csv.gz).

  ASAP essays (Kaggle, Hewlett Foundation): download training_set_rel3.tsv from
      https://www.kaggle.com/c/asap-aes/data  and pass its path with --asap.
      Target: domain1_score min-max normalised within each essay set (observed range).
  Dreaddit (Turcan & McKeown, 2019): downloaded from the Hugging Face copy andreagasparini/dreaddit (pinned revision).
      Target: the binary stress label.

  python scripts/fetch_labels.py --asap /path/to/training_set_rel3.tsv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from common import DATA

DREADDIT_REPO, DREADDIT_REV = "andreagasparini/dreaddit", "f6ef6d27b23bee7d8be1d2e63a9df9f354da950a"


def asap(tsv):
    raw = pd.read_csv(tsv, sep="\t", encoding="latin-1", dtype=str, keep_default_na=False)
    num = lambda c: pd.to_numeric(raw[c].replace("", np.nan))
    raw["essay_set"] = num("essay_set").astype("int64")
    raw["domain1_score"] = num("domain1_score")
    raw = raw[raw["domain1_score"].notna()].reset_index(drop=True)
    norm = np.empty(len(raw))
    for n, g in raw.groupby("essay_set")["domain1_score"]:
        lo, hi = float(g.min()), float(g.max())
        m = (raw["essay_set"] == n).to_numpy()
        norm[m] = (raw.loc[m, "domain1_score"].to_numpy() - lo) / (hi - lo)
    return pd.DataFrame({"id": "asap_aes:" + raw["essay_id"].astype(str), "target": norm})


def dreaddit():
    from huggingface_hub import hf_hub_download

    frames = []
    for split in ("train", "test"):
        p = hf_hub_download(DREADDIT_REPO, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=DREADDIT_REV)
        d = pd.read_parquet(p)
        frames.append(pd.DataFrame({"id": "dreaddit:" + split + ":" + d["id"].astype(str), "target": d["label"].astype(int).astype(float)}))
    return pd.concat(frames, ignore_index=True)


def write(ds, lab):
    pool = pd.read_csv(os.path.join(DATA, "pools", f"{ds}.csv.gz"), dtype={"id": str}, usecols=["id"])
    lab = lab[lab["id"].isin(set(pool["id"]))].drop_duplicates("id")
    missing = len(set(pool["id"]) - set(lab["id"]))
    if missing:
        raise SystemExit(f"{ds}: {missing} pool texts have no label in the source file")
    os.makedirs(os.path.join(DATA, "labels"), exist_ok=True)
    lab.to_csv(os.path.join(DATA, "labels", f"{ds}.csv.gz"), index=False, float_format="%.17g")
    print(f"{ds}: {len(lab)} labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asap", help="path to training_set_rel3.tsv of the Kaggle ASAP-AES competition")
    ap.add_argument("--skip-dreaddit", action="store_true")
    a = ap.parse_args()
    if a.asap:
        write("asap_aes", asap(a.asap))
    else:
        print("asap_aes: skipped (pass --asap)")
    if not a.skip_dreaddit:
        write("dreaddit", dreaddit())


if __name__ == "__main__":
    main()

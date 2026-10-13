#!/usr/bin/env python3
"""A panel of judges (Table 3): UniScore with the criteria of every judge of the panel as its judge family, against the
same-input alternatives, on the texts every judge of the panel scored.  Writes results/panel/<panel>.csv.

  python scripts/run_panel.py --panel QGMLHSX
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import BUDGETS, DATASETS, N_SPLITS, RESULTS, SCORE_TYPE, TASK, load_judge, load_pool, restrict_labels
from run_single import anchor_models

from uniscore import UniScore
from uniscore import stats as S
from uniscore.evaluate import grouped_splits, quantile_scale, spearman
from uniscore.weights import reference_groups, sample_reference


def load_panel(ds, panel):
    df, free = load_pool(ds)
    jc_all, hol_all = [], []
    for code in panel:
        df, jc, hol = load_judge(ds, code, df)
        jc_all += jc
        hol_all.append(hol)
    return df, jc_all, free, hol_all


def one_split(i, split, X0, y, jidx, free_idx, hidx, names, task):
    split = restrict_labels(i, split, X0, jidx[0])
    tr, te = split["train"], split["test"]
    X = X0.copy()
    for j in free_idx:
        X[:, j] = quantile_scale(X0[tr, j], X0[:, j])
    judged = ~np.isnan(X[:, jidx]).any(axis=1)          # texts every judge of the panel scored
    tr_j, te_j = tr[judged[tr]], te[judged[te]]
    cols = np.r_[jidx, free_idx].astype(int)
    nmc = [names[c] for c in cols]
    rows = []
    rec = lambda B, m, s: rows.append(dict(split=i, B=B, method=m, spearman=spearman(y[te_j], np.asarray(s, float))))
    for B in BUDGETS:
        h = max(2, B // 2)
        A = anchor_models(h)
        m = UniScore(n_ref=h, seed=i).fit(X[tr][:, cols], y[tr], task=task, names=nmc, judged=judged[tr])
        rec(B, "uniscore", m.score(np.nan_to_num(X[te_j][:, cols])))
        # the judge-free family's score: cross-fitted for the anchors, on the test texts
        mp = UniScore(pi=0.0, n_ref=h, seed=i).fit(X[tr][:, cols], y[tr], task=task, names=nmc, judged=judged[tr])
        Xf_tr = np.asarray(X[tr][:, free_idx], float)
        w_f = np.asarray(mp.pool_direction(Xf_tr, y[tr], task), float)
        p_oof = (np.asarray(mp.pool_oof(Xf_tr, y[tr], task), float) / (np.abs(w_f).sum() + 1e-12))[judged[tr]]
        pte = mp.score(np.nan_to_num(X[te_j][:, cols]))

        def stack(key, base, Ztr, Zte, nm):
            mm = base.fit(np.c_[Ztr, p_oof], y[tr_j], task=task, names=list(nm) + ["pool"])
            rec(B, key, mm.score(np.c_[Zte, pte]))

        # all panel criteria and the pool score stacked on the anchors
        nmj = [names[c] for c in jidx]
        for key, base in (("stack_lda", "lda"), ("stack_nb", "nb"), ("stack_logit", "logit")):
            stack(key, A[base](i), X[tr_j][:, jidx], X[te_j][:, jidx], nmj)
        # label-free panel scores stacked with the pool score; every statistic of the panel criteria is computed on the anchors only
        from sklearn.decomposition import FactorAnalysis

        lo, hi = reference_groups(y[tr_j], task, "auto", h)
        a, b = sample_reference(lo, hi, h, np.random.RandomState(i))
        anc = tr_j[np.r_[a, b]]
        Zj = X[:, jidx]
        mu, sd = np.nanmean(Zj[anc], 0), np.nanstd(Zj[anc], 0) + 1e-9
        Zs = np.nan_to_num((Zj - mu) / sd)
        sg = np.sign(S.somers_d_cov(Zs[tr_j][a], Zs[tr_j][b])["D"])
        sg[sg == 0] = 1
        eq = Zs @ sg / len(sg)
        fa = FactorAnalysis(n_components=1, random_state=i).fit(Zs[anc])
        fs = Zs @ fa.components_[0]
        if S.somers_d_cov(fs[tr_j][a][:, None], fs[tr_j][b][:, None])["D"][0] < 0:
            fs = -fs
        judges = {}
        for k, c in enumerate(jidx):
            judges.setdefault(names[c].split(":", 1)[0], []).append(k)
        JM = np.c_[[Zs[:, g] @ sg[g] / len(g) for g in judges.values()]].T          # each judge's mean signed criterion
        for key, base in (("judgemeans_lda", "lda"), ("judgemeans_logit", "logit")):
            stack(key, A[base](i), JM[tr_j], JM[te_j], [f"judge{q}" for q in range(JM.shape[1])])
        crit = {}
        for k, c in enumerate(jidx):
            code, name = names[c].split(":", 1)
            crit.setdefault((name, SCORE_TYPE[code]), []).append(k)
        CA = np.c_[[Zs[:, g].mean(axis=1) for g in crit.values()]].T                    # every criterion averaged over the judges of the same score type
        for key, base in (("critavg_lda", "lda"), ("critavg_logit", "logit")):
            stack(key, A[base](i), CA[tr_j], CA[te_j], [f"{n}/{t}" for n, t in crit])
        for key, sc in (("equal_lda", eq), ("factor_lda", fs)):
            stack(key, A["lda"](i), sc[tr_j][:, None], sc[te_j][:, None], ["panel"])
        hm = np.nanmean(X[:, list(hidx)], axis=1)                                      # the jury: the panel's mean holistic score
        for key, base in (("jury_lda", "lda"), ("jury_logit", "logit")):
            stack(key, A[base](i), hm[tr_j][:, None], hm[te_j][:, None], ["jury"])
    return rows


def main():
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="QGMLHSX", help="judge codes")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = os.path.join(RESULTS, "panel")
    os.makedirs(out, exist_ok=True)
    t0, tasks = time.time(), []
    for ds in a.datasets.split(","):
        df, jc, free, hols = load_panel(ds, a.panel)
        X0 = df[jc + free + hols].to_numpy(float)
        y = df["target"].to_numpy(float)
        jidx = list(range(len(jc)))
        free_idx = list(range(len(jc), len(jc) + len(free)))
        hidx = list(range(len(jc) + len(free), len(jc) + len(free) + len(hols)))
        for i, sp in enumerate(grouped_splits(df["group"].to_numpy(), n_splits=a.splits, seed=0)):
            tasks.append((ds, (i, sp, X0, y, jidx, free_idx, hidx, jc + free + hols, TASK[ds])))
    res = Parallel(n_jobs=a.jobs)(delayed(one_split)(*t) for _, t in tasks)
    r = pd.concat([pd.DataFrame(rows).assign(dataset=ds) for (ds, _), rows in zip(tasks, res)])
    r.to_csv(os.path.join(out, f"{a.panel}.csv"), index=False)
    print(f"panel {a.panel}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

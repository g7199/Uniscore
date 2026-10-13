#!/usr/bin/env python3
"""One judge at a time: UniScore, its ablation variants and every baseline on the main texts (Tables 2, 4 and 5).

For each dataset and judge: 30 grouped 60/10/30 splits of the pool; per split 1,000 labelled pool texts (the judged training
texts and random unjudged ones); the test texts are the judged texts of the test role; for each budget B the anchors are B/2
judged training texts from each label tail, the same for every method.  Writes results/single[_ahpd]/<judge>__<dataset>.csv
with one Spearman correlation per (split, budget, method).

  python scripts/run_single.py                      # judge-free family fitted by rank-ridge (all methods)
  python scripts/run_single.py --free-model ahpd    # judge-free family weighted by AHP priorities of its criteria's Somers' D
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import BUDGETS, DATASETS, JUDGES, N_SPLITS, RESULTS, TASK, load_judge, load_pool, restrict_labels

from uniscore import AnchorModel, EqualSigned, RankRidgeCV, SomersAHP, UniScore, standardized
from uniscore import stats as S
from uniscore.evaluate import grouped_splits, quantile_scale, spearman
from uniscore.weights import reference_groups, sample_reference


def anchor_models(h):
    """Estimators fitted on the anchors (B/2 per tail of the judged training texts)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV

    ref = ("auto", h)
    return {
        "ahp": lambda s: SomersAHP(p="auto", n_ref=h, seed=s),
        "equal": lambda s: EqualSigned(p="auto", n_ref=h, seed=s),
        "lda": lambda s: AnchorModel(lambda t: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "LDA", ref=ref, seed=s, decision=True),
        "nb": lambda s: AnchorModel(standardized(lambda t: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=1.0)), "naive Bayes", ref=ref, seed=s, decision=True),
        "logit": lambda s: AnchorModel(standardized(lambda t: LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=3, scoring="roc_auc", max_iter=3000)), "logistic", ref=ref, seed=s, decision=True),
        "pls": lambda s: AnchorModel(standardized(lambda t: PLSRegression(n_components=1, scale=False)), "PLS-1", ref=ref, seed=s),
        "ridge": lambda s: AnchorModel(standardized(lambda t: RidgeCV(alphas=np.logspace(-3, 3, 13))), "ridge", ref=ref, seed=s),
    }


def uniscore_variants(h, free_model, full):
    """UniScore and the variants of the ablation (each changes one component)."""
    U = lambda **kw: (lambda s: UniScore(n_ref=h, seed=s, free_model=free_model, **kw))
    V = {"uniscore": U(), "pool_only": U(pi=0.0)}
    if full:
        A = anchor_models(h)
        V.update({
            "judge_only": U(pi=1.0),
            "abl_equal_judge": U(anchored=lambda sd: A["equal"](sd)),
            "abl_lda_judge": U(anchored=lambda sd: A["lda"](sd)),
            "abl_lda_family": U(level1="lda"),
            "abl_one_vector": U(anchor_all=True),
            "abl_equal_family": U(level1="equal"),
            "abl_ef_anchors": U(level1="ahp"),
            "abl_no_map": U(mapping="direct", level1="ahp_pool_direct"),
            "abl_median_anchors": lambda s: UniScore(n_ref=h, seed=s, p=0.5),
            "abl_ahp_pool": lambda s: UniScore(n_ref=h, seed=s, free_model="ahpd"),
            "abl_no_pool_labels": U(free_from_anchors=True, level1="ahp"),
        })
    return V


def one_split(i, split, X0, y, jidx, free_idx, hol, task, full, free_model):
    split = restrict_labels(i, split, X0, jidx[0])
    tr, te = split["train"], split["test"]
    X = X0.copy()
    for j in free_idx:      # judge-free criteria on the 1-5 scale with the training pool's empirical CDF (no labels)
        X[:, j] = quantile_scale(X0[tr, j], X0[:, j])
    judged = ~np.isnan(X[:, jidx[0]])
    tr_j, te_j = tr[judged[tr]], te[judged[te]]
    cols = np.r_[jidx, free_idx].astype(int)
    jcols = np.array(jidx, int)
    names = [f"j{k}" for k in range(len(jidx))] + [f"q_{k}" for k in range(len(free_idx))]
    nmj = names[:len(jidx)]
    rows = []
    for B in BUDGETS:
        h = max(2, B // 2)
        A = anchor_models(h)
        res = {}
        # UniScore and its variants: fitted on the labelled pool, anchors from the judged texts
        for key, fac in uniscore_variants(h, free_model, full).items():
            m = fac(i).fit(X[tr][:, cols], y[tr], task=task, names=names, judged=judged[tr])
            res[key] = m.score(np.nan_to_num(X[te_j][:, cols]))
        # the judge-free family's score: in-sample and cross-fitted for the anchors, on the test texts
        mp = UniScore(pi=0.0, n_ref=h, seed=i, free_model=free_model).fit(X[tr][:, cols], y[tr], task=task, names=names, judged=judged[tr])
        p_in = mp.score(np.nan_to_num(X[tr_j][:, cols]))
        pte = mp.score(np.nan_to_num(X[te_j][:, cols]))
        Xf_tr = np.asarray(X[tr][:, free_idx], float)
        w_f = np.asarray(mp.pool_direction(Xf_tr, y[tr], task), float)
        p_oof = (np.asarray(mp.pool_oof(Xf_tr, y[tr], task), float) / (np.abs(w_f).sum() + 1e-12))[judged[tr]]
        Xj_tr, Xj_te = X[tr_j][:, jidx], X[te_j][:, jidx]
        nm_stack = nmj + ["pool"]
        stacks = [("stack_lda", "lda", p_in), ("stack_logit", "logit", p_in), ("stack_lda_cf", "lda", p_oof), ("stack_logit_cf", "logit", p_oof), ("stack_nb", "nb", p_oof)]
        if full:
            stacks += [("stack_pls", "pls", p_oof), ("stack_ridge", "ridge", p_oof), ("abl_flat_ahp", "ahp", p_in)]
        for key, base, ptr in stacks:
            m = A[base](i).fit(np.c_[Xj_tr, ptr], y[tr_j], task=task, names=nm_stack)
            res[key] = m.score(np.c_[Xj_te, pte])
        m = A["lda"](i).fit(np.c_[X[tr_j, hol], p_in], y[tr_j], task=task, names=["holistic", "pool"])
        res["stack_holistic"] = m.score(np.c_[X[te_j, hol], pte])
        res["holistic"] = X[te_j, hol]
        if full:
            lo, hi = reference_groups(y[tr_j], task, "auto", h)
            a, b = sample_reference(lo, hi, h, np.random.RandomState(i))
            # the mean of the judge criteria (standardised and signed on the anchors) and the cross-fitted pool score, by LDA
            anc = tr_j[np.r_[a, b]]
            Zj = X[:, jidx]
            mu, sd = Zj[anc].mean(0), Zj[anc].std(0) + 1e-9
            Zs = (Zj - mu) / sd
            sg = np.sign(S.somers_d_cov(Zs[tr_j][a], Zs[tr_j][b])["D"])
            sg[sg == 0] = 1
            jm = Zs @ sg / len(sg)
            m = A["lda"](i).fit(np.c_[jm[tr_j], p_oof], y[tr_j], task=task, names=["judge_mean", "pool"])
            res["stack_mean"] = m.score(np.c_[jm[te_j], pte])
            res["stack_shrink"] = shrunk_stack(Xj_tr, p_oof, Xj_te, pte, a, b)
            # without pool labels: the anchors are the only supervision (all criteria, or the judge criteria alone)
            for tag, cc in (("all", cols), ("judge", jcols)):
                for base in (("ahp", "lda", "nb", "logit") if tag == "all" else ("ahp", "lda", "logit", "equal")):
                    m = A[base](i).fit(X[tr_j][:, cc], y[tr_j], task=task, names=[names[c] for c in cc])
                    res[f"nopool_{base}_{tag}"] = m.score(X[te_j][:, cc])
        for key, s in res.items():
            rows.append(dict(split=i, B=B, method=key, spearman=spearman(y[te_j], np.asarray(s, float))))
    return rows


def shrunk_stack(Xj_tr, p_oof, Xj_te, pte, a, b):
    """LDA weights of the judge criteria and the cross-fitted pool score, shrunk toward the marginal (signed Somers' D) weights;
    the shrinkage in {0, .1, ..., 1} chosen by leave-one-out AUC on the anchors."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    Za = np.r_[np.c_[Xj_tr[a], p_oof[a]], np.c_[Xj_tr[b], p_oof[b]]]
    ga = np.r_[np.zeros(len(a)), np.ones(len(b))]
    mu, sd = Za.mean(0), Za.std(0) + 1e-9

    def wts(Z, g, lam):
        Zs = (Z - mu) / sd
        wl = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Zs, g).coef_[0]
        wl = wl / (np.abs(wl).sum() + 1e-12)
        D = S.somers_d_cov(Zs[g == 0], Zs[g == 1])["D"]
        wm = np.asarray(D, float) / (np.abs(D).sum() + 1e-12)
        return (1 - lam) * wl + lam * wm

    best, bl = -1, 1.0
    for lam in np.linspace(0, 1, 11):
        sc = np.zeros(len(ga))
        for k in range(len(ga)):
            keep = np.arange(len(ga)) != k
            if min((ga[keep] == 0).sum(), (ga[keep] == 1).sum()) < 2:
                continue
            sc[k] = ((Za[k] - mu) / sd) @ wts(Za[keep], ga[keep], lam)
        auc = float(S.somers_d_cov(sc[ga == 0][:, None], sc[ga == 1][:, None])["D"][0])
        if auc > best + 1e-12:
            best, bl = auc, lam
    return ((np.c_[Xj_te, pte] - mu) / sd) @ wts(Za, ga, bl)


def main():
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--judges", default="".join(c for c, _, _ in JUDGES), help="judge codes, e.g. QGMLHSX")
    ap.add_argument("--free-model", default="ridge", choices=["ridge", "ahpd"])
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    full = a.free_model == "ridge"
    out = os.path.join(RESULTS, "single" if full else "single_ahpd")
    os.makedirs(out, exist_ok=True)
    for ds in a.datasets.split(","):
        pool, free = load_pool(ds)
        for code in a.judges:
            t0 = time.time()
            df, jc, hol = load_judge(ds, code, pool)
            X0 = df[jc + free + [hol]].to_numpy(float)
            y = df["target"].to_numpy(float)
            jidx = list(range(len(jc)))
            free_idx = list(range(len(jc), len(jc) + len(free)))
            splits = grouped_splits(df["group"].to_numpy(), n_splits=a.splits, seed=0)
            res = Parallel(n_jobs=a.jobs)(delayed(one_split)(i, sp, X0, y, jidx, free_idx, len(jc) + len(free), TASK[ds], full, a.free_model) for i, sp in enumerate(splits))
            r = pd.DataFrame([x for rows in res for x in rows]).assign(dataset=ds, judge=code)
            r.to_csv(os.path.join(out, f"{code}__{ds}.csv"), index=False)
            print(f"{ds} {code}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tables 2-5 of the paper (main texts, seven judges) from results/single, results/single_ahpd and results/panel.

Writes tables/factorial.tex (Table 2), tables/panel.tex (Table 3, rows of the main texts), tables/lead.tex (Table 4, row
"main texts") and tables/ablation.tex (Table 5, columns "main texts").  Units of every test: the five datasets (judges are
averaged within a dataset); Student-t 90 % intervals, exact one-sided Wilcoxon signed-rank tests.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from common import BUDGETS, DATASETS, JUDGES, RESULTS, ROOT
from scipy import stats as sps

from uniscore import stats as S

OUT = os.path.join(ROOT, "tables")
SAME_INPUT = ["stack_lda", "stack_logit", "stack_lda_cf", "stack_logit_cf", "stack_nb", "stack_pls", "stack_ridge", "stack_mean", "stack_shrink", "stack_holistic"]
f3 = lambda v: f"{v:.3f}".replace("0.", ".", 1)
fs = lambda v: (("+" if v >= 0 else "$-$") + f"{abs(v):.4f}".lstrip("0")) if abs(v) < 0.0005 else ("+" if v >= 0 else "$-$") + f3(abs(v))
fs0 = lambda v: "$\\pm$.000" if abs(v) < 0.0005 else ("+" if v >= 0 else "$-$") + f3(abs(v))


def load(sub):
    """judge -> DataFrame indexed by (method, B), columns = datasets, values = mean Spearman over the splits"""
    T = {}
    for code, _, _ in JUDGES:
        parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RESULTS, sub, f"{code}__*.csv")))]
        if parts:
            T[code] = pd.concat(parts).groupby(["method", "B", "dataset"])["spearman"].mean().unstack("dataset").reindex(columns=DATASETS)
    return T


def over_datasets(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    q = sps.t.ppf(0.95, n - 1)
    w = S.wilcoxon_exact(d, "greater")
    return dict(mean=float(d.mean()), lo=float(d.mean() - q * se), hi=float(d.mean() + q * se), wins=int((d > 0).sum()), n=n, p=float(w["p"]))


def cell(T, k, B):
    v = []
    for t in T.values():
        if (k, B) not in t.index or t.loc[(k, B)].isna().any():
            return None
        v.append(t.loc[(k, B)].to_numpy(float))
    return float(np.mean(v))


def table_factorial(TR, TP):
    groups = [
        ("pool: rank-ridge (fitted)", TR, [("uniscore", "\\uni{}"), ("stack_lda", "LDA stack"), ("stack_logit", "logistic stack"), ("stack_lda_cf", "LDA stack, cross-fitted"),
                                           ("stack_logit_cf", "logistic stack, cross-fitted"), ("stack_nb", "naive Bayes stack"), ("stack_pls", "PLS-1 stack"), ("stack_ridge", "ridge stack"),
                                           ("stack_mean", "mean judge criterion + pool score"), ("stack_shrink", "LDA stack shrunk to marginal weights"), ("stack_holistic", "holistic + pool score")]),
        ("pool: AHP of Somers' $D$", TP, [("uniscore", "AHP priorities"), ("stack_lda", "LDA stack"), ("stack_logit", "logistic stack"), ("stack_lda_cf", "LDA stack, cross-fitted"),
                                          ("stack_logit_cf", "logistic stack, cross-fitted"), ("stack_nb", "naive Bayes stack"), ("stack_holistic", "holistic + pool score")]),
        ("no pool labels: all criteria", TR, [("nopool_ahp_all", "AHP priorities"), ("nopool_lda_all", "LDA"), ("nopool_nb_all", "naive Bayes"), ("nopool_logit_all", "logistic")]),
        ("no pool labels: judge criteria", TR, [("nopool_ahp_judge", "AHP priorities"), ("nopool_lda_judge", "LDA"), ("nopool_logit_judge", "logistic"), ("nopool_equal_judge", "equal weights")]),
    ]
    rows = [(g, lab, [cell(T, k, B) for B in BUDGETS]) for g, T, keys in groups for k, lab in keys]
    rows = [r for r in rows if any(v is not None for v in r[2])]
    best = [max(v[i] for _, _, v in rows if v[i] is not None) for i in range(len(BUDGETS))]
    L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "on the anchors & " + " & ".join(f"$B={b}$" for b in BUDGETS) + " \\\\", "\\midrule"]
    last = None
    for g, lab, vals in rows:
        if g != last:
            if last is not None:
                L.append("\\addlinespace[2pt]")
            L.append("\\multicolumn{5}{@{}l}{\\emph{%s}} \\\\" % g)
            last = g
        cells = ["--" if v is None else (f"\\textbf{{{f3(v)}}}" if np.isclose(v, best[i]) else f3(v)) for i, v in enumerate(vals)]
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += ["\\addlinespace[2pt]", "\\multicolumn{5}{@{}l}{\\emph{without anchors}} \\\\"]
    for lab, T, k in (("holistic prompt", TR, "holistic"), ("pool score, rank-ridge", TR, "pool_only"), ("pool score, AHP priorities", TP, "pool_only")):
        v = cell(T, k, 40)
        L.append(f"{lab} & \\multicolumn{{4}}{{c}}{{{f3(v) if v is not None else '--'}}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return L


def table_lead(TR):
    """lead of UniScore over each judge's strongest same-input alternative (chosen on the test results), judges averaged within a dataset"""
    cells = []
    for B in BUDGETS:
        per = {ds: [] for ds in DATASETS}
        for t in TR.values():
            cand = [m for m in SAME_INPUT if (m, B) in t.index and t.loc[(m, B)].notna().all()]
            best = max(cand, key=lambda m: t.loc[(m, B)].mean())
            for ds in DATASETS:
                per[ds].append(float(t.loc[("uniscore", B), ds] - t.loc[(best, B), ds]))
        st = over_datasets([np.mean(v) for v in per.values()])
        cells.append(f"{fs(st['mean'])}" + ("$^{\\dagger}$" if st["lo"] > 0 else "") + f" ({st['wins']}/{st['n']})")
    return ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "block (units) & " + " & ".join(f"$B={b}$" for b in BUDGETS) + " \\\\", "\\midrule",
            f"main texts ({len(TR)}) & " + " & ".join(cells) + " \\\\", "\\bottomrule", "\\end{tabular}"]


ABLATION = [("abl_equal_judge", "equal judge weights"), ("abl_lda_judge", "judge weights fitted (LDA)"), ("abl_lda_family", "family priority fitted (LDA)"), ("stack_lda", "both fitted (LDA stack)"), None,
            ("abl_flat_ahp", "flat AHP, judge + pool score"), ("abl_one_vector", "one AHP vector, all criteria"), ("abl_equal_family", "equal family priorities"), ("abl_ef_anchors", "$e_F$ on the anchors"), None,
            ("abl_no_map", "evidence without AHP map"), ("abl_median_anchors", "median-split anchors"), None,
            ("abl_ahp_pool", "AHP on the pool"), ("abl_no_pool_labels", "w/o pool labels"), ("pool_only", "w/o judge family"), ("judge_only", "w/o judge-free family")]


def table_ablation(TR, bs=(10, 40)):
    """Delta = rho(UniScore) - rho(variant), judges averaged within a dataset; * / ** one-sided Wilcoxon p < .05 / .01 over the datasets, triangle: the variant is significantly better"""
    def mark(d):
        d = np.asarray(d, float)
        if not np.any(np.abs(d) > 1e-12):
            return "n/a"
        pp, pn = S.wilcoxon_exact(d, "greater")["p"], S.wilcoxon_exact(d, "less")["p"]
        m = "**" if pp < 0.01 else ("*" if pp < 0.05 else "")
        if pn < 0.05 and d.mean() < 0:
            m += "\\triangledown"
        return fs0(d.mean()) + (f"$^{{{m}}}$" if m else "")

    absu = {B: float(np.mean([np.mean([t.loc[("uniscore", B), ds] for t in TR.values()]) for ds in DATASETS])) for B in bs}
    L = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "variant & $B=10$ & $B=40$ \\\\", "\\midrule", "\\uni{} ($\\rho$) & " + " & ".join(f3(absu[B]) for B in bs) + " \\\\", "\\midrule"]
    for r in ABLATION:
        if r is None:
            L.append("\\addlinespace[1pt]")
            continue
        k, lab = r
        cells = []
        for B in bs:
            d = [np.mean([t.loc[("uniscore", B), ds] - t.loc[(k, B), ds] for t in TR.values()]) for ds in DATASETS]
            cells.append(mark(d))
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return L


PANEL_ALTS = ["stack_lda", "stack_nb", "stack_logit", "jury_lda", "jury_logit", "equal_lda", "factor_lda", "judgemeans_lda", "judgemeans_logit", "critavg_lda", "critavg_logit"]
PANELS = [("QGM", "three open judges"), ("LHSX", "four API judges"), ("QGMLHSX", "all seven judges")]


def table_panel():
    L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "panel & " + " & ".join(f"$B={b}$" for b in BUDGETS) + " \\\\", "\\midrule"]
    for code, lab in PANELS:
        f = os.path.join(RESULTS, "panel", f"{code}.csv")
        if not os.path.exists(f):
            continue
        t = pd.read_csv(f).groupby(["method", "B", "dataset"])["spearman"].mean().unstack("dataset").reindex(columns=DATASETS)
        cells = []
        for B in BUDGETS:
            cand = [m for m in PANEL_ALTS if (m, B) in t.index and t.loc[(m, B)].notna().all()]
            best = max(cand, key=lambda m: t.loc[(m, B)].mean())
            st = over_datasets((t.loc[("uniscore", B)] - t.loc[(best, B)]).to_numpy(float))
            cells.append(f"{fs(st['mean'])}" + ("$^{\\dagger}$" if st["lo"] > 0 else "") + f" ({st['wins']})")
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return L


def main():
    os.makedirs(OUT, exist_ok=True)
    TR, TP = load("single"), load("single_ahpd")
    out = {"factorial": table_factorial(TR, TP), "lead": table_lead(TR), "ablation": table_ablation(TR), "panel": table_panel()}
    for name, L in out.items():
        open(os.path.join(OUT, f"{name}.tex"), "w").write("\n".join(L) + "\n")
        print(f"== tables/{name}.tex\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

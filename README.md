# UniScore

Code and data for **"Uniscore: Aggregating LLM-Judge Criteria with a Labelled Pool and Few Anchors"**.

UniScore turns the criterion scores of an LLM judge into one quality score. Two kinds of supervision of very different
sizes are available: quality labels and judge-free criteria for a large pool of texts, and judge scores for a few
labelled texts (the *anchors*, B/2 from each tail of the labels). UniScore is a two-level AHP hierarchy:

- **judge-free family**: weights fitted on the labelled pool (rank-ridge);
- **judge family**: AHP priorities of each judge criterion's tested evidence (Somers' D between the anchor tails);
- **level 1**: a 2 x 2 AHP comparison of the two families' out-of-fold evidence gives the judge family's priority.

## Install

Python 3.10, CPU only.

```bash
pip install -r requirements.txt
```

## Use

```python
import numpy as np
from uniscore import UniScore

# X: (n, K) criteria; judge-free columns are named with the prefixes q_ / e_, all others are judge criteria
# y: labels of the pool; judged: True for the rows the judge scored (NaN in the judge columns elsewhere)
model = UniScore(n_ref=5).fit(X, y, task="regression", names=names, judged=judged)   # B = 10 anchors
scores = model.score(np.nan_to_num(X_new))
model.pi_       # priority of the judge family
model.family_   # the evidence of both families that generated it
```

`uniscore.SomersAHP(...).fit(...).audit()` returns the evidence table of the judge criteria (Somers' D, DeLong standard
error, 95 % interval, weight).

## Reproduce the main results

The repository contains the judge scores of the seven judges and the judge-free criteria of the five human-labelled
datasets (`data/`, no texts). The labels of ASAP and Dreaddit are not redistributed: download `training_set_rel3.tsv`
from the [ASAP-AES competition](https://www.kaggle.com/c/asap-aes/data); Dreaddit is fetched from Hugging Face.

```bash
bash reproduce.sh /path/to/training_set_rel3.tsv
```

or step by step (from `scripts/`):

| step | command | output |
|---|---|---|
| labels of ASAP and Dreaddit | `python fetch_labels.py --asap /path/to/training_set_rel3.tsv` | `data/labels/` |
| one judge at a time, all methods | `python run_single.py` | `results/single/` |
| judge-free family by AHP priorities | `python run_single.py --free-model ahpd` | `results/single_ahpd/` |
| panels of judges | `python run_panel.py --panel QGM` (and `LHSX`, `QGMLHSX`) | `results/panel/` |
| tables | `python make_tables.py` | `tables/*.tex` |
| comparison with the paper | `python check_against_paper.py` | |

The whole run takes about 1.5 hours on 32 CPU cores. `make_tables.py` writes

| file | paper |
|---|---|
| `tables/factorial.tex` | Table 2 (where the weights are fitted and where they are generated) |
| `tables/panel.tex` | Table 3, panels on the main texts |
| `tables/lead.tex` | Table 4, row "main texts" (lead over the strongest same-input alternative) |
| `tables/ablation.tex` | Table 5, columns "main texts" |

and `check_against_paper.py` compares them line by line with the values printed in the paper (`expected/`).

Protocol: 30 grouped 60/10/30 splits per dataset; per split 1,000 labelled pool texts (the judged training texts and
random unjudged ones); budgets B = 10, 20, 40 and 100 anchors; the test texts are the judged texts of the test role.
Every method sees the same splits, labelled texts and anchors; the unit of every test is the dataset.

## Layout

```
uniscore/          the method (model.py), the anchor estimators and baselines (estimators.py), evidence and AHP maps
                   (weights.py), Somers' D with DeLong covariance and the exact Wilcoxon test (stats.py)
scripts/           the experiments of the paper's main results
data/              judge-free criteria, labels where the terms allow, judge scores (see data/README.md)
prompts/           the judge prompts (criteria.json)
expected/          the tables as printed in the paper
tests/             unit tests (python -m pytest tests)
```

## License

Code: Apache-2.0. The derived criteria and judge scores in `data/` keep the terms of their source datasets
(`data/README.md`).

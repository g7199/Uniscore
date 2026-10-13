#!/usr/bin/env bash
# Reproduce Tables 2-5 of the paper (main texts, seven judges) from the released judge scores and judge-free criteria.
#   bash reproduce.sh /path/to/training_set_rel3.tsv        (ASAP essays, from Kaggle; see README)
# About 1.5 hours on 32 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/scripts"
python fetch_labels.py --asap "$1"
python run_single.py
python run_single.py --free-model ahpd
for p in QGM LHSX QGMLHSX; do python run_panel.py --panel "$p"; done
python make_tables.py
python check_against_paper.py

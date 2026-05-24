# UniScore Main Table Reproduction

This folder contains the files needed to reproduce the paper's main result
table. The Git-tracked code starts from raw datasets, loads Qwen3-1.7B from
Hugging Face, regenerates criterion-level judge scores, and then recomputes the
main table.

## Contents

- `data/raw/`: local raw datasets. This directory is ignored by Git.
- `inputs/`: generated criterion-level Qwen judge scores. This directory is
  ignored by Git.
- `src/uniscore/`: minimal UniScore helper code used by the scripts.
- `scripts/score_qwen.py`: generates the ignored scored CSVs under `inputs/`.
- `scripts/recompute_main_numbers.py`: recomputes baseline and UniScore
  statistics from the cached inputs.
- `scripts/render_main_table.py`: renders the submitted rounded table as text
  or LaTeX from `data/main_table_values.csv`.
- `scripts/mapping_delta_ablation.py`: reproduces the AHP mapping-function
  ablation.
- `requirements.txt`: Python dependencies.

The scripts do not require the original working-directory layout. All paths are
relative to this folder.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Raw Data

Place the raw datasets in one of the supported layouts:

```text
data/raw/amazonqa/Software_5.json
data/raw/samsum/train.csv
data/raw/samsum/test.csv
```

For the SAMSum/RoSE experiment, this layout is also supported:

```text
data/raw/rose/samsum.test.acus.aggregated.jsonl
```

For Depression Tweet, the script downloads `ziq/depression_tweet` through
Hugging Face `datasets` unless `data/raw/depression/depression_tweet.csv` is
provided locally.

## End-to-End Reproduction

After setup and raw-data placement, the usual command is:

```bash
make
```

`make` checks the scored-cache targets under `inputs/`. If a required scored CSV
already exists, it is reused. If a required scored CSV is missing, Make invokes
`scripts/score_qwen.py` for that dataset before recomputing the table.

The command creates:

- `generated/recomputed_main_numbers.csv`
- `main_table.txt`
- `main_table_latex.txt`

To explicitly run Qwen scoring first and then recompute the table:

```bash
make full
```

To run only Qwen scoring:

```bash
make score
```

The scorer itself is cache-aware: complete scored CSVs are skipped, and missing
or incomplete scored CSVs are regenerated.

To force a clean Qwen regeneration, remove the ignored generated inputs first:

```bash
rm -rf inputs/software_qwen3_1_7b inputs/samsum inputs/depression
make full
```

To render only the submitted rounded table:

```bash
make table
```

To run the mapping ablation:

```bash
make mapping
```

## Notes

- `inputs/`, `data/raw/`, generated CSVs, rendered tables, and local virtual
  environments are ignored by Git.
- `make` reuses existing scored CSVs under `inputs/` and regenerates missing
  ones with Qwen3-1.7B.
- `make score` runs the cache-aware Qwen scoring command for all datasets.
- Software and SAMSum use train-anchored square-root word-count scaling.
- Depression uses only the four criterion scores and class labels.

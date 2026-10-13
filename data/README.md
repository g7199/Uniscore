# Data

Derived data only: no text of any dataset is included.

## `pools/<dataset>.csv.gz`

One row per pool text (the first 3,000 texts of a fixed random order of the dataset; SummEval: all 1,600).

| column | meaning |
|---|---|
| `id` | text id (`<dataset>:<id in the source dataset>`) |
| `group` | group of the grouped splits (source article for SummEval, prompt for HelpSteer2, post for Dreaddit, the text itself otherwise) |
| `target` | the human label (higher = better / more of the construct); empty for ASAP and Dreaddit, see `scripts/fetch_labels.py` |
| `q_*` | judge-free lexical criteria: length, word and sentence length, type-token ratio, digits, punctuation, capitals, exclamation and question marks, first and second person, negation and absolutist words, Flesch reading ease, VADER sentiment and its magnitude; overlap with the source or prompt (`q_ctx_*`) where one exists |
| `e_*` | judge-free encoder criteria from small pretrained models never trained on a target label: grammatical acceptability (`textattack/distilbert-base-uncased-CoLA`), sentiment (`distilbert-base-uncased-finetuned-sst-2-english`) and seven emotions (`j-hartmann/emotion-english-distilroberta-base`), or for summaries NLI entailment and contradiction with the source (`cross-encoder/nli-deberta-v3-xsmall`) and embedding similarity (`sentence-transformers/all-MiniLM-L6-v2`) |

Constant columns of a pool are dropped. The judge-free criteria are mapped to the 1-5 scale inside every split by the
empirical CDF of its training texts (`uniscore.evaluate.quantile_scale`).

## `judges/<judge>/<dataset>.csv.gz`

The scores of one judge on the 300 texts it scored (the first 300 of the pool order): one column per criterion and
`overall`, the holistic score. Score types: `qwen3-1.7b`, `gemma-4-e4b`, `gemma-4-26b-a4b` and `gpt-6-luna` give the
expected score of the next digit token; `claude-haiku-4.5`, `claude-sonnet-5.5` and `gpt-6.1-sol` give the integer
answer. Unparsable answers of the hosted judges were asked again (twice at most); the few that remained are replaced by the
median of the criterion.

## Sources and terms

| dataset | target | source | terms |
|---|---|---|---|
| `summeval` | mean of 3 experts over 4 dimensions | SummEval (Fabbri et al., 2021) | MIT |
| `helpsteer2` | helpfulness (0-4) | HelpSteer2 (Wang et al., 2024) | CC-BY-4.0 |
| `asap_aes` | essay score, min-max normalised per essay set | ASAP-AES (Kaggle) | competition rules; labels not redistributed |
| `dreaddit` | stress (binary) | Dreaddit (Turcan & McKeown, 2019) | none stated; labels not redistributed |
| `civil_comments` | toxicity fraction | Civil Comments (Borkan et al., 2019) | CC0-1.0 |

The derived criteria and judge scores keep the terms of their source datasets.

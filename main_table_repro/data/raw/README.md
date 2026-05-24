# Raw Data

Place raw datasets here before running `make score`.

Supported layouts:

```text
data/raw/amazonqa/Software_5.json
data/raw/samsum/train.csv
data/raw/samsum/test.csv
```

For SAMSum/RoSE, the scorer also accepts:

```text
data/raw/rose/samsum.test.acus.aggregated.jsonl
```

If the local depression CSV is not provided, `scripts/score_qwen.py` downloads
`ziq/depression_tweet` through Hugging Face `datasets`.

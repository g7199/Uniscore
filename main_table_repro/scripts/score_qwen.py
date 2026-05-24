#!/usr/bin/env python3
"""Generate Qwen3-1.7B criterion-score caches for the main table.

The output files are written under `inputs/`, which is ignored by Git. The
subsequent `make` step recomputes the main table from those generated caches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uniscore.models import gen_score, setup_model  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-1.7B"
RANDOM_STATE = 42


P_SOFTWARE = {
    "polarity": """Rate the sentiment polarity of the review on a 1-5 Likert scale.
1=very negative, 2=negative, 3=neutral/mixed, 4=positive, 5=very positive.
Return JSON only.
Review: {txt}
{{"score": N}}""",
    "expertise": """Rate the reviewer expertise shown in the text (domain terms, procedures, precise specs, comparisons).
1=none, 2=low, 3=moderate, 4=high, 5=very high expertise.
Return JSON only.
Review: {txt}
{{"score": N}}""",
    "specificity": """Rate how specific and concrete the review is (numbers, model names, scenarios, measurable details).
1=very vague, 2=vague, 3=some specifics, 4=specific, 5=highly specific with concrete details.
Return JSON only.
Review: {txt}
{{"score": N}}""",
    "consistency": """Given the star rating and the review text, rate rating-content consistency (1-5).
1=strongly inconsistent, 3=partly consistent/mixed, 5=fully consistent in tone and claims.
Star rating (0-5): {star}
Review: {txt}
Return JSON only.
{{"score": N}}""",
}

P_SAMSUM = {
    "coherence": """Evaluate the coherence of the given summary using a 1-5 Likert scale assessment.
Coherence encompasses the logical progression of ideas, semantic connectivity between sentences, and overall structural integrity of the narrative flow within the text.
1=severely fragmented/disjointed, 2=poorly connected, 3=adequately structured, 4=well-organized, 5=exceptionally cohesive.
Provide response in JSON format exclusively.
Summary: {txt}
{{"score": N}}""",
    "consistency": """Assess the consistency of the summary against the original source material using a 1-5 Likert scale evaluation.
Consistency encompasses factual accuracy, semantic fidelity, preservation of original meaning, and absence of contradictory information or misrepresentations relative to the source content.
1=substantially contradictory, 2=frequently inconsistent, 3=partially aligned, 4=largely faithful, 5=completely accurate.
Original source: {source}
Summary: {txt}
Respond with JSON format only.
{{"score": N}}""",
    "fluency": """Determine the fluency of the summary using a 1-5 Likert scale measurement.
Fluency incorporates syntactic correctness, lexical appropriateness, idiomatic expression, and the naturalness of language patterns that facilitate smooth comprehension.
1=extensively malformed, 2=grammatically problematic, 3=acceptably readable, 4=linguistically sound, 5=exceptionally polished.
Provide JSON response exclusively.
Summary: {txt}
{{"score": N}}""",
    "relevance": """Analyze the relevance of the summary in relation to the original source using a 1-5 Likert scale rating.
Relevance encompasses coverage of salient points, exclusion of extraneous content, proportional emphasis on key themes, and alignment with the source's communicative intent.
1=completely off-topic, 2=tangentially related, 3=moderately focused, 4=well-targeted, 5=perfectly aligned.
Original source: {source}
Summary: {txt}
Return JSON format only.
{{"score": N}}""",
}

P_DEP = {
    "negative_affect": """Rate the negative affect in the tweet (e.g., sadness, hopelessness expressions, as identified in depression-related social media analysis).
1=very positive, 2=positive, 3=neutral, 4=negative, 5=highly negative.
Return JSON only.
Tweet: {txt}
{{"score": N}}""",
    "self_focus": """Rate the self-focus in the tweet (e.g., use of first-person pronouns like I, me, my, indicating self-centered language in depressed texts).
1=none, 2=low, 3=moderate, 4=high, 5=very high.
Return JSON only.
Tweet: {txt}
{{"score": N}}""",
    "absolutist": """Rate the use of absolutist language in the tweet (e.g., words like always, never, everything, which are markers of black-and-white thinking in depression).
1=none, 2=low, 3=moderate, 4=high, 5=very high.
Return JSON only.
Tweet: {txt}
{{"score": N}}""",
    "social_isolation": """Rate the level of social isolation in the tweet (e.g., lack of social words like friend, we, talk, indicating withdrawal in depressed individuals).
1=highly social, 2=social, 3=neutral, 4=isolated, 5=highly isolated.
Return JSON only.
Tweet: {txt}
{{"score": N}}""",
}


def parse_vote(v) -> int:
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    s = re.sub(r"[^\d]", "", str(v))
    return int(s) if s else 0


def is_complete(path: Path, cols: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return all(c in df.columns and df[c].notna().all() for c in cols)


def score_rows(df: pd.DataFrame, prompts: dict[str, str], tok, lm, out_path: Path, text_col: str, source_col: str | None = None) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.read_csv(out_path) if out_path.exists() else df.copy()
    for col, template in prompts.items():
        if col not in out.columns:
            out[col] = np.nan
        missing = out[col].isna().to_numpy()
        for idx in tqdm(np.flatnonzero(missing), desc=f"{out_path.parent.name}:{col}", mininterval=10):
            row = out.iloc[idx]
            kwargs = {"txt": str(row[text_col])}
            if "{star}" in template:
                kwargs["star"] = float(row.get("overall", 3.0)) if pd.notnull(row.get("overall", np.nan)) else 3.0
            if source_col is not None:
                kwargs["source"] = str(row[source_col])
            out.iat[idx, out.columns.get_loc(col)] = gen_score(template.format(**kwargs), tok, lm, max_new_tokens=16)
            if idx % 100 == 0:
                out.to_csv(out_path, index=False)
        out.to_csv(out_path, index=False)
    return out


def load_software() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "data/raw/amazonqa/Software_5.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    rows = []
    with path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = obj.get("reviewText")
            if txt:
                rows.append({
                    "asin": obj.get("asin"),
                    "overall": obj.get("overall"),
                    "vote": parse_vote(obj.get("vote", 0)),
                    "review_text": txt,
                    "word_count": len(str(txt).split()),
                })
    df = pd.DataFrame(rows)
    tr, te = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def load_rose_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            source = obj.get("source", "")
            example_id = obj.get("example_id", "")
            outputs = obj.get("system_outputs", {})
            annotations = obj.get("annotations", {})
            for model_name, summary in outputs.items():
                ann = annotations.get(model_name) or {}
                if "acu" in ann:
                    rows.append({
                        "asin": f"{example_id}_{model_name}",
                        "overall": np.nan,
                        "vote": float(ann["acu"]),
                        "review_text": summary,
                        "source": source,
                        "model_name": model_name,
                    })
    return pd.DataFrame(rows)


def load_samsum() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = ROOT / "data/raw/samsum/train.csv"
    test_csv = ROOT / "data/raw/samsum/test.csv"
    if train_csv.exists() and test_csv.exists():
        return pd.read_csv(train_csv), pd.read_csv(test_csv)
    rose_jsonl = ROOT / "data/raw/rose/samsum.test.acus.aggregated.jsonl"
    if rose_jsonl.exists():
        df = load_rose_jsonl(rose_jsonl)
        tr, te = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
        return tr.reset_index(drop=True), te.reset_index(drop=True)
    raise FileNotFoundError(f"Missing {train_csv}/{test_csv} or {rose_jsonl}")


def load_depression() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    local = ROOT / "data/raw/depression/depression_tweet.csv"
    if local.exists():
        df = pd.read_csv(local)
        if "text" in df.columns and "tweet" not in df.columns:
            df = df.rename(columns={"text": "tweet"})
    else:
        from datasets import load_dataset
        ds = load_dataset("ziq/depression_tweet", split="train")
        df = pd.DataFrame({"tweet": ds["text"], "label": ds["label"]})
    tr, te = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    return (
        tr[tr["label"].astype(int) == 1].reset_index(drop=True),
        tr[tr["label"].astype(int) == 0].reset_index(drop=True),
        te.reset_index(drop=True),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help="all, software, samsum, depression or comma-separated list")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    wanted = {"software", "samsum", "depression"} if args.datasets == "all" else set(args.datasets.split(","))
    print(f"Loading {args.model}")
    tok, lm = setup_model(model_name=args.model, device_map=args.device_map)

    if "software" in wanted:
        out_dir = ROOT / "inputs/software_qwen3_1_7b"
        cols = list(P_SOFTWARE)
        train_path, test_path = out_dir / "train_results_with_scores.csv", out_dir / "test_results_with_scores.csv"
        train, test = load_software()
        if not is_complete(train_path, cols):
            score_rows(train, P_SOFTWARE, tok, lm, train_path, "review_text")
        if not is_complete(test_path, cols):
            score_rows(test, P_SOFTWARE, tok, lm, test_path, "review_text")

    if "samsum" in wanted:
        out_dir = ROOT / "inputs/samsum"
        cols = list(P_SAMSUM)
        train_path, test_path = out_dir / "train_results_with_scores.csv", out_dir / "test_results_with_scores.csv"
        train, test = load_samsum()
        if not is_complete(train_path, cols):
            score_rows(train, P_SAMSUM, tok, lm, train_path, "review_text", "source")
        if not is_complete(test_path, cols):
            score_rows(test, P_SAMSUM, tok, lm, test_path, "review_text", "source")

    if "depression" in wanted:
        out_dir = ROOT / "inputs/depression"
        cols = list(P_DEP)
        high_path = out_dir / "train_high_scored.csv"
        low_path = out_dir / "train_low_scored.csv"
        test_path = out_dir / "test_results_with_scores.csv"
        high, low, test = load_depression()
        if not is_complete(high_path, cols):
            score_rows(high, P_DEP, tok, lm, high_path, "tweet")
        if not is_complete(low_path, cols):
            score_rows(low, P_DEP, tok, lm, low_path, "tweet")
        if not is_complete(test_path, cols):
            score_rows(test, P_DEP, tok, lm, test_path, "tweet")


if __name__ == "__main__":
    main()

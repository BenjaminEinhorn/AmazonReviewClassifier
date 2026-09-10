"""Step 1 (Goal.md): predict Positive/Negative sentiment from a review's
title and text only. The star rating is intentionally never read as a
feature or label here -- it is held out for Step 2's evaluation.

Approach: lexicon-based scoring against the NRC Word-Emotion Association
Lexicon (NRCLexicon/*.txt, loaded via lexicon.py). Title+text is split
into sentences, and only sentences with more than MIN_SENTENCE_WORDS
words are kept -- short sentences ("Great!", "Works great.") are dropped
before scoring. The surviving sentences are tokenized with sklearn's
CountVectorizer restricted to the full NRC vocabulary, giving:

  - a positive-word count and negative-word count per review, which
    decide predicted_sentiment (whichever is higher wins; a tie,
    including 0-0, is logged as ambiguous instead of being forced into
    a class, per Goal.md's "In the case of ambiguity, append to a new
    file called ambiguity.md")
  - a count per basic emotion (anger, anticipation, disgust, fear, joy,
    sadness, surprise, trust), which decide dominant_emotion the same
    way (highest count wins; a tie, including 0-0 across all eight, is
    "ambiguous")
"""

import gzip
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from lexicon import EMOTIONS, EMOTION_WORDS, NEGATIVE_WORDS, POSITIVE_WORDS

INPUT_PATH = "amazonreviewdata.jsonl.gz"
PREDICTIONS_PATH = "predictions.csv"
AMBIGUITY_PATH = "ambiguity.md"

MIN_SENTENCE_WORDS = 8

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def filter_long_sentences(text: str, min_words: int = MIN_SENTENCE_WORDS) -> str:
    """Keep only sentences with more than `min_words` words."""
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    kept = [s for s in sentences if len(s.split()) > min_words]
    return " ".join(kept)


def load_reviews(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        records = [json.loads(line) for line in f]
    df = pd.DataFrame.from_records(records)
    df = df.reset_index().rename(columns={"index": "review_id"})
    return df


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    combined_text = (df["title"].fillna("") + " " + df["text"].fillna(""))
    filtered_text = combined_text.apply(filter_long_sentences)

    full_vocab = set(POSITIVE_WORDS) | set(NEGATIVE_WORDS)
    for words in EMOTION_WORDS.values():
        full_vocab |= words

    vectorizer = CountVectorizer(
        vocabulary=sorted(full_vocab),
        lowercase=True,
        token_pattern=r"(?u)\b[\w'-]+\b",
    )
    counts = vectorizer.transform(filtered_text)
    vocab_index = vectorizer.vocabulary_

    def sum_cols(words):
        cols = [vocab_index[w] for w in words if w in vocab_index]
        if not cols:
            return np.zeros(counts.shape[0], dtype=int)
        return np.asarray(counts[:, cols].sum(axis=1)).ravel()

    pos_count = sum_cols(POSITIVE_WORDS)
    neg_count = sum_cols(NEGATIVE_WORDS)

    emotion_counts = {emotion: sum_cols(EMOTION_WORDS[emotion]) for emotion in EMOTIONS}

    df = df.copy()
    df["pos_count"] = pos_count
    df["neg_count"] = neg_count

    conditions = [pos_count > neg_count, neg_count > pos_count]
    choices = ["positive", "negative"]
    df["predicted_sentiment"] = np.select(conditions, choices, default="ambiguous")

    emotion_matrix = np.column_stack([emotion_counts[e] for e in EMOTIONS])
    for emotion in EMOTIONS:
        df[f"{emotion}_count"] = emotion_counts[emotion]

    max_count = emotion_matrix.max(axis=1)
    is_max = emotion_matrix == max_count[:, None]
    tie_or_zero = (is_max.sum(axis=1) != 1) | (max_count == 0)
    argmax_idx = emotion_matrix.argmax(axis=1)
    dominant = np.array(EMOTIONS, dtype=object)[argmax_idx]
    dominant[tie_or_zero] = "ambiguous"
    df["dominant_emotion"] = dominant

    return df


def write_ambiguity_log(df: pd.DataFrame, path: str) -> None:
    ambiguous = df[df["predicted_sentiment"] == "ambiguous"]

    lines = []
    lines.append("# Ambiguity Log\n")
    lines.append(
        "Design-level ambiguities encountered while building the sentiment "
        "model (Step 1 / Goal.md):\n"
    )
    lines.append(
        "- Goal.md asks for a **binary** classifier (Positive vs Negative), "
        "but scoring title+text against a positive/negative word lexicon "
        "can produce ties (equal positive and negative word counts, "
        "including 0-0 when no sentiment words are present at all). A tie "
        "cannot be forced into either binary class without guessing, so "
        "those reviews are logged below as `ambiguous` instead of being "
        "predicted, and are excluded from the binary output.\n"
    )
    lines.append(
        "- Sarcasm, negation (\"not good\"), and mixed reviews (praises "
        "one aspect, criticizes another) are not modeled by a lexicon "
        "count -- these can silently land on the wrong side of the tie "
        "rather than being flagged. Not logged individually since the "
        "model has no way to detect them, noted here as a known limitation.\n"
    )
    lines.append(
        f"\n## Per-review ties ({len(ambiguous)} of {len(df)} reviews, "
        f"{len(ambiguous) / len(df):.2%})\n"
    )
    lines.append(
        "Reviews where the lexicon found an equal count of positive and "
        "negative words (pos_count == neg_count). Text truncated to 200 "
        "chars.\n"
    )
    lines.append(
        "\n| review_id | asin | pos_count | neg_count | title | text_snippet |"
    )
    lines.append("|---|---|---|---|---|---|")

    def esc(s: str) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ").strip()

    for _, row in ambiguous.iterrows():
        snippet = esc(row["text"])[:200]
        lines.append(
            f"| {row['review_id']} | {row['asin']} | {row['pos_count']} | "
            f"{row['neg_count']} | {esc(row['title'])[:80]} | {snippet} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df = load_reviews(INPUT_PATH)
    scored = score_sentiment(df)

    write_ambiguity_log(scored, AMBIGUITY_PATH)

    output_cols = [
        "review_id", "asin", "parent_asin", "title", "text",
        "pos_count", "neg_count", "predicted_sentiment",
        *[f"{emotion}_count" for emotion in EMOTIONS], "dominant_emotion",
    ]
    scored[output_cols].to_csv(PREDICTIONS_PATH, index=False)

    counts = scored["predicted_sentiment"].value_counts()
    emotion_counts = scored["dominant_emotion"].value_counts()
    print(f"Scored {len(scored)} reviews (title + text only, rating not used).")
    print(counts)
    print(emotion_counts)
    print(f"Wrote predictions to {PREDICTIONS_PATH}")
    print(f"Wrote ambiguity log to {AMBIGUITY_PATH}")


if __name__ == "__main__":
    main()

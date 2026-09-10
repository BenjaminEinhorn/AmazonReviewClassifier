"""Step 2 (Analysis.md): compare Step 1's title/text-only predictions
against the ground truth derived from the star rating, and surface it in
a locally hosted Streamlit dashboard.

Ground truth mapping (per Analysis.md):
  rating <= 2         -> negative
  rating == 3          -> ambiguous (excluded from the confusion matrix)
  rating >= 4           -> positive

Step 1 also scores each review against the 8 NRC basic emotions
(anger, anticipation, disgust, fear, joy, sadness, surprise, trust) and
assigns a dominant_emotion; this dashboard surfaces that breakdown too.

Run with:
    .venv/bin/streamlit run step2_dashboard.py
"""

import gzip
import json
import re
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from lexicon import EMOTIONS
from step1_sentiment_model import MIN_SENTENCE_WORDS, score_sentiment

RAW_DATA_PATH = "amazonreviewdata.jsonl.gz"
PREDICTIONS_PATH = "predictions.csv"
AMBIGUITY_PATH = "ambiguity.md"

EMOTION_DISPLAY_ORDER = EMOTIONS + ["ambiguous"]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "was", "were", "are",
    "it", "its", "this", "that", "to", "of", "in", "on", "for", "with",
    "as", "at", "by", "from", "i", "my", "me", "you", "your", "we",
    "they", "them", "he", "she", "his", "her", "had", "have", "has",
    "be", "been", "not", "no", "so", "if", "just", "very", "did",
    "do", "does", "will", "would", "can", "could", "would've", "im",
    "one", "up", "out", "all", "too", "than", "then", "when", "what",
    "which", "there", "here", "some", "into", "about", "gift", "card",
    "amazon", "get", "got", "use", "used", "product", "item",
}


@st.cache_data
def load_ground_truth(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        records = [json.loads(line) for line in f]
    df = pd.DataFrame.from_records(records)
    df = df.reset_index().rename(columns={"index": "review_id"})

    def rating_to_class(r):
        if r <= 2:
            return "negative"
        if r >= 4:
            return "positive"
        return "ambiguous"

    df["true_sentiment"] = df["rating"].apply(rating_to_class)
    return df[["review_id", "rating", "true_sentiment"]]


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_ambiguity_snippets(path: str) -> pd.DataFrame:
    """Parse the per-review ties table written by step1 out of ambiguity.md."""
    with open(path, "r") as f:
        lines = f.readlines()

    rows = []
    in_table = False
    header_seen = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| review_id"):
            in_table = True
            header_seen = False
            continue
        if in_table and stripped.startswith("|---"):
            header_seen = True
            continue
        if in_table and header_seen and stripped.startswith("|"):
            parts = re.split(r"(?<!\\)\|", stripped)
            # Drop only the empty artifacts from the table's enclosing
            # pipes, not legitimately-empty interior cells.
            if parts and parts[0].strip() == "":
                parts = parts[1:]
            if parts and parts[-1].strip() == "":
                parts = parts[:-1]
            parts = [p.strip().replace("\\|", "|") for p in parts]
            if len(parts) == 6:
                rows.append(parts)
        elif in_table and header_seen and not stripped.startswith("|"):
            in_table = False

    cols = ["review_id", "asin", "pos_count", "neg_count", "title", "text_snippet"]
    return pd.DataFrame(rows, columns=cols)


def binary_metrics(scored: pd.DataFrame) -> dict:
    """Precision/recall/sensitivity/specificity with 'positive' as the
    positive class, over rows already restricted to a firm prediction and
    a firm (non-3-star) rating."""
    pred, true = scored["predicted_sentiment"], scored["true_sentiment"]
    tp = int(((pred == "positive") & (true == "positive")).sum())
    fn = int(((pred == "negative") & (true == "positive")).sum())
    fp = int(((pred == "positive") & (true == "negative")).sum())
    tn = int(((pred == "negative") & (true == "negative")).sum())

    def safe_div(n, d):
        return n / d if d else float("nan")

    return {
        "n": len(scored),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "accuracy": safe_div(tp + tn, tp + fn + fp + tn),
    }


def per_product_metrics(merged_all: pd.DataFrame) -> pd.DataFrame:
    """binary_metrics(), grouped by asin, for every product with at least
    one firmly-scored review."""
    scored_all = merged_all[
        merged_all["predicted_sentiment"].isin(["positive", "negative"])
        & merged_all["true_sentiment"].isin(["positive", "negative"])
    ]
    rows = []
    for asin, group in scored_all.groupby("asin"):
        m = binary_metrics(group)
        rows.append({"asin": asin, "reviews_scored": m["n"], **{
            k: m[k] for k in ("precision", "recall", "sensitivity", "specificity", "accuracy")
        }})
    return pd.DataFrame(rows)


def classify_review(title: str, text: str) -> pd.Series:
    """Run a single ad-hoc title/text pair through Step 1's model."""
    input_df = pd.DataFrame({"title": [title], "text": [text]})
    return score_sentiment(input_df).iloc[0]


def top_recurring_words(snippets: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    text = " ".join((snippets["title"] + " " + snippets["text_snippet"]).tolist())
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counts = Counter(words)
    common = counts.most_common(top_n)
    return pd.DataFrame(common, columns=["word", "count"])


def main():
    st.set_page_config(page_title="Sentiment Model vs. Rating", layout="wide")
    st.title("Title/Text Sentiment Model vs. Star-Rating Ground Truth")
    st.caption(
        "Step 1 predicted sentiment from review title + text only "
        "(rating was never used as a feature). This dashboard scores those "
        "predictions against the rating-derived ground truth."
    )

    st.subheader("Classify a review")
    st.caption(
        "Enter a title and/or text and run it through the same Step 1 "
        f"model: sentences of {MIN_SENTENCE_WORDS} words or fewer are "
        "ignored, and the remaining text is scored against the NRC "
        "lexicon. Whichever of positive/negative has more word matches "
        "wins; a tie (including no matches) is ambiguous."
    )
    with st.form("classify_form"):
        input_title = st.text_input("Title")
        input_text = st.text_area("Text", height=120)
        submitted = st.form_submit_button("Classify")
    if submitted:
        if not input_title.strip() and not input_text.strip():
            st.warning("Enter a title and/or text to classify.")
        else:
            result = classify_review(input_title, input_text)
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Predicted sentiment", result["predicted_sentiment"])
            rc2.metric("Positive words matched", int(result["pos_count"]))
            rc3.metric("Negative words matched", int(result["neg_count"]))
            rc4.metric("Dominant emotion", result["dominant_emotion"])
    st.divider()

    truth = load_ground_truth(RAW_DATA_PATH)
    preds = load_predictions(PREDICTIONS_PATH)
    merged = preds.merge(truth, on="review_id", how="inner")
    merged_all = merged

    st.sidebar.header("Filters")
    asin_counts = merged["asin"].value_counts()
    asin_options = ["All products"] + asin_counts.index.tolist()
    selected_asin = st.sidebar.selectbox(
        "Product (ASIN)",
        options=asin_options,
        format_func=lambda a: a if a == "All products" else f"{a}  ({asin_counts[a]:,} reviews)",
    )
    if selected_asin != "All products":
        merged = merged[merged["asin"] == selected_asin]

    rating_options = [1.0, 2.0, 3.0, 4.0, 5.0]
    selected_ratings = st.sidebar.multiselect(
        "Star rating",
        options=rating_options,
        default=rating_options,
        format_func=lambda r: f"{int(r)} star" + ("" if r == 1 else "s"),
    )
    merged = merged[merged["rating"].isin(selected_ratings)]

    pred_options = ["positive", "negative", "ambiguous"]
    selected_preds = st.sidebar.multiselect(
        "Predicted sentiment (Step 1)",
        options=pred_options,
        default=pred_options,
    )
    merged = merged[merged["predicted_sentiment"].isin(selected_preds)]

    emotion_options = [e for e in EMOTION_DISPLAY_ORDER if e in merged["dominant_emotion"].unique()]
    selected_emotions = st.sidebar.multiselect(
        "Dominant emotion (NRC lexicon)",
        options=emotion_options,
        default=emotion_options,
    )
    merged = merged[merged["dominant_emotion"].isin(selected_emotions)]

    active_filters = []
    if selected_asin != "All products":
        active_filters.append(f"ASIN **{selected_asin}**")
    if set(selected_ratings) != set(rating_options):
        active_filters.append("rating in {" + ", ".join(str(int(r)) for r in sorted(selected_ratings)) + "}")
    if set(selected_preds) != set(pred_options):
        active_filters.append("predicted in {" + ", ".join(selected_preds) + "}")
    if set(selected_emotions) != set(emotion_options):
        active_filters.append("dominant emotion in {" + ", ".join(selected_emotions) + "}")
    if active_filters:
        st.caption("Filtered to " + ", ".join(active_filters) + f" -- {len(merged):,} reviews.")

    total = len(merged)
    if total == 0:
        st.warning("No reviews match this filter.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Total reviews", f"{total:,}")
    col2.metric(
        "Predicted ambiguous (Step 1)",
        f"{(merged['predicted_sentiment'] == 'ambiguous').sum():,}",
    )
    col3.metric(
        "True ambiguous (3-star)",
        f"{(merged['true_sentiment'] == 'ambiguous').sum():,}",
    )

    st.subheader("Class breakdown")
    breakdown = pd.DataFrame({
        "predicted": merged["predicted_sentiment"].value_counts(),
        "ground truth": merged["true_sentiment"].value_counts(),
    }).fillna(0).astype(int)
    st.dataframe(breakdown, width='stretch')

    with st.expander(f"Browse the {total:,} filtered reviews (title + text)"):
        browse_cols = [
            "review_id", "asin", "rating", "predicted_sentiment",
            "true_sentiment", "pos_count", "neg_count",
            "dominant_emotion", "title", "text",
        ]
        browse_cap = 500
        st.dataframe(merged[browse_cols].head(browse_cap), width='stretch', hide_index=True)
        if total > browse_cap:
            st.caption(f"Showing the first {browse_cap:,} of {total:,} matching reviews.")

    st.subheader("Confusion matrix")
    st.caption(
        "Restricted to reviews where BOTH the prediction and the rating "
        "resolve to a firm positive/negative -- 3-star reviews and reviews "
        "the lexicon scored as a tie are excluded here (see counts above), "
        "since neither side has a binary answer to score against."
    )
    scored = merged[
        merged["predicted_sentiment"].isin(["positive", "negative"])
        & merged["true_sentiment"].isin(["positive", "negative"])
    ]
    labels = ["positive", "negative"]
    cm = confusion_matrix(
        scored["true_sentiment"], scored["predicted_sentiment"], labels=labels
    )
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                          columns=[f"pred_{l}" for l in labels])

    cm_long = cm_df.reset_index().melt(id_vars="index", var_name="predicted", value_name="count")
    cm_long = cm_long.rename(columns={"index": "actual"})
    heatmap = alt.Chart(cm_long).mark_rect().encode(
        x=alt.X("predicted:N", title="Predicted"),
        y=alt.Y("actual:N", title="Actual (from rating)"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["actual", "predicted", "count"],
    )
    text = heatmap.mark_text(baseline="middle", fontSize=16).encode(
        text="count:Q",
        color=alt.value("black"),
    )
    st.altair_chart(heatmap + text, width='stretch')

    accuracy = (scored["predicted_sentiment"] == scored["true_sentiment"]).mean()
    st.metric("Accuracy on non-ambiguous overlap", f"{accuracy:.2%}",
              help=f"Computed over {len(scored):,} reviews with a firm prediction and a firm (non-3-star) rating.")

    m = binary_metrics(scored)
    st.caption(
        "'Positive' is the positive class: TP = predicted positive & rated "
        "positive. Recall and sensitivity are the same quantity here."
    )

    def fmt(v):
        return "N/A (no such reviews)" if pd.isna(v) else f"{v:.2%}"

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Precision", fmt(m["precision"]),
               help="Of reviews predicted positive, the fraction actually rated positive: TP / (TP + FP).")
    mc2.metric("Recall", fmt(m["recall"]),
               help="Of reviews actually rated positive, the fraction predicted positive: TP / (TP + FN).")
    mc3.metric("Sensitivity", fmt(m["sensitivity"]),
               help="Same as recall for the positive class: TP / (TP + FN).")
    mc4.metric("Specificity", fmt(m["specificity"]),
               help="Of reviews actually rated negative, the fraction predicted negative: TN / (TN + FP).")

    st.divider()
    st.subheader("Emotion breakdown (NRC lexicon)")
    st.caption(
        "Each review's dominant_emotion is whichever of the 8 NRC basic "
        "emotions (anger, anticipation, disgust, fear, joy, sadness, "
        "surprise, trust) scored the most word matches in the review's "
        "long sentences (>8 words). A tie -- including no emotion words "
        "matched at all -- is 'ambiguous'."
    )

    ec1, ec2 = st.columns(2)
    with ec1:
        st.caption("Dominant emotion, filtered reviews:")
        emotion_counts = (
            merged["dominant_emotion"]
            .value_counts()
            .reindex(EMOTION_DISPLAY_ORDER)
            .dropna()
            .astype(int)
        )
        st.bar_chart(emotion_counts)
    with ec2:
        st.caption("Dominant emotion by star rating:")
        by_rating = (
            merged.groupby(["rating", "dominant_emotion"])
            .size()
            .reset_index(name="count")
        )
        emotion_by_rating_chart = alt.Chart(by_rating).mark_bar().encode(
            x=alt.X("rating:O", title="Star rating"),
            y=alt.Y("count:Q", title="Reviews", stack="normalize"),
            color=alt.Color(
                "dominant_emotion:N",
                title="Dominant emotion",
                sort=EMOTION_DISPLAY_ORDER,
            ),
            tooltip=["rating", "dominant_emotion", "count"],
        )
        st.altair_chart(emotion_by_rating_chart, width='stretch')

    with st.expander("Average emotion word counts per review (filtered)"):
        emotion_cols = [f"{e}_count" for e in EMOTIONS]
        avg_emotion = merged[emotion_cols].mean().rename(
            lambda c: c.replace("_count", "")
        )
        avg_emotion = avg_emotion.reindex(EMOTIONS)
        st.bar_chart(avg_emotion)

    st.divider()
    st.subheader("Ambiguity.md: counts and recurring sentiments")
    snippets = load_ambiguity_snippets(AMBIGUITY_PATH)
    if selected_asin != "All products":
        snippets = snippets[snippets["asin"] == selected_asin]
    st.metric("Per-review ties logged in ambiguity.md", f"{len(snippets):,}")

    if len(snippets) == 0:
        st.caption("No ambiguous (tied) reviews for this product.")
    else:
        left, right = st.columns([1, 1])
        with left:
            st.caption("Where those ambiguous (tied) reviews actually landed by rating:")
            amb_ids = set(snippets["review_id"].astype(int))
            amb_truth = merged[merged["review_id"].isin(amb_ids)]["true_sentiment"].value_counts()
            st.bar_chart(amb_truth)
        with right:
            st.caption("Most frequent words across ambiguous review titles/text:")
            top_words = top_recurring_words(snippets)
            st.bar_chart(top_words.set_index("word"))

        with st.expander("Sample of logged ambiguous reviews"):
            st.dataframe(snippets.sample(min(50, len(snippets)), random_state=0),
                         width='stretch')

    st.divider()
    st.subheader("Per-product metrics")
    st.caption(
        "Precision / recall / sensitivity / specificity for every product "
        "(not affected by the sidebar filter above). Products with very few "
        "scored reviews give unstable metrics, so a minimum-review floor is "
        "applied below."
    )
    min_reviews = st.number_input(
        "Minimum scored reviews per product", min_value=1, value=20, step=1,
    )
    full_product_table = per_product_metrics(merged_all)
    product_table_total = len(full_product_table)
    product_table = full_product_table[full_product_table["reviews_scored"] >= min_reviews]
    product_table = product_table.sort_values("reviews_scored", ascending=False)
    display_table = product_table.copy()
    for col in ("precision", "recall", "sensitivity", "specificity", "accuracy"):
        display_table[col] = display_table[col].map(
            lambda v: "N/A" if pd.isna(v) else f"{v:.2%}"
        )
    st.dataframe(display_table, width='stretch', hide_index=True)
    st.caption(f"Showing {len(product_table):,} of {product_table_total:,} products with "
               f"at least one scored review.")


if __name__ == "__main__":
    main()

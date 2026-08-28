"""Feature engineering (Phase 2): Feature Set A (metadata) and Feature Set B additions (sentiment)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from textblob import TextBlob
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config

logger = logging.getLogger(__name__)

_vader = SentimentIntensityAnalyzer()


def build_metadata_features(users_df: pd.DataFrame) -> pd.DataFrame:
    """Feature Set A: account-level metadata only."""
    df = users_df.copy()

    if "account_age_days" not in df.columns:
        created = pd.to_datetime(
            df.get("created_at"), format=config.TWITTER_CREATED_AT_FORMAT, errors="coerce", utc=True
        )
        now = pd.Timestamp.now(tz="UTC")
        df["account_age_days"] = (now - created).dt.days

    for col in ("followers_count", "friends_count", "statuses_count", "listed_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    if "verified" in df.columns:
        df["verified"] = (
            df["verified"]
            .map({True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0, 1: 1, 0: 0})
            .fillna(0)
            .astype(int)
        )
    else:
        df["verified"] = 0

    out = df[["account_key", "label"] + config.METADATA_FEATURES].copy()
    medians = out[config.METADATA_FEATURES].median(numeric_only=True)
    out[config.METADATA_FEATURES] = out[config.METADATA_FEATURES].fillna(medians)
    return out


def _polarity_scores(texts: list[str]) -> np.ndarray:
    return np.array([_vader.polarity_scores(t)["compound"] for t in texts])


def _subjectivity_scores(texts: list[str]) -> np.ndarray:
    return np.array([TextBlob(t).sentiment.subjectivity for t in texts])


def _classify_topic(text: str) -> str | None:
    """Run 3: assign a tweet to the first matching keyword bucket, or None
    if it matches none of config.TOPIC_KEYWORDS."""
    lowered = text.lower()
    for topic, keywords in config.TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return None


def _contextual_consistency_score(texts: list[str], polarity: np.ndarray) -> float:
    """Run 3, Feature 1: std of mean sentiment across topic buckets.

    LOW std = sentiment doesn't shift across contexts = high consistency =
    AI-like. Needs tweets in >=2 distinct topic buckets; otherwise NaN
    (filled with the column median - or 0.0 if every account in the run
    lacks bucket coverage - once the full per-account table is built).
    """
    bucket_scores: dict[str, list[float]] = {}
    for text, score in zip(texts, polarity):
        topic = _classify_topic(text)
        if topic is not None:
            bucket_scores.setdefault(topic, []).append(score)

    if len(bucket_scores) < 2:
        return float("nan")

    bucket_means = [float(np.mean(scores)) for scores in bucket_scores.values()]
    return float(np.std(bucket_means, ddof=0))


def _temporal_sentiment_drift(polarity: np.ndarray) -> float:
    """Run 3, Feature 3: std of mean sentiment across three chronological
    windows (early/mid/late - `polarity` must already be in chronological
    order, oldest first).

    LOW drift = emotionally flat over time = AI-like. Needs at least
    config.MIN_TWEETS_FOR_TEMPORAL_DRIFT tweets; otherwise NaN (filled with
    the column median at build time).
    """
    if len(polarity) < config.MIN_TWEETS_FOR_TEMPORAL_DRIFT:
        return float("nan")

    windows = [w for w in np.array_split(polarity, 3) if len(w) > 0]
    if len(windows) < 2:
        return float("nan")

    window_means = [float(w.mean()) for w in windows]
    return float(np.std(window_means, ddof=0))


def _topic_mismatch(topic: str, polarity_score: float) -> bool:
    """Run 3, Feature 2: whether one tweet's sentiment breaks the "expected"
    pattern for its topic bucket.

    These four rules are an explicit authorial interpretation filling a gap
    in the original brief, not a claim grounded in cited literature - the
    brief only gave qualitative descriptions for three of four buckets
    ("expect strong polarity" / "expect positive lean" / "expect high
    variance") and none for "news". Documented here rather than left
    implicit:
      - politics:       mismatch if |polarity| < 0.5 (not strongly opinionated)
      - entertainment:  mismatch if polarity <= 0 (not positive)
      - sports:         "expect high variance" isn't a per-tweet property
                         (variance needs multiple tweets); reinterpreted as
                         "expect non-neutral" - mismatch if |polarity| < 0.2
      - news:           not specified in the brief; inferred as the inverse
                         of politics (news expected to read neutral/factual)
                         - mismatch if |polarity| > 0.5
    """
    if topic == "politics":
        return abs(polarity_score) < 0.5
    if topic == "entertainment":
        return polarity_score <= 0
    if topic == "sports":
        return abs(polarity_score) < 0.2
    if topic == "news":
        return abs(polarity_score) > 0.5
    return False


def _sentiment_topic_alignment_score(texts: list[str], polarity: np.ndarray) -> float:
    """Run 3, Feature 2: proportion of topic-tagged tweets whose sentiment
    breaks the expected pattern for their bucket (see _topic_mismatch).
    Higher = more misalignment = more AI-like. NaN if an account has zero
    topic-tagged tweets (filled with the column median - or 0.0 if every
    account in the run lacks bucket coverage - once the full per-account
    table is built).
    """
    mismatches = []
    for text, score in zip(texts, polarity):
        topic = _classify_topic(text)
        if topic is not None:
            mismatches.append(_topic_mismatch(topic, score))

    if not mismatches:
        return float("nan")
    return float(np.mean(mismatches))


def _fill_with_median_or_zero(series: pd.Series) -> pd.Series:
    """fillna(median), falling back to 0.0 if every value in the column is
    NaN (median of an all-NaN series is itself NaN, which would otherwise
    leave the column unfillable and crash model training downstream)."""
    median = series.median()
    return series.fillna(0.0 if pd.isna(median) else median)


def build_sentiment_features(
    tweets_df: pd.DataFrame,
    text_col: str = "text",
    max_tweets_per_account: int | None = config.MAX_TWEETS_PER_ACCOUNT_FOR_SENTIMENT,
) -> pd.DataFrame:
    """Per-account sentiment features computed from each account's tweet history.

    mean_polarity / polarity_std use VADER's compound score (built for social
    media text: slang, caps, emoji). subjectivity is TextBlob's 0 (factual) to
    1 (opinionated) score. polarity_std is the key discriminating feature per
    Dickerson et al. (2014): bots show low sentiment variance, humans high.

    If max_tweets_per_account is set, only the most recent N tweets (by
    created_at, when available) are scored per account, to keep runtime
    tractable on large corpora.
    """
    records = []
    threshold = config.EXTREME_SENTIMENT_THRESHOLD

    if max_tweets_per_account is not None and "created_at" in tweets_df.columns:
        # Parsed once, vectorized, over the whole corpus - not per account
        # group inside the loop below - since pandas can't auto-infer
        # Twitter's format and would otherwise fall back to a slow per-row
        # dateutil parse on every single loop iteration.
        tweets_df = tweets_df.copy()
        tweets_df["_parsed_created_at"] = pd.to_datetime(
            tweets_df["created_at"], format=config.TWITTER_CREATED_AT_FORMAT, errors="coerce", utc=True
        )

    grouped = tweets_df.groupby("account_key")
    for account_key, group in tqdm(
        grouped, total=grouped.ngroups, desc="Scoring sentiment per account", unit="account"
    ):
        if max_tweets_per_account is not None and len(group) > max_tweets_per_account:
            if "_parsed_created_at" in group.columns:
                group = group.sort_values("_parsed_created_at", ascending=False)
            group = group.head(max_tweets_per_account)

        # Chronological order (oldest first) for temporal_sentiment_drift's
        # windowing below; harmless for the other, order-independent stats.
        if "_parsed_created_at" in group.columns:
            group = group.sort_values("_parsed_created_at", ascending=True)

        texts = group[text_col].dropna().astype(str).tolist()
        if not texts:
            continue

        polarity = _polarity_scores(texts)
        subjectivity = _subjectivity_scores(texts)

        records.append(
            {
                "account_key": account_key,
                "mean_polarity": float(polarity.mean()),
                "polarity_std": float(polarity.std(ddof=0)),
                "pct_extreme": float(np.mean(np.abs(polarity) > threshold)),
                "pct_positive": float(np.mean(polarity > 0.05)),
                "pct_negative": float(np.mean(polarity < -0.05)),
                "pct_neutral": float(np.mean((polarity >= -0.05) & (polarity <= 0.05))),
                "mean_subjectivity": float(subjectivity.mean()),
                "subjectivity_std": float(subjectivity.std(ddof=0)),
                "contextual_consistency_score": _contextual_consistency_score(texts, polarity),
                "temporal_sentiment_drift": _temporal_sentiment_drift(polarity),
                "sentiment_topic_alignment_score": _sentiment_topic_alignment_score(texts, polarity),
            }
        )

    df = pd.DataFrame.from_records(records)
    for col in (
        "contextual_consistency_score",
        "temporal_sentiment_drift",
        "sentiment_topic_alignment_score",
    ):
        if col in df.columns:
            df[col] = _fill_with_median_or_zero(df[col])
    return df


def build_feature_sets(
    users_df: pd.DataFrame, tweets_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series]:
    """Return (X_metadata, X_metadata_plus_sentiment, y, account_keys), aligned row-for-row."""
    metadata = build_metadata_features(users_df)
    sentiment = build_sentiment_features(tweets_df)

    merged = metadata.merge(sentiment, on="account_key", how="inner").reset_index(drop=True)
    logger.info("Feature engineering: %d accounts have both metadata and sentiment features", len(merged))

    y = merged["label"].to_numpy()
    set_a = merged[config.METADATA_FEATURES].copy()
    set_b = merged[config.METADATA_FEATURES + config.SENTIMENT_FEATURES].copy()
    return set_a, set_b, y, merged["account_key"]

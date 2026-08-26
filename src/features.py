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
        created = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
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

    grouped = tweets_df.groupby("account_key")
    for account_key, group in tqdm(
        grouped, total=grouped.ngroups, desc="Scoring sentiment per account", unit="account"
    ):
        if max_tweets_per_account is not None and len(group) > max_tweets_per_account:
            if "created_at" in group.columns:
                order = pd.to_datetime(group["created_at"], errors="coerce", utc=True)
                group = group.loc[order.sort_values(ascending=False).index]
            group = group.head(max_tweets_per_account)

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
            }
        )

    return pd.DataFrame.from_records(records)


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

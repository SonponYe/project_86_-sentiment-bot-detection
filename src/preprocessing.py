"""Data cleaning (Phase 1, steps 3-4) and optional class-imbalance handling."""
from __future__ import annotations

import logging

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def clean_accounts(
    users_df: pd.DataFrame, tweets_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop accounts with fewer than MIN_TWEETS_PER_ACCOUNT tweets and drop duplicates."""
    users_df = users_df.drop_duplicates(subset="account_key")
    tweets_df = tweets_df.drop_duplicates()

    tweet_counts = tweets_df.groupby("account_key").size()
    eligible = tweet_counts[tweet_counts >= config.MIN_TWEETS_PER_ACCOUNT].index

    kept_users = users_df[users_df["account_key"].isin(eligible)].copy()
    kept_tweets = tweets_df[tweets_df["account_key"].isin(eligible)].copy()

    logger.info(
        "Cleaning: kept %d/%d accounts with >= %d tweets",
        len(kept_users),
        len(users_df),
        config.MIN_TWEETS_PER_ACCOUNT,
    )
    return kept_users, kept_tweets


def maybe_balance(X, y, use_smote: bool = True, random_state: int = config.RANDOM_STATE):
    """Oversample the minority class with SMOTE if classes are notably imbalanced.

    Only intended to be applied to a training split, never to a held-out test set.
    """
    counts = pd.Series(y).value_counts()
    imbalance_ratio = counts.max() / counts.min()
    if not use_smote or imbalance_ratio < 1.5:
        return X, y

    from imblearn.over_sampling import SMOTE

    logger.info("Class imbalance ratio %.2f >= 1.5, applying SMOTE", imbalance_ratio)
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

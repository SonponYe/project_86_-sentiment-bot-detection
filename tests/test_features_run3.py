"""Run 3: direct unit tests for the new feature-engineering helpers
(topic classification, contextual_consistency_score, temporal_sentiment_drift,
and the NaN-fallback logic), since the end-to-end smoke test's synthetic
phrases don't happen to hit the topic-bucket path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    _classify_topic,
    _contextual_consistency_score,
    _fill_with_median_or_zero,
    _temporal_sentiment_drift,
)


def test_classify_topic_matches_expected_bucket():
    assert _classify_topic("The senate voted on the new policy today") == "politics"
    assert _classify_topic("What a great goal in that match!") == "sports"
    assert _classify_topic("Loved the new album from that singer") == "entertainment"
    assert _classify_topic("Breaking: officials confirmed the news") == "news"
    assert _classify_topic("I had cereal for breakfast") is None


def test_contextual_consistency_score_needs_two_buckets():
    # Only sports-related text - a single bucket - must be NaN.
    texts = ["Great match today", "The team scored a goal", "What a game"]
    polarity = np.array([0.5, 0.6, 0.4])
    assert np.isnan(_contextual_consistency_score(texts, polarity))


def test_contextual_consistency_score_computes_std_across_buckets():
    texts = [
        "The senate voted today",  # politics
        "The senate policy passed",  # politics
        "What a great goal in the match",  # sports
        "The team won the game",  # sports
    ]
    polarity = np.array([0.2, 0.2, 0.8, 0.8])
    # politics mean = 0.2, sports mean = 0.8 -> std of [0.2, 0.8] = 0.3
    score = _contextual_consistency_score(texts, polarity)
    assert score == pytest.approx(0.3)


def test_temporal_sentiment_drift_needs_minimum_tweets():
    polarity = np.array([0.1] * 10)  # below MIN_TWEETS_FOR_TEMPORAL_DRIFT (15)
    assert np.isnan(_temporal_sentiment_drift(polarity))


def test_temporal_sentiment_drift_computes_std_across_windows():
    # 15 tweets, chronological order: window means should differ noticeably.
    polarity = np.array([0.0] * 5 + [0.5] * 5 + [1.0] * 5)
    score = _temporal_sentiment_drift(polarity)
    # window means = [0.0, 0.5, 1.0] -> std = ~0.408
    assert score == pytest.approx(0.4082, abs=1e-3)


def test_fill_with_median_or_zero_uses_median_when_available():
    series = pd.Series([1.0, np.nan, 3.0])
    filled = _fill_with_median_or_zero(series)
    assert filled.tolist() == [1.0, 2.0, 3.0]


def test_fill_with_median_or_zero_falls_back_to_zero_when_all_nan():
    series = pd.Series([np.nan, np.nan, np.nan])
    filled = _fill_with_median_or_zero(series)
    assert filled.tolist() == [0.0, 0.0, 0.0]

"""Shared paths and constants for the sentiment-assisted bot detection pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TWEETS_PER_ACCOUNT = 10
EXTREME_SENTIMENT_THRESHOLD = 0.6

# Cap on how many (most recent) tweets per account are scored for sentiment.
# VADER + TextBlob are CPU-bound per tweet; on the full Cresci-2017 corpus
# (~6.6M tweets) scoring every tweet is a multi-hour job regardless of disk
# speed. Sampling recent tweets keeps runtime tractable without materially
# changing the variance/consistency signal. Set to None to score every tweet.
MAX_TWEETS_PER_ACCOUNT_FOR_SENTIMENT = 200

# Thresholds used to classify the headline result (Section 8 of the dossier)
SIGNIFICANCE_ALPHA = 0.05
F1_IMPROVEMENT_THRESHOLD = 0.03  # "3%+" improvement, per the dossier's Outcome A

# Format of Cresci-2017's `created_at` timestamps (classic Twitter API v1.1),
# e.g. "Mon Jan 01 00:00:00 +0000 2015". Passing this explicitly to
# pd.to_datetime lets pandas use its fast vectorized parser instead of
# falling back to a slow per-row dateutil parse for every timestamp.
TWITTER_CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"

# Feature Set A (baseline)
#
# Run 3 - metadata-constrained scenario: followers_count, friends_count,
# statuses_count, and account_age_days removed. Across Run 1 and Run 2 these
# four dominated Random Forest importance (ranks 1-4); removing them
# simulates modern LLM agent accounts that mimic normal account statistics,
# leaving sentiment as the primary remaining signal. See
# findings/2026-08-28_run1_unscaled_features.md and _run2_scaled_features.md
# for the full-metadata baselines this is compared against.
METADATA_FEATURES = [
    "verified",
    "listed_count",
]

# Added on top of METADATA_FEATURES to form Feature Set B
SENTIMENT_FEATURES = [
    "mean_polarity",
    "polarity_std",
    "pct_extreme",
    "pct_positive",
    "pct_negative",
    "pct_neutral",
    "mean_subjectivity",
    "subjectivity_std",
    # Run 3 additions
    "contextual_consistency_score",
    "temporal_sentiment_drift",
    "sentiment_topic_alignment_score",
]

# Run 3: minimum tweets across an account's chronological windows for
# temporal_sentiment_drift to be computed; below this, NaN (filled with the
# column median at build time).
MIN_TWEETS_FOR_TEMPORAL_DRIFT = 15

# Run 3: simple keyword buckets for contextual_consistency_score. A tweet is
# assigned to the first bucket (in this order) whose keyword appears in it;
# tweets matching none of these aren't counted in any bucket.
TOPIC_KEYWORDS = {
    "politics": [
        "president", "election", "vote", "senate", "congress", "government",
        "policy", "democrat", "republican", "politician", "minister", "parliament",
    ],
    "sports": [
        "game", "team", "score", "match", "player", "coach", "championship",
        "league", "tournament", "goal",
    ],
    "entertainment": [
        "movie", "music", "celebrity", "concert", "album", "actor", "actress",
        "film", "song", "singer", "hollywood",
    ],
    "news": [
        "breaking", "report", "announced", "official statement", "confirmed",
        "according to", "sources say",
    ],
}

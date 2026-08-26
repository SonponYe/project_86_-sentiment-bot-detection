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

# Feature Set A (baseline)
METADATA_FEATURES = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "account_age_days",
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
]

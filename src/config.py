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

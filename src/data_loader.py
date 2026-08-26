"""Load and merge the Cresci-2017 bot/human account dataset (Phase 1, step 1-2).

Expected layout under data/raw/, one subdirectory per class, each holding
users.csv and tweets.csv as distributed by the IU Bot Repository
(https://botometer.osome.iu.edu/bot-repository/datasets.html):

data/raw/
  genuine_accounts.csv/
    users.csv
    tweets.csv
  social_spambots_1.csv/
    users.csv
    tweets.csv
  traditional_spambots_1.csv/
    users.csv
    tweets.csv
  ...

Any subdirectory whose name contains "genuine" is labelled human (0);
every other subdirectory is labelled bot (1).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from . import config

logger = logging.getLogger(__name__)

USER_ID_CANDIDATES = ("id", "user_id")
TWEET_USER_ID_CANDIDATES = ("user_id", "id")


def _first_present(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"None of {candidates} found in columns {list(columns)}")


def _label_for_dir(dirname: str) -> int:
    """0 = human/genuine, 1 = bot."""
    return 0 if "genuine" in dirname.lower() else 1


def load_cresci2017(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every account subset found under raw_dir.

    Returns (users_df, tweets_df). Both carry a shared `account_key` column
    (subset-prefixed user id, since raw ids can collide across subsets);
    users_df additionally carries `label` (0=human, 1=bot) and
    `source_subset` (the originating folder name).
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else Path(config.RAW_DATA_DIR)
    subset_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir()) if raw_dir.exists() else []
    if not subset_dirs:
        raise FileNotFoundError(
            f"No dataset subfolders found in {raw_dir}. Download Cresci-2017 from "
            "the IU Bot Repository and extract each class (genuine_accounts.csv, "
            "social_spambots_*.csv, traditional_spambots_*.csv) as a subfolder of "
            "data/raw/. See README.md for the exact layout expected."
        )

    all_users = []
    all_tweets = []
    progress = tqdm(subset_dirs, desc="Loading Cresci-2017 subsets", unit="subset")
    for subset_dir in progress:
        progress.set_postfix_str(subset_dir.name)
        users_path = subset_dir / "users.csv"
        tweets_path = subset_dir / "tweets.csv"
        if not users_path.exists() or not tweets_path.exists():
            logger.warning("Skipping %s: missing users.csv or tweets.csv", subset_dir)
            continue

        users = pd.read_csv(users_path, low_memory=False)
        tweets = pd.read_csv(tweets_path, low_memory=False)

        user_id_col = _first_present(users.columns, USER_ID_CANDIDATES)
        tweet_user_id_col = _first_present(tweets.columns, TWEET_USER_ID_CANDIDATES)
        users = users.rename(columns={user_id_col: "user_id"})
        tweets = tweets.rename(columns={tweet_user_id_col: "user_id"})

        prefix = subset_dir.name
        users["account_key"] = prefix + "_" + users["user_id"].astype(str)
        tweets["account_key"] = prefix + "_" + tweets["user_id"].astype(str)
        users["label"] = _label_for_dir(prefix)
        users["source_subset"] = prefix

        all_users.append(users)
        all_tweets.append(tweets)

    if not all_users:
        raise FileNotFoundError(f"No valid users.csv/tweets.csv pairs found under {raw_dir}")

    users_df = pd.concat(all_users, ignore_index=True, sort=False)
    tweets_df = pd.concat(all_tweets, ignore_index=True, sort=False)
    logger.info(
        "Loaded %d accounts (%d bot, %d human) and %d tweets from %d subsets",
        len(users_df),
        int((users_df["label"] == 1).sum()),
        int((users_df["label"] == 0).sum()),
        len(tweets_df),
        len(subset_dirs),
    )
    return users_df, tweets_df

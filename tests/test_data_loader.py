"""Regression test: real Cresci-2017 CSV dumps contain occasional invalid UTF-8
byte sequences (found while running the real dataset - genuine_accounts.csv's
tweets.csv failed with `UnicodeDecodeError: invalid continuation byte`).
load_cresci2017 must tolerate this instead of crashing the whole load.
"""
from __future__ import annotations

from src.data_loader import load_cresci2017


def test_load_cresci2017_tolerates_invalid_utf8_bytes(tmp_path):
    subset_dir = tmp_path / "genuine_accounts.csv"
    subset_dir.mkdir()

    (subset_dir / "users.csv").write_bytes(b'"id","followers_count"\n"1","10"\n')

    # 0xf3 is a valid UTF-8 lead byte for a 4-byte sequence but is followed here
    # by plain ASCII, not continuation bytes - exactly the malformed pattern
    # found in the real dataset.
    (subset_dir / "tweets.csv").write_bytes(
        b'"id","text","user_id"\n"100","hello \xf3 world","1"\n'
    )

    users_df, tweets_df = load_cresci2017(tmp_path)

    assert len(users_df) == 1
    assert len(tweets_df) == 1
    assert "�" in tweets_df.iloc[0]["text"]  # replacement char, not a crash

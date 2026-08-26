"""Smoke test: run the full pipeline on a small synthetic dataset shaped like
Cresci-2017, so the pipeline's correctness can be verified without needing
the real (multi-gigabyte) dataset downloaded first.
"""
from __future__ import annotations

import random

import pandas as pd
import pytest

from src import pipeline, report

POSITIVE_PHRASES = [
    "I love this so much, best day ever!",
    "This is absolutely wonderful and exciting!",
    "I am so happy right now, amazing news!",
]
NEGATIVE_PHRASES = [
    "This is terrible, I hate it so much.",
    "Worst experience ever, absolutely awful.",
    "I am furious about this disaster.",
]
NEUTRAL_PHRASES = [
    "The meeting is scheduled for 3pm today.",
    "I went to the store and bought some bread.",
    "The weather report says it might rain tomorrow.",
]
HUMAN_PHRASE_POOL = POSITIVE_PHRASES + NEGATIVE_PHRASES + NEUTRAL_PHRASES
BOT_PHRASE_POOL = ["Buy now! Best deal ever! Click here!!!"] * 3


def _make_subset(root, name: str, n_accounts: int, phrase_pool, tweets_per_account: int = 15):
    subset_dir = root / name
    subset_dir.mkdir()

    rng = random.Random(0)
    users_rows = []
    tweets_rows = []
    for i in range(n_accounts):
        user_id = f"{name}_{i}"
        users_rows.append(
            {
                "id": user_id,
                "followers_count": rng.randint(10, 5000),
                "friends_count": rng.randint(10, 3000),
                "statuses_count": rng.randint(50, 10000),
                "created_at": "Mon Jan 01 00:00:00 +0000 2015",
                "verified": False,
                "listed_count": rng.randint(0, 50),
            }
        )
        for t in range(tweets_per_account):
            tweets_rows.append(
                {"id": f"{user_id}_tweet_{t}", "user_id": user_id, "text": rng.choice(phrase_pool)}
            )

    pd.DataFrame(users_rows).to_csv(subset_dir / "users.csv", index=False)
    pd.DataFrame(tweets_rows).to_csv(subset_dir / "tweets.csv", index=False)


@pytest.fixture()
def synthetic_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_subset(raw_dir, "genuine_accounts.csv", n_accounts=30, phrase_pool=HUMAN_PHRASE_POOL)
    _make_subset(raw_dir, "social_spambots_1.csv", n_accounts=30, phrase_pool=BOT_PHRASE_POOL)
    return raw_dir


def test_pipeline_runs_end_to_end(synthetic_raw_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.config, "RESULTS_TABLES_DIR", tmp_path / "results_tables")
    monkeypatch.setattr(pipeline.config, "RESULTS_FIGURES_DIR", tmp_path / "results_figures")

    outputs = pipeline.run_pipeline(raw_dir=synthetic_raw_dir, use_smote=False)

    summary = outputs["summary"]
    assert len(summary) == 6  # 3 classifiers x 2 feature sets
    assert set(summary["feature_set"]) == {"metadata_only", "metadata_plus_sentiment"}
    assert set(summary["model"]) == {"logistic_regression", "random_forest", "svm"}
    assert summary["f1"].between(0, 1).all()

    assert len(outputs["mcnemar"]) == 3
    assert outputs["mcnemar"]["p_value"].between(0, 1).all()

    assert not outputs["feature_importance"].empty
    assert (pipeline.config.RESULTS_TABLES_DIR / "experiment_summary.csv").exists()
    assert (pipeline.config.RESULTS_TABLES_DIR / "mcnemar_significance.csv").exists()
    assert (pipeline.config.RESULTS_TABLES_DIR / "feature_importance.csv").exists()

    confusion_matrices = list(pipeline.config.RESULTS_FIGURES_DIR.glob("confusion_matrix_*.png"))
    assert len(confusion_matrices) == 6


def test_report_generates_from_pipeline_output(synthetic_raw_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.config, "RESULTS_TABLES_DIR", tmp_path / "results_tables")
    monkeypatch.setattr(pipeline.config, "RESULTS_FIGURES_DIR", tmp_path / "results_figures")

    pipeline.run_pipeline(raw_dir=synthetic_raw_dir, use_smote=False)
    result = report.generate_report()

    assert result["outcome"] in {"A", "B", "C"}
    assert len(result["classified"]) == 3
    assert set(result["classified"]["verdict"]).issubset(
        {"significant_improvement", "significant_but_small", "improvement_not_significant", "no_improvement"}
    )

    report_path = report.config.RESULTS_TABLES_DIR / "interpretation_report.md"
    assert report_path.exists()
    assert report.OUTCOME_LABELS[result["outcome"]] in report_path.read_text(encoding="utf-8")

    assert (report.config.RESULTS_FIGURES_DIR / "f1_comparison.png").exists()
    assert (report.config.RESULTS_FIGURES_DIR / "feature_importance.png").exists()

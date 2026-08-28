"""End-to-end pipeline: load -> clean -> engineer features -> train -> evaluate -> save results.

Implements the within-subjects design from Section 7.1 of the research
dossier: the SAME accounts, SAME stratified split and SAME classifiers are
used for both feature sets — only the features change. Feature Set A and B
therefore share one train/test split (computed once, on account indices),
so predictions on the two feature sets can be compared example-for-example
with McNemar's test. SMOTE, if triggered, is fit on the training portion
only, never on the held-out test set.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from . import config, evaluation, features, modeling, preprocessing
from .data_loader import load_cresci2017

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _save_confusion_matrices(experiment_results, out_dir) -> None:
    for r in experiment_results:
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            r.y_test, r.y_pred, display_labels=["human", "bot"], ax=ax, colorbar=False
        )
        ax.set_title(f"{r.model_name} — {r.feature_set}")
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_matrix_{r.feature_set}_{r.model_name}.png", dpi=150)
        plt.close(fig)


def _load_or_build_features(raw_dir, use_cache: bool):
    """Feature engineering is the expensive step (sentiment scoring over the
    whole tweet corpus). Cache its output so re-running the pipeline after a
    downstream bugfix doesn't repeat it.
    """
    cache_path = config.PROCESSED_DATA_DIR / "engineered_features.csv"
    if use_cache and cache_path.exists():
        logger.info("Loading cached engineered features from %s", cache_path)
        merged = pd.read_csv(cache_path)
        y = merged["label"].to_numpy()
        set_a = merged[config.METADATA_FEATURES].copy()
        set_b = merged[config.METADATA_FEATURES + config.SENTIMENT_FEATURES].copy()
        return set_a, set_b, y, merged["account_key"]

    users_df, tweets_df = load_cresci2017(raw_dir)
    users_df, tweets_df = preprocessing.clean_accounts(users_df, tweets_df)
    set_a, set_b, y, account_keys = features.build_feature_sets(users_df, tweets_df)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_df = set_b.copy()
    cache_df.insert(0, "account_key", account_keys.to_numpy())
    cache_df["label"] = y
    cache_df.to_csv(cache_path, index=False)
    logger.info("Cached engineered features to %s", cache_path)

    return set_a, set_b, y, account_keys


def run_pipeline(raw_dir=None, use_smote: bool = True, use_cache: bool = True) -> dict:
    raw_dir = raw_dir if raw_dir is not None else config.RAW_DATA_DIR
    config.RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    set_a, set_b, y, account_keys = _load_or_build_features(raw_dir, use_cache)
    logger.info(
        "Full dataset class balance: %s",
        pd.Series(y).value_counts().sort_index().to_dict(),
    )

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    y_train, y_test = y[idx_train], y[idx_test]

    def split(X: pd.DataFrame):
        return (
            X.iloc[idx_train].reset_index(drop=True),
            X.iloc[idx_test].reset_index(drop=True),
        )

    set_a_train, set_a_test = split(set_a)
    set_b_train, set_b_test = split(set_b)

    set_a_train, set_a_test = preprocessing.scale_features(set_a_train, set_a_test)
    set_b_train, set_b_test = preprocessing.scale_features(set_b_train, set_b_test)

    set_a_train_bal, y_a_train_bal = preprocessing.maybe_balance(set_a_train, y_train, use_smote=use_smote)
    set_b_train_bal, y_b_train_bal = preprocessing.maybe_balance(set_b_train, y_train, use_smote=use_smote)

    logger.info(
        "Feature Set A training class balance: %s",
        pd.Series(y_a_train_bal).value_counts().sort_index().to_dict(),
    )
    logger.info(
        "Feature Set B training class balance: %s",
        pd.Series(y_b_train_bal).value_counts().sort_index().to_dict(),
    )

    results_a = modeling.run_experiments(
        set_a_train_bal, y_a_train_bal, set_a_test, y_test, "metadata_only"
    )
    results_b = modeling.run_experiments(
        set_b_train_bal, y_b_train_bal, set_b_test, y_test, "metadata_plus_sentiment"
    )

    summary = evaluation.summarize_experiments(results_a + results_b)
    summary_path = config.RESULTS_TABLES_DIR / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved experiment summary to %s", summary_path)

    _save_confusion_matrices(results_a + results_b, config.RESULTS_FIGURES_DIR)
    logger.info("Saved confusion matrices to %s", config.RESULTS_FIGURES_DIR)

    # McNemar's test per model: metadata-only vs metadata+sentiment (Section 7.4).
    # Valid because both feature sets share the same held-out test set.
    results_a_by_model = {r.model_name: r for r in results_a}
    results_b_by_model = {r.model_name: r for r in results_b}
    mcnemar_rows = []
    for model_name, r_a in results_a_by_model.items():
        r_b = results_b_by_model[model_name]
        stat = evaluation.mcnemar_significance(y_test, r_a.y_pred, r_b.y_pred)
        mcnemar_rows.append({"model": model_name, **stat})

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_path = config.RESULTS_TABLES_DIR / "mcnemar_significance.csv"
    mcnemar_df.to_csv(mcnemar_path, index=False)
    logger.info("Saved McNemar's test results to %s", mcnemar_path)

    rf_b = next(r for r in results_b if r.model_name == "random_forest")
    importance = evaluation.feature_importance_report(rf_b.model, set_b.columns.tolist())
    importance_path = config.RESULTS_TABLES_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    logger.info("Saved feature importance to %s", importance_path)

    return {
        "summary": summary,
        "mcnemar": mcnemar_df,
        "feature_importance": importance,
    }


if __name__ == "__main__":
    run_pipeline()

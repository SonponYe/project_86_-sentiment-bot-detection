"""Phase 4 / Section 7: metrics, McNemar significance test, feature importance."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def summarize_experiments(experiment_results) -> pd.DataFrame:
    rows = []
    for r in experiment_results:
        metrics = compute_metrics(r.y_test, r.y_pred, r.y_proba)
        rows.append(
            {
                "model": r.model_name,
                "feature_set": r.feature_set,
                "cv_f1_mean": r.cv_f1_scores.mean(),
                "cv_f1_std": r.cv_f1_scores.std(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def mcnemar_significance(y_true, y_pred_a, y_pred_b) -> dict:
    """McNemar's test comparing two classifiers' predictions on the SAME test set.

    Section 7.4: used to check whether metadata+sentiment predictions differ
    significantly from metadata-only predictions for the same model type.
    Requires y_pred_a and y_pred_b to be predictions on identical, aligned
    test examples.
    """
    y_true = np.asarray(y_true)
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    both_correct = int(np.sum(correct_a & correct_b))
    only_a_correct = int(np.sum(correct_a & ~correct_b))
    only_b_correct = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, only_a_correct], [only_b_correct, both_wrong]]
    exact = (only_a_correct + only_b_correct) < 25
    result = mcnemar(table, exact=exact, correction=True)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "table": table}


def feature_importance_report(model, feature_names) -> pd.DataFrame:
    """Section 7.5: rank features by Random Forest importance."""
    if not hasattr(model, "feature_importances_"):
        raise TypeError(f"{type(model).__name__} has no feature_importances_ (use Random Forest)")
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

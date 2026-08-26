"""Phase 3: train Logistic Regression, Random Forest and SVM on a given feature set."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from tqdm.auto import tqdm

from . import config


def get_classifiers(random_state: int = config.RANDOM_STATE) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state),
    }


@dataclass
class ExperimentResult:
    model_name: str
    feature_set: str
    model: object
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    cv_f1_scores: np.ndarray


def run_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_set_name: str,
    random_state: int = config.RANDOM_STATE,
) -> list[ExperimentResult]:
    """Fit each classifier on (X_train, y_train); evaluate on the held-out (X_test, y_test).

    5-fold stratified cross-validation (F1) on the training set gives a
    variance estimate independent of the single test split (Section 7.2).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_test_arr = np.asarray(y_test)

    results = []
    classifiers = get_classifiers(random_state)
    progress = tqdm(classifiers.items(), desc=f"Training ({feature_set_name})", unit="model")
    for name, model in progress:
        progress.set_postfix_str(name)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append(
            ExperimentResult(
                model_name=name,
                feature_set=feature_set_name,
                model=model,
                y_test=y_test_arr,
                y_pred=y_pred,
                y_proba=y_proba,
                cv_f1_scores=cv_scores,
            )
        )
    return results

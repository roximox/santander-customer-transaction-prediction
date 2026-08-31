"""Single-use evaluation helpers for an already locked final model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_label(y: pd.Series) -> Any:
    """Return the same deterministic positive label used by shared evaluation."""
    labels = list(pd.unique(y))
    for preferred in ("True", True, 1):
        if preferred in labels:
            return preferred
    return sorted(labels, key=lambda value: (type(value).__name__, repr(value)))[-1]


def evaluate_fitted_model_once(
    estimator: BaseEstimator,
    X_final: pd.DataFrame,
    y_final: pd.Series,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compute the fixed project metrics without fitting or tuning the estimator."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1.")
    if not X_final.index.equals(y_final.index):
        raise ValueError("Final-test feature and target indexes must match.")
    if not hasattr(estimator, "classes_") or not hasattr(estimator, "predict_proba"):
        raise ValueError("estimator must be fitted and expose predict_proba.")

    positive = positive_label(y_final)
    classes = list(estimator.classes_)
    if positive not in classes:
        raise ValueError(f"Positive label {positive!r} is absent from estimator classes.")
    scores = np.asarray(estimator.predict_proba(X_final))[:, classes.index(positive)]
    truth = np.asarray(y_final == positive, dtype=int)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
    }

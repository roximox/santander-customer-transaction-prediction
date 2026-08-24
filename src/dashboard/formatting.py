"""Formatting helpers shared by dashboard pages."""

from __future__ import annotations

from typing import Any

import pandas as pd


METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Average Precision",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def format_number(value: Any, digits: int = 4, missing: str = "Not recorded") -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return missing
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_status(value: Any) -> str:
    return str(value or "unknown").replace("_", " ").upper()

"""Tests for the fixed, single-use final metric calculation."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.final_evaluation import evaluate_fitted_model_once


ROOT = Path(__file__).resolve().parents[1]


def test_final_metrics_are_computed_from_a_fitted_estimator() -> None:
    X = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0]})
    y = pd.Series(["False", "False", "True", "True"], index=X.index)
    estimator = LogisticRegression(random_state=42).fit(X, y)

    result = evaluate_fitted_model_once(estimator, X, y, threshold=0.5)

    assert result["roc_auc"] == 1.0
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert result["threshold"] == 0.5


def test_final_evaluation_rejects_invalid_inputs() -> None:
    X = pd.DataFrame({"x": [-1.0, 1.0]})
    y = pd.Series([False, True], index=X.index)
    estimator = LogisticRegression(random_state=42).fit(X, y)

    with pytest.raises(ValueError, match="threshold"):
        evaluate_fitted_model_once(estimator, X, y, threshold=1.0)
    with pytest.raises(ValueError, match="indexes"):
        evaluate_fitted_model_once(estimator, X, y.set_axis([2, 3]), threshold=0.5)


def test_scientific_conclusions_reference_the_single_recorded_result() -> None:
    conclusions = (ROOT / "reports/scientific_conclusions.md").read_text(encoding="utf-8")

    assert "M04-HGB-002" in conclusions
    assert "FINAL-M04-HGB-002-001" in conclusions
    assert "0.891214" in conclusions
    assert "exactly once" in conclusions
    assert re.search(r"selection\s+was not reopened", conclusions, re.IGNORECASE)

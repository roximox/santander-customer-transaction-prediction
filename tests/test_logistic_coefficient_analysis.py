"""Offline tests for Logistic Regression coefficient analysis."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logistic_coefficient_analysis import (
    EPSILON,
    audit_configurations_by_fold,
    create_coefficient_audit_pipeline,
    create_l1_sparsity_figure,
    save_analysis_figure,
    strict_json_records,
    summarize_coefficient_stability,
    summarize_l1_sparsity,
)


@pytest.fixture(scope="module")
def analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    values, target = make_classification(n_samples=100, n_features=8, n_informative=5, random_state=42)
    X = pd.DataFrame(values, columns=[f"feature_{index}" for index in range(8)])
    y = pd.Series(target, index=X.index)
    original_X, original_y = X.copy(), y.copy()
    result = audit_configurations_by_fold(
        X, y, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    )
    pd.testing.assert_frame_equal(X, original_X); pd.testing.assert_series_equal(y, original_y)
    return result


def test_audit_captures_iterations_convergence_coefficients_and_names(analysis: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    audit, coefficients = analysis
    assert len(audit) == 8 and audit["n_iter"].ge(1).all()
    assert audit["convergence_warning"].map(type).eq(bool).all()
    assert set(coefficients["feature"]) == {f"feature_{index}" for index in range(8)}
    assert {"coefficient", "absolute_coefficient", "is_zero", "converged"} <= set(coefficients)


def test_new_and_legacy_regularization_formulations_are_equivalent() -> None:
    values, target = make_classification(n_samples=80, n_features=5, random_state=7)
    for penalty, ratio in (("l2", 0.0), ("l1", 1.0)):
        legacy = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(penalty=penalty, solver="saga", C=0.5, max_iter=2000, random_state=42))])
        current = create_coefficient_audit_pipeline(l1_ratio=ratio, C=0.5, class_weight=None)
        # Deprecation warnings differ across supported scikit-learn releases;
        # predictive/coef equivalence is the compatibility contract.
        legacy.fit(values, target)
        current.fit(values, target)
        assert np.array_equal(legacy.predict(values), current.predict(values))
        assert np.allclose(legacy.named_steps["classifier"].coef_, current.named_steps["classifier"].coef_, atol=1e-8)


def test_stability_definitions_are_exact() -> None:
    coefficients = pd.DataFrame({
        "configuration_id": ["A"] * 5, "feature": ["x"] * 5,
        "coefficient": [1.0, 2.0, 0.0, 1.0, -1.0],
    })
    result = summarize_coefficient_stability(coefficients).iloc[0]
    assert result["sign_consistency"] == pytest.approx(3 / 5)
    assert result["selection_frequency"] == pytest.approx(4 / 5)
    values = np.array([1.0, 2.0, 0.0, 1.0, -1.0])
    assert result["stability_ratio"] == pytest.approx(abs(values.mean()) / (values.std(ddof=0) + EPSILON))


def test_l1_exact_zero_audit_and_sparsity_summary(analysis: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, coefficients = analysis
    selected = coefficients[coefficients["configuration_id"] == "LR-SELECTED-AP"]
    assert selected["is_zero"].equals(selected["coefficient"].eq(0.0))
    folds, overall = summarize_l1_sparsity(coefficients)
    assert len(folds) == 4
    assert set(overall) == {"LR-SELECTED-AP", "LR-L1-WEAK-REG"}
    assert all(item["union_feature_count"] >= item["intersection_feature_count"] for item in overall.values())
    json.dumps(overall, allow_nan=False)


def test_results_serializable_and_figure_creatable(analysis: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, coefficients = analysis
    sparsity, _ = summarize_l1_sparsity(coefficients)
    json.dumps(strict_json_records(sparsity), allow_nan=False)
    monkeypatch.setattr("src.logistic_coefficient_analysis.get_project_root", lambda: tmp_path)
    output = save_analysis_figure(create_l1_sparsity_figure(sparsity), "figure.pdf")
    assert output.read_bytes().startswith(b"%PDF")
    plt.close("all")


def test_api_has_no_final_test_arguments() -> None:
    parameters = set(inspect.signature(audit_configurations_by_fold).parameters)
    assert not parameters & {"X_test", "y_test", "test_data", "final_test"}

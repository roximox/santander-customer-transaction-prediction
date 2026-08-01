"""Offline tests for M01-LR-001 construction and reporting helpers."""

from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.logistic_baseline as baseline
from scripts.run_logistic_baseline import main, refuse_existing_outputs
from src.experiments import run_experiment
from src.logistic_baseline import (
    COMPARISON_COLUMNS,
    build_logistic_comparison,
    calculate_baseline_improvements,
    create_logistic_baseline_pipeline,
    save_logistic_comparison,
    save_logistic_cv_figure,
    save_logistic_vs_dummy_figure,
)
from src.modeling import create_logistic_regression_pipeline
from src.validation import create_stratified_cv


def _summary(experiment_id: str, auc: float, ap: float, fit_time: float = 1.0) -> dict[str, object]:
    metric_names = (
        "roc_auc", "average_precision", "f1", "precision", "recall",
        "accuracy", "balanced_accuracy",
    )
    metrics = {
        name: {
            "train_mean": auc + 0.02 if name == "roc_auc" else ap,
            "train_std": 0.01,
            "validation_mean": auc if name == "roc_auc" else ap,
            "validation_std": 0.01,
        }
        for name in metric_names
    }
    return {
        "experiment_id": experiment_id,
        "model_name": experiment_id,
        "metrics": metrics,
        "fit_time_mean": fit_time,
        "score_time_mean": 0.1,
    }


def _comparison() -> pd.DataFrame:
    return build_logistic_comparison(
        [
            _summary("M01-DUMMY-001", 0.5, 0.1, 0.2),
            _summary("M01-DUMMY-002", 0.5, 0.1, 0.2),
            _summary("M01-DUMMY-003", 0.5, 0.1, 0.2),
            _summary("M01-DUMMY-004", 0.5, 0.1, 0.2),
            _summary("M01-LR-001", 0.7, 0.2, 2.0),
        ]
    )


def test_baseline_pipeline_has_exact_unfitted_configuration() -> None:
    pipeline = create_logistic_baseline_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.penalty == "l2"
    assert classifier.C == 1.0
    assert classifier.class_weight is None
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42
    assert pipeline.named_steps["scaler"].with_mean is True
    assert pipeline.named_steps["scaler"].with_std is True
    assert not hasattr(classifier, "coef_")


def test_comparison_columns_gap_and_dummy_rows() -> None:
    comparison = _comparison()
    assert tuple(comparison.columns) == COMPARISON_COLUMNS
    assert set(comparison["experiment_id"]) == {
        "M01-DUMMY-001", "M01-DUMMY-002", "M01-DUMMY-003",
        "M01-DUMMY-004", "M01-LR-001",
    }
    logistic = comparison.set_index("experiment_id").loc["M01-LR-001"]
    assert logistic["roc_auc_generalization_gap"] == pytest.approx(0.02)
    json.dumps(comparison.to_dict(orient="records"), allow_nan=False)


def test_improvements_are_calculated_absolutely_and_relatively() -> None:
    improvements = calculate_baseline_improvements(_comparison())
    assert improvements["roc_auc_absolute_improvement"] == pytest.approx(0.2)
    assert improvements["average_precision_absolute_improvement"] == pytest.approx(0.1)
    assert improvements["average_precision_relative_improvement"] == pytest.approx(1.0)
    assert improvements["logistic_to_dummy_fit_time_ratio"] == pytest.approx(10.0)


def test_comparison_save_is_relative_serializable_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baseline, "get_project_root", lambda: tmp_path)
    csv_path, json_path = save_logistic_comparison(
        _comparison(), csv_path="comparison.csv", json_path="comparison.json"
    )
    assert csv_path.is_file() and json_path.is_file()
    assert str(tmp_path) not in json_path.read_text(encoding="utf-8")
    json.loads(json_path.read_text(encoding="utf-8"))
    with pytest.raises(FileExistsError, match="already exists"):
        save_logistic_comparison(
            _comparison(), csv_path="comparison.csv", json_path="comparison.json"
        )


def test_figures_are_created_in_temporary_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baseline, "get_project_root", lambda: tmp_path)
    comparison_figure = save_logistic_vs_dummy_figure(
        _comparison(), "comparison.pdf"
    )
    folds = pd.DataFrame(
        {
            "fold": [1, 2],
            "validation_roc_auc": [0.7, 0.71],
            "validation_average_precision": [0.2, 0.21],
        }
    )
    cv_figure = save_logistic_cv_figure(folds, "folds.pdf")
    assert comparison_figure.read_bytes().startswith(b"%PDF")
    assert cv_figure.read_bytes().startswith(b"%PDF")


def test_script_public_api_has_no_final_test_parameters() -> None:
    for function in (main, refuse_existing_outputs):
        parameters = inspect.signature(function).parameters
        assert "X_test" not in parameters and "y_test" not in parameters


def test_convergence_warning_is_detected_and_not_silently_masked() -> None:
    values, labels = make_classification(
        n_samples=200, n_features=20, n_informative=10, random_state=42
    )
    X = pd.DataFrame(values)
    y = pd.Series(labels, index=X.index)
    estimator = create_logistic_regression_pipeline(max_iter=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _, summary = run_experiment(
            estimator, X, y,
            experiment_id="TECH-LR-CONVERGENCE",
            model_name="Technical convergence check",
            member="test", branch="test",
            cv=create_stratified_cv(n_splits=2), n_jobs=1,
        )
    assert summary["convergence_warning_detected"] is True
    assert summary["convergence_warning_messages"]
    assert any(issubclass(item.category, ConvergenceWarning) for item in caught)

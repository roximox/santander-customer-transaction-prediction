"""Offline tests for the controlled Logistic class-weight experiment."""

from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning

import src.logistic_class_weight as class_weight
from scripts.run_logistic_class_weight_comparison import main, refuse_existing_outputs
from src.experiments import run_experiment
from src.logistic_baseline import create_logistic_baseline_pipeline
from src.logistic_class_weight import (
    COMPARISON_COLUMNS,
    build_class_weight_comparison,
    create_balanced_logistic_pipeline,
    save_class_weight_comparison,
    save_class_weight_cv_figure,
    save_class_weight_metrics_figure,
)
from src.modeling import create_logistic_regression_pipeline
from src.validation import create_stratified_cv


def _summary(experiment_id: str, shift: float, class_weight_value: str | None) -> dict[str, object]:
    base = {
        "roc_auc": 0.80, "average_precision": 0.40, "precision": 0.60,
        "recall": 0.30, "f1": 0.40, "accuracy": 0.90,
        "balanced_accuracy": 0.64,
    }
    metrics = {
        name: {
            "train_mean": value + shift + (0.01 if name == "roc_auc" else 0),
            "train_std": 0.01,
            "validation_mean": value + shift,
            "validation_std": 0.02,
        }
        for name, value in base.items()
    }
    return {
        "experiment_id": experiment_id,
        "metrics": metrics,
        "fit_time_mean": 1.0 + shift,
        "estimator_parameters": {"classifier__class_weight": class_weight_value},
    }


def _comparison() -> pd.DataFrame:
    return build_class_weight_comparison(
        _summary("M01-LR-001", 0.0, None),
        _summary("M01-LR-002", 0.05, "balanced"),
    )


def test_balanced_pipeline_changes_only_class_weight() -> None:
    baseline = create_logistic_baseline_pipeline()
    balanced = create_balanced_logistic_pipeline()
    base_params = baseline.get_params(deep=True)
    balanced_params = balanced.get_params(deep=True)
    assert balanced_params["classifier__class_weight"] == "balanced"
    assert list(balanced.named_steps) == ["scaler", "classifier"]
    compared = (
        "classifier__penalty", "classifier__C", "classifier__solver",
        "classifier__max_iter", "classifier__random_state",
        "scaler__with_mean", "scaler__with_std",
    )
    assert all(base_params[name] == balanced_params[name] for name in compared)


def test_comparison_contains_required_columns_and_correct_deltas() -> None:
    table = _comparison()
    assert tuple(table.columns) == COMPARISON_COLUMNS
    assert table["experiment_id"].tolist() == ["M01-LR-001", "M01-LR-002"]
    balanced = table.iloc[1]
    for metric in (
        "roc_auc", "average_precision", "precision", "recall", "f1",
        "accuracy", "balanced_accuracy",
    ):
        assert balanced[f"delta_{metric}"] == pytest.approx(0.05)
    assert balanced["generalization_gap"] == pytest.approx(0.01)
    json.dumps(table.to_dict(orient="records"), allow_nan=False)


def test_save_is_relative_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(class_weight, "get_project_root", lambda: tmp_path)
    csv_file, json_file = save_class_weight_comparison(
        _comparison(), csv_path="comparison.csv", json_path="comparison.json"
    )
    assert csv_file.is_file() and json_file.is_file()
    assert str(tmp_path) not in json_file.read_text(encoding="utf-8")
    json.loads(json_file.read_text(encoding="utf-8"))
    with pytest.raises(FileExistsError, match="already exists"):
        save_class_weight_comparison(
            _comparison(), csv_path="comparison.csv", json_path="comparison.json"
        )


def test_figures_are_created_in_temporary_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(class_weight, "get_project_root", lambda: tmp_path)
    metrics_path = save_class_weight_metrics_figure(_comparison(), "metrics.pdf")
    folds = pd.DataFrame(
        {
            "fold": [1, 2], "validation_roc_auc": [0.8, 0.81],
            "validation_average_precision": [0.4, 0.41],
            "validation_recall": [0.3, 0.31], "validation_f1": [0.4, 0.41],
        }
    )
    balanced_folds = folds.copy()
    for column in balanced_folds.columns.drop("fold"):
        balanced_folds[column] += 0.01
    cv_path = save_class_weight_cv_figure(folds, balanced_folds, "cv.pdf")
    assert metrics_path.read_bytes().startswith(b"%PDF")
    assert cv_path.read_bytes().startswith(b"%PDF")


def test_script_api_has_no_final_test_parameters() -> None:
    for function in (main, refuse_existing_outputs):
        parameters = inspect.signature(function).parameters
        assert "X_test" not in parameters and "y_test" not in parameters


def test_balanced_convergence_warning_is_detectable() -> None:
    values, labels = make_classification(
        n_samples=120, n_features=10, n_informative=6, weights=[0.8, 0.2],
        random_state=42,
    )
    X, y = pd.DataFrame(values), pd.Series(labels)
    estimator = create_logistic_regression_pipeline(
        class_weight="balanced", max_iter=1
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _, summary = run_experiment(
            estimator, X, y,
            experiment_id="TECH-LR-BALANCED-CONVERGENCE",
            model_name="Technical balanced convergence check",
            member="test", branch="test",
            cv=create_stratified_cv(n_splits=2), n_jobs=1,
        )
    assert summary["convergence_warning_detected"] is True
    assert any(issubclass(item.category, ConvergenceWarning) for item in caught)

"""Offline tests for the common cross-validation evaluation framework."""

import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

from src.evaluation import (
    append_experiment_registry,
    evaluate_model_cv,
    get_primary_metric_name,
    get_scoring_metrics,
    save_experiment_results,
    validate_evaluation_inputs,
)
from src.validation import create_stratified_cv, get_cv_split_fingerprints


def _synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    values, target = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        weights=[0.8, 0.2],
        random_state=42,
    )
    index = pd.Index(range(500, 620), name="training_row")
    return (
        pd.DataFrame(values, index=index, columns=[f"x_{i}" for i in range(5)]),
        pd.Series(target, index=index, name="target"),
    )


def _evaluation() -> tuple[pd.DataFrame, dict[str, object]]:
    X, y = _synthetic_data()
    return evaluate_model_cv(
        DummyClassifier(strategy="prior"),
        X,
        y,
        model_name="Synthetic smoke estimator",
        experiment_id="TECH-SMOKE-001",
        member="test",
        branch="test",
        cv=create_stratified_cv(n_splits=3),
        n_jobs=1,
    )


def test_primary_metric_is_roc_auc() -> None:
    assert get_primary_metric_name() == "roc_auc"


def test_scoring_metrics_are_complete() -> None:
    expected = {
        "roc_auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "balanced_accuracy",
    }
    assert expected == get_scoring_metrics().keys()


def test_precision_and_recall_handle_no_positive_predictions() -> None:
    X, y = _synthetic_data()
    estimator = DummyClassifier(strategy="most_frequent").fit(X, y)
    scoring = get_scoring_metrics()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert scoring["precision"](estimator, X, y) == 0.0
        assert scoring["recall"](estimator, X, y) == 0.0


def test_stratified_cv_and_fingerprints_are_reproducible() -> None:
    X, y = _synthetic_data()
    first = get_cv_split_fingerprints(create_stratified_cv(), X, y)
    second = get_cv_split_fingerprints(create_stratified_cv(), X, y)
    assert first == second
    assert len(first) == 5
    assert all(item["train_size"] == 96 for item in first)
    assert all(item["validation_size"] == 24 for item in first)


def test_cv_fingerprints_change_with_random_state() -> None:
    X, y = _synthetic_data()
    first = get_cv_split_fingerprints(create_stratified_cv(random_state=1), X, y)
    second = get_cv_split_fingerprints(create_stratified_cv(random_state=2), X, y)
    assert first != second


def test_evaluate_model_cv_returns_fold_results_and_summary() -> None:
    folds, summary = _evaluation()
    expected_columns = {
        "experiment_id",
        "model_name",
        "fold",
        "fit_time",
        "score_time",
        "train_size",
        "validation_size",
        "train_roc_auc",
        "validation_roc_auc",
        "validation_average_precision",
        "validation_f1",
        "validation_precision",
        "validation_recall",
        "validation_accuracy",
        "validation_balanced_accuracy",
    }
    assert len(folds) == 3
    assert expected_columns <= set(folds.columns)
    expected_summary = {
        "experiment_id",
        "model_name",
        "member",
        "branch",
        "date_utc",
        "n_samples",
        "n_features",
        "cv_strategy",
        "n_splits",
        "shuffle",
        "random_state",
        "primary_metric",
        "primary_score_mean",
        "primary_score_std",
        "metrics",
        "fit_time_mean",
        "fit_time_std",
        "score_time_mean",
        "score_time_std",
        "estimator_class",
        "estimator_parameters",
        "target_distribution",
        "cv_fingerprints",
        "status",
    }
    assert expected_summary <= summary.keys()
    assert summary["status"] == "completed"
    assert summary["primary_score_mean"] == pytest.approx(
        folds["validation_roc_auc"].mean()
    )
    assert summary["primary_score_std"] == pytest.approx(
        folds["validation_roc_auc"].std(ddof=0)
    )
    json.dumps(summary, allow_nan=False)


def test_evaluate_model_cv_supports_santander_string_labels() -> None:
    X, y = _synthetic_data()
    y = y.map({0: "False", 1: "True"})
    folds, summary = evaluate_model_cv(
        DummyClassifier(strategy="prior"),
        X,
        y,
        model_name="String-label smoke estimator",
        experiment_id="TECH-STRING-001",
        cv=create_stratified_cv(n_splits=3),
        n_jobs=1,
    )
    assert folds["validation_roc_auc"].eq(0.5).all()
    assert summary["target_distribution"]["True"] > 0


def test_evaluation_api_has_no_final_test_argument() -> None:
    parameters = inspect.signature(evaluate_model_cv).parameters
    assert "X_test" not in parameters
    assert "y_test" not in parameters


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("empty", "must not be empty"),
        ("misaligned", "indexes must be identical"),
        ("single_class", "at least two classes"),
        ("empty_model", "model_name"),
        ("empty_experiment", "experiment_id"),
    ],
)
def test_validate_evaluation_inputs_rejects_invalid_input(
    change: str, message: str
) -> None:
    X, y = _synthetic_data()
    model_name = "model"
    experiment_id = "M01-TEST-001"
    if change == "empty":
        X, y = X.iloc[:0], y.iloc[:0]
    elif change == "misaligned":
        y.index = range(len(y))
    elif change == "single_class":
        y[:] = 0
    elif change == "empty_model":
        model_name = " "
    elif change == "empty_experiment":
        experiment_id = ""
    with pytest.raises(ValueError, match=message):
        validate_evaluation_inputs(
            X, y, model_name=model_name, experiment_id=experiment_id
        )


def test_save_experiment_results_creates_files_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    folds, summary = _evaluation()
    fold_path, summary_path = save_experiment_results(
        folds, summary, output_dir=tmp_path
    )
    assert fold_path.is_file()
    assert summary_path.is_file()
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["experiment_id"] == "TECH-SMOKE-001"
    assert str(tmp_path) not in summary_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in fold_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_experiment_results(folds, summary, output_dir=tmp_path)


def test_experiment_registry_is_unique_and_uses_relative_summary_file(
    tmp_path: Path,
) -> None:
    _, summary = _evaluation()
    registry_path = tmp_path / "experiment_registry.csv"
    result = append_experiment_registry(summary, registry_path=registry_path)
    registry = pd.read_csv(result)
    assert len(registry) == 1
    assert registry.loc[0, "experiment_id"] == "TECH-SMOKE-001"
    assert registry.loc[0, "summary_file"] == "TECH-SMOKE-001_summary.json"
    assert str(tmp_path) not in result.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        append_experiment_registry(summary, registry_path=registry_path)

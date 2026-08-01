"""Offline tests for scientific Dummy baseline definitions and reports."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

import src.dummy_baselines as baselines
from src.dummy_baselines import (
    COMPARISON_COLUMNS,
    DUMMY_EXPERIMENTS,
    build_comparison_table,
    build_dummy_classifiers,
    save_comparison_table,
)
from src.experiments import run_and_save_experiment, run_experiment
from src.validation import create_stratified_cv


def _data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"feature": range(100), "other": [i % 3 for i in range(100)]})
    y = pd.Series([0] * 90 + [1] * 10, index=X.index)
    return X, y


def _summary(experiment_id: str, strategy: str) -> dict[str, object]:
    X, y = _data()
    _, summary = run_experiment(
        build_dummy_classifiers()[strategy], X, y,
        experiment_id=experiment_id,
        model_name=f"DummyClassifier ({strategy})",
        member="Member 01", branch="feature/data_processing",
        cv=create_stratified_cv(n_splits=2), n_jobs=1,
    )
    return summary


def test_builds_exact_four_unique_strategies() -> None:
    classifiers = build_dummy_classifiers()
    assert set(classifiers) == {"most_frequent", "prior", "stratified", "uniform"}
    assert len({item["experiment_id"] for item in DUMMY_EXPERIMENTS}) == 4
    assert all(isinstance(model, DummyClassifier) for model in classifiers.values())


def test_random_strategies_use_shared_seed() -> None:
    classifiers = build_dummy_classifiers(random_state=42)
    assert classifiers["stratified"].random_state == 42
    assert classifiers["uniform"].random_state == 42
    assert classifiers["most_frequent"].random_state is None


def test_comparison_has_expected_columns_and_serializable_values() -> None:
    table = build_comparison_table([_summary("M01-DUMMY-001", "most_frequent")])
    assert tuple(table.columns) == COMPARISON_COLUMNS
    json.dumps(table.to_dict(orient="records"), allow_nan=False)
    assert not any(Path(value).is_absolute() for value in table.select_dtypes(include=["object", "string"]).iloc[0])


def test_majority_baseline_exposes_accuracy_imbalance() -> None:
    X, y = _data()
    model = DummyClassifier(strategy="most_frequent").fit(X, y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    assert model.score(X, y) == pytest.approx(0.9)
    assert balanced_accuracy_score(y, predictions) == pytest.approx(0.5)
    assert roc_auc_score(y, probabilities) == pytest.approx(0.5)
    assert average_precision_score(y, probabilities) == pytest.approx(y.mean())


def test_orchestration_functions_have_no_test_partition_parameters() -> None:
    for function in (run_experiment, run_and_save_experiment):
        parameters = inspect.signature(function).parameters
        assert "X_test" not in parameters and "y_test" not in parameters


def test_save_comparison_writes_csv_json_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baselines, "get_project_root", lambda: tmp_path)
    table = build_comparison_table([_summary("M01-DUMMY-001", "most_frequent")])
    csv_path, json_path = save_comparison_table(
        table, csv_path="comparison.csv", json_path="comparison.json"
    )
    assert pd.read_csv(csv_path).loc[0, "experiment_id"] == "M01-DUMMY-001"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["strategy"] == "most_frequent"
    assert str(tmp_path) not in json_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        save_comparison_table(
            table, csv_path="comparison.csv", json_path="comparison.json"
        )

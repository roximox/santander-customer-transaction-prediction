"""Offline tests for dataset loading, validation, and minimal auditing."""

import json

from types import SimpleNamespace

import pandas as pd
import pytest

import src.data as data_module
from src.data import get_dataset_summary, load_dataset, memory_usage_mb, validate_dataset


def _openml_result() -> SimpleNamespace:
    X = pd.DataFrame(
        {"numeric": [1.0, 2.0, None, 1.0], "category": ["a", "b", "b", "a"]}
    )
    y = pd.Series([0, 1, 1, 0], name="official_target")
    return SimpleNamespace(
        data=X,
        target=y,
        feature_names=list(X.columns),
        target_names=["official_target"],
        details={"name": "mock-santander", "version": 3},
    )


def test_load_dataset_returns_data_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_fetch_openml(**kwargs: object) -> SimpleNamespace:
        calls.update(kwargs)
        return _openml_result()

    monkeypatch.setattr(data_module, "fetch_openml", fake_fetch_openml)
    X, y, metadata = load_dataset()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert calls == {
        "data_id": 45566,
        "as_frame": True,
        "cache": True,
        "parser": "auto",
    }
    assert metadata == {
        "openml_id": 45566,
        "project_dataset_name": "Santander Customer Transaction Prediction",
        "openml_dataset_name": "mock-santander",
        "dataset_name": "mock-santander",
        "target_name": "official_target",
        "n_rows": 4,
        "n_features": 2,
        "feature_names": ["numeric", "category"],
        "data_version": 3,
        "source": "OpenML",
    }


def test_openml_id_is_read_from_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(data_module, "load_config", lambda: {"data": {"openml_id": 123}})

    def fake_fetch_openml(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return _openml_result()

    monkeypatch.setattr(data_module, "fetch_openml", fake_fetch_openml)
    load_dataset()
    assert observed["data_id"] == 123


def test_validate_dataset_accepts_valid_data() -> None:
    result = _openml_result()
    validate_dataset(result.data, result.target)


def test_validate_dataset_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        validate_dataset(pd.DataFrame({"a": [1, 2]}), pd.Series([1], name="target"))


def test_validate_dataset_rejects_duplicate_columns() -> None:
    X = pd.DataFrame([[1, 2]], columns=["duplicate", "duplicate"])
    with pytest.raises(ValueError, match="column names must be unique"):
        validate_dataset(X, pd.Series([1], name="target"))


def test_memory_usage_mb_is_positive() -> None:
    assert memory_usage_mb(pd.DataFrame({"text": ["a", "b"]})) > 0


def test_get_dataset_summary() -> None:
    X = pd.DataFrame(
        {
            "numeric": [1.0, None, 2.0, 1.0],
            "category": ["a", "b", "a", "a"],
        }
    )
    y = pd.Series([0, 1, 0, 0], name="target")

    summary = get_dataset_summary(X, y)

    assert summary["n_rows"] == 4
    assert summary["n_features"] == 2
    assert summary["missing_values_X"] == 1
    assert summary["missing_values_y"] == 0
    assert summary["duplicate_rows_X"] == 1
    assert summary["numeric_feature_count"] == 1
    assert summary["categorical_feature_count"] == 1
    assert summary["target_value_counts"] == {0: 3, 1: 1}
    assert summary["target_proportions"] == {0: 0.75, 1: 0.25}
    assert summary["memory_X_mb"] > 0
    assert summary["memory_y_mb"] > 0
    assert summary["total_memory_mb"] == pytest.approx(
        summary["memory_X_mb"] + summary["memory_y_mb"]
    )
    json.dumps(summary)

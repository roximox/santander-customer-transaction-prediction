"""Offline tests for the shared reproducible train/test split."""

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    create_train_test_split,
    split_fingerprint,
    validate_train_test_split,
)


def _classification_data() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.Index(range(1000, 1100), name="customer_row")
    X = pd.DataFrame(
        {
            "feature_a": np.arange(100, dtype=np.float32),
            "feature_b": np.linspace(-1, 1, 100, dtype=np.float32),
        },
        index=index,
    )
    y = pd.Series([0] * 80 + [1] * 20, index=index, name="target")
    return X, y


def test_create_train_test_split_uses_shared_defaults() -> None:
    X, y = _classification_data()
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    assert len(X_train) == len(y_train) == 80
    assert len(X_test) == len(y_test) == 20
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert X_train.index.intersection(X_test.index).empty
    assert set(X_train.index) | set(X_test.index) == set(X.index)
    assert X_train.columns.equals(X.columns)
    assert X_test.columns.equals(X.columns)
    assert X_train.dtypes.equals(X.dtypes)
    assert X_test.dtypes.equals(X.dtypes)
    assert y_train.value_counts(normalize=True).to_dict() == {0: 0.8, 1: 0.2}
    assert y_test.value_counts(normalize=True).to_dict() == {0: 0.8, 1: 0.2}


def test_split_is_reproducible_for_same_random_state() -> None:
    X, y = _classification_data()
    first = create_train_test_split(X, y, random_state=7)
    second = create_train_test_split(X, y, random_state=7)
    assert first[0].index.equals(second[0].index)
    assert first[1].index.equals(second[1].index)


def test_raw_and_float32_features_produce_identical_indices() -> None:
    X_float32, y = _classification_data()
    X_float64 = X_float32.astype("float64")
    raw_split = create_train_test_split(X_float64, y)
    optimized_split = create_train_test_split(X_float32, y)
    assert raw_split[0].index.equals(optimized_split[0].index)
    assert raw_split[1].index.equals(optimized_split[1].index)


def test_different_random_state_changes_split() -> None:
    X, y = _classification_data()
    first = create_train_test_split(X, y, random_state=7)
    second = create_train_test_split(X, y, random_state=8)
    assert not first[0].index.equals(second[0].index)
    assert split_fingerprint(first[0].index) != split_fingerprint(second[0].index)


def test_split_preserves_non_default_index() -> None:
    X, y = _classification_data()
    partitions = create_train_test_split(X, y)
    assert all(partition.index.name == "customer_row" for partition in partitions)
    assert all(partition.index.isin(X.index).all() for partition in partitions)


def test_create_train_test_split_rejects_different_lengths() -> None:
    X, y = _classification_data()
    with pytest.raises(ValueError, match="same number of rows"):
        create_train_test_split(X.iloc[:-1], y)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_create_train_test_split_rejects_invalid_test_size(test_size: float) -> None:
    X, y = _classification_data()
    with pytest.raises(ValueError, match="0 < test_size < 1"):
        create_train_test_split(X, y, test_size=test_size)


def test_create_train_test_split_rejects_single_class() -> None:
    X, y = _classification_data()
    y[:] = 0
    with pytest.raises(ValueError, match="at least two classes"):
        create_train_test_split(X, y)


def test_split_fingerprint_is_stable_and_order_sensitive() -> None:
    indices = pd.Index([7, 2, 11], name="row")
    assert split_fingerprint(indices) == split_fingerprint(indices.copy())
    assert split_fingerprint(indices) != split_fingerprint(pd.Index([2, 7, 11]))


def test_validate_train_test_split_returns_complete_summary() -> None:
    X, y = _classification_data()
    partitions = create_train_test_split(X, y)
    summary = validate_train_test_split(X, y, *partitions)
    expected_fields = {
        "n_train",
        "n_test",
        "train_percentage",
        "test_percentage",
        "original_target_distribution",
        "train_target_distribution",
        "test_target_distribution",
        "maximum_target_proportion_difference",
        "overlap_count",
        "split_random_state",
        "split_test_size",
        "stratified",
        "train_indices_sha256",
        "test_indices_sha256",
    }
    assert expected_fields <= summary.keys()
    assert summary["n_train"] == 80
    assert summary["n_test"] == 20
    assert summary["overlap_count"] == 0
    assert summary["maximum_target_proportion_difference"] == pytest.approx(0.0)
    assert summary["split_random_state"] == 42
    assert summary["split_test_size"] == pytest.approx(0.2)
    assert summary["stratified"] is True
    assert len(summary["train_indices_sha256"]) == 64
    assert len(summary["test_indices_sha256"]) == 64


def test_validate_train_test_split_rejects_modified_features() -> None:
    X, y = _classification_data()
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    X_train.iloc[0, 0] = -999.0
    with pytest.raises(ValueError, match="Feature values were modified"):
        validate_train_test_split(X, y, X_train, X_test, y_train, y_test)

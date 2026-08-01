"""Reproducible train/test splitting and validation utilities."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import load_config

_SPLIT_METADATA_KEY = "train_test_split"


def _configured_split_defaults() -> tuple[float, int]:
    """Read and validate shared split defaults from the project configuration."""
    config = load_config()
    data_config = config.get("data")
    project_config = config.get("project")
    if not isinstance(data_config, Mapping):
        raise ValueError("Configuration must contain a 'data' mapping.")
    if not isinstance(project_config, Mapping):
        raise ValueError("Configuration must contain a 'project' mapping.")
    return _validate_test_size(data_config.get("test_size")), _validate_random_state(
        project_config.get("random_state")
    )


def _validate_test_size(value: Any) -> float:
    """Return a valid fractional test size."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"test_size must be numeric and between 0 and 1; got {value!r}.")
    test_size = float(value)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must satisfy 0 < test_size < 1; got {value!r}.")
    return test_size


def _validate_random_state(value: Any) -> int:
    """Return a valid integer random state."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"random_state must be an integer; got {value!r}.")
    return int(value)


def _validate_split_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate invariants required for an unambiguous index-based split."""
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of rows; got {len(X)} and {len(y)}."
        )
    if X.empty or y.empty:
        raise ValueError("X and y must not be empty.")
    if not X.index.equals(y.index):
        raise ValueError("X and y indexes must be identical and in the same order.")
    if not X.index.is_unique:
        raise ValueError("X and y indexes must be unique for split verification.")
    if y.nunique(dropna=True) < 2:
        raise ValueError("Target y must contain at least two classes.")


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float | None = None,
    random_state: int | None = None,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the shared shuffled train/test split while preserving source indexes.

    Configuration supplies ``data.test_size`` and ``project.random_state`` when
    arguments are omitted. No transformation is learned or applied here. Since
    scikit-learn shuffles row positions independently of feature values, the
    resulting indexes are identical for raw and dtype-optimized feature frames.
    """
    _validate_split_inputs(X, y)
    configured_test_size, configured_random_state = _configured_split_defaults()
    resolved_test_size = _validate_test_size(
        configured_test_size if test_size is None else test_size
    )
    resolved_random_state = _validate_random_state(
        configured_random_state if random_state is None else random_state
    )
    if not isinstance(stratify, bool):
        raise ValueError(f"stratify must be a boolean; got {stratify!r}.")
    if stratify and int(y.value_counts(dropna=False).min()) < 2:
        raise ValueError(
            "Every target class must contain at least two rows for stratification."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=resolved_test_size,
        random_state=resolved_random_state,
        shuffle=True,
        stratify=y if stratify else None,
    )
    split_metadata = {
        "test_size": resolved_test_size,
        "random_state": resolved_random_state,
        "stratified": stratify,
        "shuffle": True,
    }
    for partition in (X_train, X_test, y_train, y_test):
        partition.attrs[_SPLIT_METADATA_KEY] = split_metadata.copy()
    return X_train, X_test, y_train, y_test


def _plain_scalar(value: Any) -> Any:
    """Convert a NumPy scalar to its Python equivalent."""
    return value.item() if isinstance(value, np.generic) else value


def _target_distribution(y: pd.Series) -> dict[Any, float]:
    """Return target proportions as plain serializable values."""
    return {
        _plain_scalar(key): float(value)
        for key, value in y.value_counts(dropna=False, normalize=True).items()
    }


def split_fingerprint(indices: pd.Index) -> str:
    """Return an order-sensitive deterministic SHA-256 fingerprint of index values.

    Each value is encoded with its fully qualified type name and representation,
    separated using length prefixes. No filesystem or machine-specific context is
    included.
    """
    digest = sha256()
    for value in indices:
        normalized = _plain_scalar(value)
        type_name = f"{type(normalized).__module__}.{type(normalized).__qualname__}"
        representation = repr(normalized)
        for component in (type_name, representation):
            encoded = component.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
            digest.update(encoded)
    return digest.hexdigest()


def create_stratified_cv(
    *,
    n_splits: int | None = None,
    shuffle: bool | None = None,
    random_state: int | None = None,
) -> StratifiedKFold:
    """Create the common stratified cross-validator from shared configuration.

    When shuffling is disabled, ``random_state`` is set to ``None`` because
    scikit-learn rejects or ignores a random seed without shuffling.
    """
    config = load_config()
    validation_config = config.get("validation")
    project_config = config.get("project")
    if not isinstance(validation_config, Mapping):
        raise ValueError("Configuration must contain a 'validation' mapping.")
    if not isinstance(project_config, Mapping):
        raise ValueError("Configuration must contain a 'project' mapping.")

    resolved_n_splits = validation_config.get("n_splits") if n_splits is None else n_splits
    if (
        isinstance(resolved_n_splits, bool)
        or not isinstance(resolved_n_splits, Integral)
        or resolved_n_splits < 2
    ):
        raise ValueError(f"n_splits must be an integer >= 2; got {resolved_n_splits!r}.")
    resolved_shuffle = validation_config.get("shuffle") if shuffle is None else shuffle
    if not isinstance(resolved_shuffle, bool):
        raise ValueError(f"shuffle must be a boolean; got {resolved_shuffle!r}.")
    configured_random_state = project_config.get("random_state")
    resolved_random_state = _validate_random_state(
        configured_random_state if random_state is None else random_state
    )
    if not resolved_shuffle:
        resolved_random_state = None

    return StratifiedKFold(
        n_splits=int(resolved_n_splits),
        shuffle=resolved_shuffle,
        random_state=resolved_random_state,
    )


def get_cv_split_fingerprints(
    cv: StratifiedKFold,
    X: pd.DataFrame,
    y: pd.Series,
) -> list[dict[str, Any]]:
    """Describe and fingerprint every ordered train/validation fold."""
    _validate_split_inputs(X, y)
    fingerprints: list[dict[str, Any]] = []
    for fold, (train_positions, validation_positions) in enumerate(
        cv.split(X, y), start=1
    ):
        train_indices = X.index.take(train_positions)
        validation_indices = X.index.take(validation_positions)
        fingerprints.append(
            {
                "fold": fold,
                "train_size": int(len(train_positions)),
                "validation_size": int(len(validation_positions)),
                "train_indices_sha256": split_fingerprint(train_indices),
                "validation_indices_sha256": split_fingerprint(validation_indices),
                "train_target_distribution": _target_distribution(
                    y.iloc[train_positions]
                ),
                "validation_target_distribution": _target_distribution(
                    y.iloc[validation_positions]
                ),
            }
        )
    return fingerprints


def _split_metadata(*partitions: pd.DataFrame | pd.Series) -> dict[str, Any]:
    """Return consistent split metadata attached by the creation function."""
    metadata_values = [partition.attrs.get(_SPLIT_METADATA_KEY) for partition in partitions]
    if not all(isinstance(value, Mapping) for value in metadata_values):
        raise ValueError(
            "Split partitions lack reproducibility metadata; create them with "
            "create_train_test_split."
        )
    first = dict(metadata_values[0])
    if any(dict(value) != first for value in metadata_values[1:]):
        raise ValueError("Split partitions contain inconsistent reproducibility metadata.")
    return first


def validate_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Validate split integrity, stratification, and deterministic reproducibility."""
    _validate_split_inputs(X, y)
    metadata = _split_metadata(X_train, X_test, y_train, y_test)
    n_train = len(X_train)
    n_test = len(X_test)
    if n_train != len(y_train) or n_test != len(y_test):
        raise ValueError("Feature and target partition lengths are inconsistent.")
    if n_train + n_test != len(X):
        raise ValueError("Train and test sizes do not reconstruct the original dataset.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train indexes are not aligned.")
    if not X_test.index.equals(y_test.index):
        raise ValueError("X_test and y_test indexes are not aligned.")

    overlap = X_train.index.intersection(X_test.index)
    if len(overlap):
        raise ValueError(f"Train and test indexes overlap in {len(overlap)} positions.")
    combined_indices = X_train.index.append(X_test.index)
    if (
        len(combined_indices) != len(X.index)
        or not X.index.difference(combined_indices).empty
        or not combined_indices.difference(X.index).empty
    ):
        raise ValueError("Train/test index union does not equal the original index.")
    if not X_train.columns.equals(X.columns) or not X_test.columns.equals(X.columns):
        raise ValueError("Train and test columns must match the original columns.")
    if not X_train.dtypes.equals(X.dtypes) or not X_test.dtypes.equals(X.dtypes):
        raise ValueError("Train and test dtypes must match the original dtypes.")
    if not X_train.equals(X.loc[X_train.index]) or not X_test.equals(X.loc[X_test.index]):
        raise ValueError("Feature values were modified while creating the split.")
    if not y_train.equals(y.loc[y_train.index]) or not y_test.equals(y.loc[y_test.index]):
        raise ValueError("Target values were modified while creating the split.")

    original_distribution = _target_distribution(y)
    train_distribution = _target_distribution(y_train)
    test_distribution = _target_distribution(y_test)
    classes = set(original_distribution) | set(train_distribution) | set(test_distribution)
    maximum_difference = max(
        max(
            abs(train_distribution.get(label, 0.0) - original_distribution.get(label, 0.0)),
            abs(test_distribution.get(label, 0.0) - original_distribution.get(label, 0.0)),
        )
        for label in classes
    )
    if bool(metadata["stratified"]):
        rounding_tolerance = max(1.0 / n_train, 1.0 / n_test)
        if maximum_difference > rounding_tolerance:
            raise ValueError(
                "Stratified target proportions differ from the original beyond "
                f"rounding tolerance ({maximum_difference} > {rounding_tolerance})."
            )

    reproduced = create_train_test_split(
        X,
        y,
        test_size=float(metadata["test_size"]),
        random_state=int(metadata["random_state"]),
        stratify=bool(metadata["stratified"]),
    )
    if not (
        reproduced[0].index.equals(X_train.index)
        and reproduced[1].index.equals(X_test.index)
        and reproduced[2].index.equals(y_train.index)
        and reproduced[3].index.equals(y_test.index)
    ):
        raise ValueError("Split is not reproducible with its recorded parameters.")

    return {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_percentage": float(n_train / len(X) * 100.0),
        "test_percentage": float(n_test / len(X) * 100.0),
        "original_target_distribution": original_distribution,
        "train_target_distribution": train_distribution,
        "test_target_distribution": test_distribution,
        "maximum_target_proportion_difference": float(maximum_difference),
        "overlap_count": int(len(overlap)),
        "split_random_state": int(metadata["random_state"]),
        "split_test_size": float(metadata["test_size"]),
        "stratified": bool(metadata["stratified"]),
        "train_indices_sha256": split_fingerprint(X_train.index),
        "test_indices_sha256": split_fingerprint(X_test.index),
    }

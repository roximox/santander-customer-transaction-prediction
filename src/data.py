"""Loading and minimal auditing utilities for the configured OpenML dataset."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import Parameter, signature
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from src.config import load_config

PROJECT_DATASET_NAME = "Santander Customer Transaction Prediction"
ALLOWED_NUMERIC_DTYPES = frozenset({"float32", "float64"})
DEFAULT_QUASI_CONSTANT_THRESHOLD = 0.99


def _configured_openml_id(data_config: Mapping[str, Any]) -> int:
    """Return a validated OpenML dataset identifier from project configuration."""
    openml_id = data_config.get("openml_id")
    if (
        not isinstance(openml_id, Integral)
        or isinstance(openml_id, bool)
        or openml_id <= 0
    ):
        raise ValueError(
            "Configuration value 'data.openml_id' must be a positive integer; "
            f"received {openml_id!r}."
        )
    return int(openml_id)


def _fetch_openml_dataset(openml_id: int, *, as_frame: bool, cache: bool) -> Any:
    """Fetch one OpenML dataset while supporting older scikit-learn releases."""
    fetch_parameters = signature(fetch_openml).parameters.values()
    supports_parser = any(
        parameter.name == "parser" or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in fetch_parameters
    )
    kwargs: dict[str, Any] = {
        "data_id": openml_id,
        "as_frame": as_frame,
        "cache": cache,
    }
    if supports_parser:
        kwargs["parser"] = "auto"

    try:
        return fetch_openml(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to download OpenML dataset ID {openml_id}: {exc}"
        ) from exc


def _as_dataframe(data: Any, feature_names: Any) -> pd.DataFrame:
    """Represent fetched features as a DataFrame without changing values."""
    if isinstance(data, pd.DataFrame):
        return data
    columns = list(feature_names) if feature_names is not None else None
    return pd.DataFrame(data, columns=columns)


def _as_series(target: Any, target_names: Any) -> pd.Series:
    """Represent a single fetched target as a Series without changing values."""
    if isinstance(target, pd.Series):
        return target
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("OpenML returned multiple target columns; one was expected.")
        return target.iloc[:, 0]

    target_name: str | None = None
    if isinstance(target_names, str):
        target_name = target_names
    elif target_names is not None:
        names = list(target_names)
        if len(names) == 1:
            target_name = str(names[0])
    return pd.Series(target, name=target_name)


def load_dataset(
    *,
    as_frame: bool = True,
    cache: bool = True,
    optimize_memory: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load, validate, and describe the configured dataset from OpenML.

    The standard scikit-learn cache is used by default. Feature values, target
    values, and numeric dtypes are not transformed. When ``as_frame`` is false,
    scikit-learn may return arrays; they are wrapped in pandas containers solely
    to provide the project's stable DataFrame/Series API.

    Args:
        as_frame: Ask OpenML to return pandas objects when true.
        cache: Reuse scikit-learn's standard local OpenML cache when true.
        optimize_memory: Explicitly convert numeric features to the configured
            recommended dtype after validating the raw dataset. False preserves
            the raw OpenML dtypes.

    Returns:
        The feature frame, target series, and source metadata.

    Raises:
        FileNotFoundError: If the project configuration is missing.
        ValueError: If configuration or returned dataset content is invalid.
        RuntimeError: If OpenML retrieval fails.
    """
    config = load_config()
    data_config = config.get("data")
    if not isinstance(data_config, Mapping):
        raise ValueError("Configuration must contain a 'data' mapping.")
    openml_id = _configured_openml_id(data_config)
    dataset = _fetch_openml_dataset(openml_id, as_frame=as_frame, cache=cache)

    data = getattr(dataset, "data", None)
    target = getattr(dataset, "target", None)
    if data is None:
        raise ValueError(f"OpenML dataset ID {openml_id} did not return feature data.")
    if target is None:
        raise ValueError(f"OpenML dataset ID {openml_id} did not return a target.")
    if as_frame and not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"OpenML dataset ID {openml_id} did not return a DataFrame "
            "although as_frame=True."
        )

    X = _as_dataframe(data, getattr(dataset, "feature_names", None))
    y = _as_series(target, getattr(dataset, "target_names", None))
    validate_dataset(X, y)

    details = getattr(dataset, "details", None)
    details = details if isinstance(details, Mapping) else {}
    target_name = y.name
    if target_name is None:
        default_target = details.get("default_target_attribute")
        if default_target:
            target_name = str(default_target)
            y = y.rename(target_name)
    if target_name is None:
        raise ValueError(
            f"OpenML dataset ID {openml_id} returned a target without a name."
        )

    openml_dataset_name = str(
        details.get("name") or f"OpenML dataset {openml_id}"
    )
    metadata: dict[str, Any] = {
        "openml_id": openml_id,
        "project_dataset_name": PROJECT_DATASET_NAME,
        "openml_dataset_name": openml_dataset_name,
        # Backward-compatible alias; new code should use openml_dataset_name.
        "dataset_name": openml_dataset_name,
        "target_name": str(target_name),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": [str(name) for name in X.columns],
        "data_version": _python_scalar(details.get("version")),
        "source": "OpenML",
    }
    if optimize_memory:
        configured_dtype = data_config.get("numeric_dtype")
        if not isinstance(configured_dtype, str):
            raise ValueError(
                "Configuration value 'data.numeric_dtype' must be a supported "
                f"dtype string; received {configured_dtype!r}."
            )
        X_converted = convert_numeric_dtype(X, dtype=configured_dtype, copy=True)
        validate_dtype_conversion(X, X_converted, target_dtype=configured_dtype)
        metadata["dtype_conversion"] = compare_numeric_precision(X, X_converted)
        metadata["numeric_dtype"] = configured_dtype
        X = X_converted
    return X, y, metadata


def validate_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate structural invariants without cleaning or rejecting missing features."""
    if X.empty:
        raise ValueError("Feature dataset X must not be empty.")
    if y.empty:
        raise ValueError("Target y must not be empty.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of rows; got {len(X)} and {len(y)}."
        )
    duplicated_columns = X.columns[X.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(
            "Feature column names must be unique; duplicates: "
            f"{duplicated_columns}."
        )
    if y.isna().all():
        raise ValueError("Target y must not be entirely missing.")
    if not X.index.equals(y.index):
        raise ValueError("X and y indexes must be identical and in the same order.")


def memory_usage_mb(obj: pd.DataFrame | pd.Series) -> float:
    """Return deep object memory usage in MiB (bytes divided by 1024 squared)."""
    usage = obj.memory_usage(index=True, deep=True)
    bytes_used = int(usage.sum()) if isinstance(usage, pd.Series) else int(usage)
    return bytes_used / (1024**2)


def _validate_quasi_constant_threshold(threshold: float) -> None:
    """Validate a dominant-value threshold expressed as a fraction."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("quasi_constant_threshold must be in the interval (0, 1].")


def audit_numeric_features(
    X: pd.DataFrame,
    *,
    quasi_constant_threshold: float = DEFAULT_QUASI_CONSTANT_THRESHOLD,
) -> pd.DataFrame:
    """Audit every feature without changing its values.

    A feature is considered quasi-constant when one non-missing value represents
    at least ``quasi_constant_threshold`` of its non-missing observations, while
    the feature still has more than one distinct non-missing value. The default
    threshold is 99%. Infinities are counted but excluded from distribution
    statistics, for which they are not finite observations. Numeric distribution
    fields are missing for non-numeric features, which remain represented in the
    returned table.
    """
    _validate_quasi_constant_threshold(quasi_constant_threshold)
    rows: list[dict[str, Any]] = []
    n_rows = len(X)

    for feature in X.columns:
        series = X[feature]
        non_missing = series.dropna()
        unique_count = int(series.nunique(dropna=True))
        dominant_fraction = (
            float(non_missing.value_counts(normalize=True, dropna=False).iloc[0])
            if not non_missing.empty
            else 0.0
        )
        is_constant = unique_count <= 1
        row: dict[str, Any] = {
            "feature": str(feature),
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": (
                float(series.isna().mean() * 100.0) if n_rows else 0.0
            ),
            "unique_count": unique_count,
            "is_constant": bool(is_constant),
            "is_quasi_constant": bool(
                not is_constant and dominant_fraction >= quasi_constant_threshold
            ),
            "dominant_value_percentage": float(dominant_fraction * 100.0),
            "memory_bytes": int(series.memory_usage(index=False, deep=True)),
        }

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric_values = series.to_numpy()
            positive_infinity = int(np.isposinf(numeric_values).sum())
            negative_infinity = int(np.isneginf(numeric_values).sum())
            finite_series = series.mask(np.isinf(numeric_values))
            quantiles = finite_series.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            row.update(
                {
                    "mean": float(finite_series.mean()),
                    "std": float(finite_series.std()),
                    "min": float(finite_series.min()),
                    "q01": float(quantiles.loc[0.01]),
                    "q25": float(quantiles.loc[0.25]),
                    "median": float(quantiles.loc[0.5]),
                    "q75": float(quantiles.loc[0.75]),
                    "q99": float(quantiles.loc[0.99]),
                    "max": float(finite_series.max()),
                    "zero_count": int(series.eq(0).sum()),
                    "zero_percentage": float(series.eq(0).mean() * 100.0),
                    "positive_infinity_count": positive_infinity,
                    "negative_infinity_count": negative_infinity,
                }
            )
        else:
            row.update(
                {
                    key: None
                    for key in (
                        "mean",
                        "std",
                        "min",
                        "q01",
                        "q25",
                        "median",
                        "q75",
                        "q99",
                        "max",
                        "zero_count",
                        "zero_percentage",
                        "positive_infinity_count",
                        "negative_infinity_count",
                    )
                }
            )
        rows.append(row)

    columns = [
        "feature",
        "dtype",
        "count",
        "missing_count",
        "missing_percentage",
        "unique_count",
        "mean",
        "std",
        "min",
        "q01",
        "q25",
        "median",
        "q75",
        "q99",
        "max",
        "zero_count",
        "zero_percentage",
        "positive_infinity_count",
        "negative_infinity_count",
        "is_constant",
        "is_quasi_constant",
        "dominant_value_percentage",
        "memory_bytes",
    ]
    return pd.DataFrame(rows, columns=columns)


def convert_numeric_dtype(
    X: pd.DataFrame,
    *,
    dtype: str = "float32",
    copy: bool = True,
) -> pd.DataFrame:
    """Convert numeric features to an explicitly allowed floating dtype.

    Non-numeric columns, the index, column labels, and the target (which is not
    accepted by this function) remain untouched. Float16 is intentionally not
    supported because its precision and range are unsuitable for this audit.
    """
    try:
        normalized_dtype = np.dtype(dtype).name
    except TypeError as exc:
        raise ValueError(f"Unsupported numeric dtype: {dtype!r}.") from exc
    if normalized_dtype not in ALLOWED_NUMERIC_DTYPES:
        raise ValueError(
            f"Unsupported numeric dtype {dtype!r}; allowed values are "
            f"{sorted(ALLOWED_NUMERIC_DTYPES)}."
        )

    converted = X.copy(deep=True) if copy else X
    numeric_columns = converted.select_dtypes(include="number").columns
    converted[numeric_columns] = converted[numeric_columns].astype(normalized_dtype)
    return converted


def compare_numeric_precision(
    X_original: pd.DataFrame,
    X_converted: pd.DataFrame,
) -> dict[str, Any]:
    """Compare numeric precision and deep memory use between two feature frames.

    Absolute errors use positions where both values are finite. Relative errors
    use finite positions whose original value is non-zero and are defined as
    ``abs(converted - original) / abs(original)``. Original zeros are excluded,
    avoiding artificial infinity or NaN; their exact changes are still included
    in ``changed_value_count``.
    """
    shape_preserved = X_original.shape == X_converted.shape
    columns_preserved = X_original.columns.equals(X_converted.columns)
    index_preserved = X_original.index.equals(X_converted.index)
    if not shape_preserved or not columns_preserved:
        raise ValueError("Precision comparison requires identical shapes and columns.")

    absolute_error_max = 0.0
    absolute_error_sum = 0.0
    absolute_error_count = 0
    relative_error_max = 0.0
    relative_error_sum = 0.0
    relative_error_count = 0
    changed_value_count = 0
    value_count = 0
    missing_values_preserved = True
    infinities_preserved = True
    numeric_columns = X_original.select_dtypes(include="number").columns
    for column in numeric_columns:
        original = X_original[column].to_numpy(dtype=np.float64, copy=False)
        converted = X_converted[column].to_numpy(dtype=np.float64, copy=False)
        finite_mask = np.isfinite(original) & np.isfinite(converted)
        absolute_errors = np.abs(converted[finite_mask] - original[finite_mask])
        if absolute_errors.size:
            absolute_error_max = max(absolute_error_max, float(absolute_errors.max()))
            absolute_error_sum += float(absolute_errors.sum())
            absolute_error_count += int(absolute_errors.size)
        relative_mask = finite_mask & (original != 0)
        relative_errors = np.abs(
            (converted[relative_mask] - original[relative_mask])
            / original[relative_mask]
        )
        if relative_errors.size:
            relative_error_max = max(relative_error_max, float(relative_errors.max()))
            relative_error_sum += float(relative_errors.sum())
            relative_error_count += int(relative_errors.size)
        equal_values = (original == converted) | (
            np.isnan(original) & np.isnan(converted)
        )
        changed_value_count += int((~equal_values).sum())
        value_count += int(original.size)
        missing_values_preserved &= bool(
            np.array_equal(np.isnan(original), np.isnan(converted))
        )
        infinities_preserved &= bool(
            np.array_equal(np.isposinf(original), np.isposinf(converted))
            and np.array_equal(np.isneginf(original), np.isneginf(converted))
        )
    original_memory = memory_usage_mb(X_original)
    converted_memory = memory_usage_mb(X_converted)
    memory_saved = original_memory - converted_memory

    return {
        "original_dtype_counts": {
            str(key): int(value)
            for key, value in X_original.dtypes.astype(str).value_counts().items()
        },
        "converted_dtype_counts": {
            str(key): int(value)
            for key, value in X_converted.dtypes.astype(str).value_counts().items()
        },
        "original_memory_mb": float(original_memory),
        "converted_memory_mb": float(converted_memory),
        "memory_saved_mb": float(memory_saved),
        "memory_reduction_percentage": (
            float(memory_saved / original_memory * 100.0) if original_memory else 0.0
        ),
        "maximum_absolute_error": float(absolute_error_max),
        "mean_absolute_error": (
            float(absolute_error_sum / absolute_error_count)
            if absolute_error_count
            else 0.0
        ),
        "maximum_relative_error": float(relative_error_max),
        "mean_relative_error": (
            float(relative_error_sum / relative_error_count)
            if relative_error_count
            else 0.0
        ),
        "changed_value_count": changed_value_count,
        "changed_value_percentage": (
            float(changed_value_count / value_count * 100.0) if value_count else 0.0
        ),
        "missing_values_preserved": missing_values_preserved,
        "infinities_preserved": infinities_preserved,
        "shape_preserved": shape_preserved,
        "index_preserved": index_preserved,
        "columns_preserved": columns_preserved,
    }


def validate_dtype_conversion(
    X_original: pd.DataFrame,
    X_converted: pd.DataFrame,
    *,
    target_dtype: str = "float32",
) -> None:
    """Validate structural and special-value preservation after numeric conversion."""
    try:
        normalized_dtype = np.dtype(target_dtype).name
    except TypeError as exc:
        raise ValueError(f"Unsupported target dtype: {target_dtype!r}.") from exc
    if normalized_dtype not in ALLOWED_NUMERIC_DTYPES:
        raise ValueError(
            f"Unsupported target dtype {target_dtype!r}; allowed values are "
            f"{sorted(ALLOWED_NUMERIC_DTYPES)}."
        )
    if X_original.shape != X_converted.shape:
        raise ValueError("Dtype conversion must preserve the DataFrame shape.")
    if not X_original.columns.equals(X_converted.columns):
        raise ValueError("Dtype conversion must preserve columns and their order.")
    if not X_original.index.equals(X_converted.index):
        raise ValueError("Dtype conversion must preserve the index and its order.")

    numeric_columns = X_original.select_dtypes(include="number").columns
    non_numeric_columns = X_original.columns.difference(numeric_columns, sort=False)
    for column in numeric_columns:
        original = X_original[column].to_numpy(dtype=np.float64, copy=False)
        converted = X_converted[column].to_numpy(dtype=np.float64, copy=False)
        if not np.array_equal(np.isnan(original), np.isnan(converted)):
            raise ValueError(
                f"Dtype conversion changed missing-value positions in {column!r}."
            )
        if not (
            np.array_equal(np.isposinf(original), np.isposinf(converted))
            and np.array_equal(np.isneginf(original), np.isneginf(converted))
        ):
            raise ValueError(
                "Dtype conversion changed infinity positions or introduced "
                f"overflow in {column!r}."
            )
        if (np.isfinite(original) & np.isinf(converted)).any():
            raise ValueError(
                f"Dtype conversion introduced overflow in {column!r}."
            )
    wrong_dtypes = {
        str(column): str(X_converted[column].dtype)
        for column in numeric_columns
        if np.dtype(X_converted[column].dtype).name != normalized_dtype
    }
    if wrong_dtypes:
        raise ValueError(
            f"Numeric columns do not use target dtype {normalized_dtype}: {wrong_dtypes}."
        )
    for column in non_numeric_columns:
        if not X_original[column].equals(X_converted[column]):
            raise ValueError(f"Non-numeric column {column!r} was modified.")


def _python_scalar(value: Any) -> Any:
    """Convert pandas/NumPy scalar values into serializable Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _target_mapping(series: pd.Series, *, normalize: bool) -> dict[Any, Any]:
    """Return target counts or proportions using plain Python scalar values."""
    values = series.value_counts(dropna=False, normalize=normalize)
    return {
        _python_scalar(key): float(value) if normalize else int(value)
        for key, value in values.items()
    }


def get_dataset_summary(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: Mapping[str, Any] | None = None,
    *,
    quasi_constant_threshold: float = DEFAULT_QUASI_CONSTANT_THRESHOLD,
) -> dict[str, Any]:
    """Compute a serializable, non-mutating summary of dataset structure and size."""
    _validate_quasi_constant_threshold(quasi_constant_threshold)
    validate_dataset(X, y)
    memory_X_mb = memory_usage_mb(X)
    memory_y_mb = memory_usage_mb(y)
    numeric_count = int(X.select_dtypes(include="number").shape[1])
    numeric_X = X.select_dtypes(include="number")
    unique_counts = X.nunique(dropna=True)
    constant_mask = unique_counts.le(1)
    dominant_fractions = X.apply(
        lambda column: (
            float(column.dropna().value_counts(normalize=True).iloc[0])
            if not column.dropna().empty
            else 0.0
        )
    )
    quasi_constant_mask = dominant_fractions.ge(quasi_constant_threshold) & ~constant_mask
    numeric_values = numeric_X.to_numpy()

    summary: dict[str, Any] = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_dtypes": {str(name): str(dtype) for name, dtype in X.dtypes.items()},
        "numeric_feature_count": numeric_count,
        "categorical_feature_count": int(X.shape[1] - numeric_count),
        "missing_values_X": int(X.isna().sum().sum()),
        "missing_values_y": int(y.isna().sum()),
        "duplicate_rows_X": int(X.duplicated().sum()),
        "duplicate_column_name_count": int(X.columns.duplicated().sum()),
        "constant_feature_count": int(constant_mask.sum()),
        "quasi_constant_feature_count": int(quasi_constant_mask.sum()),
        "quasi_constant_threshold": float(quasi_constant_threshold),
        "infinity_count": int(np.isinf(numeric_values).sum()),
        "non_numeric_feature_count": int(X.shape[1] - numeric_count),
        "index_is_unique": bool(X.index.is_unique),
        "index_matches_target": bool(X.index.equals(y.index)),
        "target_unique_values": [_python_scalar(value) for value in y.unique()],
        "target_value_counts": _target_mapping(y, normalize=False),
        "target_proportions": _target_mapping(y, normalize=True),
        "memory_X_mb": float(memory_X_mb),
        "memory_y_mb": float(memory_y_mb),
        "total_memory_mb": float(memory_X_mb + memory_y_mb),
    }
    if metadata is not None:
        summary["metadata"] = {str(key): _python_scalar(value) for key, value in metadata.items()}
    return summary

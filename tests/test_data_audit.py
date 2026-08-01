"""Offline tests for raw-data auditing and explicit numeric dtype conversion."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.data as data_module
from src.data import (
    audit_numeric_features,
    compare_numeric_precision,
    convert_numeric_dtype,
    get_dataset_summary,
    load_dataset,
    validate_dtype_conversion,
)


def _mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "values": [0.0, 1.123456789, np.nan, np.inf, -np.inf],
            "constant": [7.0, 7.0, 7.0, 7.0, 7.0],
            "label": ["a", "b", "a", "b", "a"],
        },
        index=pd.Index([10, 20, 30, 40, 50], name="row"),
    )


def _mock_openml() -> SimpleNamespace:
    X = pd.DataFrame({"a": np.array([1.1, 2.2], dtype=np.float64)})
    return SimpleNamespace(
        data=X,
        target=pd.Series(["False", "True"], name="target", dtype="category"),
        feature_names=["a"],
        target_names=["target"],
        details={"name": "SantanderCustomerSatisfaction", "version": 1},
    )


def test_audit_numeric_features_reports_known_quality_values() -> None:
    audit = audit_numeric_features(_mixed_frame())
    values = audit.set_index("feature").loc["values"]
    constant = audit.set_index("feature").loc["constant"]

    assert len(audit) == 3
    assert values["missing_count"] == 1
    assert values["missing_percentage"] == pytest.approx(20.0)
    assert values["positive_infinity_count"] == 1
    assert values["negative_infinity_count"] == 1
    assert values["zero_count"] == 1
    assert bool(constant["is_constant"])
    assert constant["std"] == pytest.approx(0.0)
    assert audit["memory_bytes"].gt(0).all()


def test_global_summary_reports_quality_flags() -> None:
    X = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "quasi": [2.0, 2.0, 2.0, 3.0],
            "infinite": [0.0, np.inf, 1.0, 2.0],
            "text": ["a", "b", "c", "d"],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="target")
    summary = get_dataset_summary(X, y, quasi_constant_threshold=0.75)

    assert summary["constant_feature_count"] == 1
    assert summary["quasi_constant_feature_count"] == 1
    assert summary["quasi_constant_threshold"] == 0.75
    assert summary["infinity_count"] == 1
    assert summary["non_numeric_feature_count"] == 1
    assert summary["duplicate_column_name_count"] == 0
    assert summary["index_is_unique"] is True
    assert summary["index_matches_target"] is True


def test_convert_numeric_dtype_preserves_structure_and_special_values() -> None:
    original = _mixed_frame()
    converted = convert_numeric_dtype(original)

    assert converted is not original
    assert converted.shape == original.shape
    assert converted.index.equals(original.index)
    assert converted.columns.equals(original.columns)
    assert converted["values"].dtype == np.dtype("float32")
    assert converted["constant"].dtype == np.dtype("float32")
    assert converted["label"].equals(original["label"])
    assert original["values"].dtype == np.dtype("float64")
    assert converted["values"].isna().equals(original["values"].isna())
    assert np.array_equal(
        np.isposinf(converted["values"]), np.isposinf(original["values"])
    )
    assert np.array_equal(
        np.isneginf(converted["values"]), np.isneginf(original["values"])
    )


def test_convert_numeric_dtype_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported numeric dtype"):
        convert_numeric_dtype(_mixed_frame(), dtype="float16")


def test_compare_numeric_precision_reports_memory_and_errors() -> None:
    original = _mixed_frame()
    converted = convert_numeric_dtype(original)
    comparison = compare_numeric_precision(original, converted)

    assert comparison["converted_memory_mb"] < comparison["original_memory_mb"]
    assert comparison["memory_saved_mb"] > 0
    assert comparison["memory_reduction_percentage"] > 0
    assert comparison["maximum_absolute_error"] > 0
    assert comparison["maximum_relative_error"] > 0
    assert comparison["changed_value_count"] == 1
    assert comparison["missing_values_preserved"] is True
    assert comparison["infinities_preserved"] is True
    assert comparison["shape_preserved"] is True
    assert comparison["index_preserved"] is True
    assert comparison["columns_preserved"] is True


def test_validate_dtype_conversion_accepts_correct_conversion() -> None:
    original = _mixed_frame()
    validate_dtype_conversion(original, convert_numeric_dtype(original))


def test_validate_dtype_conversion_rejects_changed_index() -> None:
    original = _mixed_frame()
    converted = convert_numeric_dtype(original)
    converted.index = range(len(converted))
    with pytest.raises(ValueError, match="preserve the index"):
        validate_dtype_conversion(original, converted)


def test_validate_dtype_conversion_rejects_overflow() -> None:
    original = pd.DataFrame({"large": [float(np.finfo(np.float32).max) * 2.0]})
    with np.errstate(over="ignore"):
        converted = convert_numeric_dtype(original)
    with pytest.raises(ValueError, match="infinity positions|overflow"):
        validate_dtype_conversion(original, converted)


@pytest.mark.parametrize("optimize_memory", [False, True])
def test_load_dataset_memory_optimization_is_explicit(
    monkeypatch: pytest.MonkeyPatch, optimize_memory: bool
) -> None:
    monkeypatch.setattr(data_module, "fetch_openml", lambda **_: _mock_openml())
    X, _, metadata = load_dataset(optimize_memory=optimize_memory)

    expected_dtype = np.dtype("float32" if optimize_memory else "float64")
    assert X["a"].dtype == expected_dtype
    assert metadata["project_dataset_name"] == (
        "Santander Customer Transaction Prediction"
    )
    assert metadata["openml_dataset_name"] == "SantanderCustomerSatisfaction"
    assert metadata["dataset_name"] == metadata["openml_dataset_name"]
    assert ("dtype_conversion" in metadata) is optimize_memory

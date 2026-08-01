"""Manually load and display a concise audit of the configured OpenML dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataset_summary, load_dataset  # noqa: E402


def _display_mapping(title: str, values: dict[Any, Any]) -> None:
    """Print a short mapping with a heading."""
    print(f"{title}: {values}")


def main() -> None:
    """Load the dataset and print metadata plus its minimal audit summary."""
    X, y, metadata = load_dataset()
    summary = get_dataset_summary(X, y)
    feature_preview = metadata["feature_names"][:8]

    print(f"Project dataset: {metadata['project_dataset_name']}")
    print(
        f"OpenML dataset: {metadata['openml_dataset_name']} "
        f"(ID {metadata['openml_id']})"
    )
    print(f"Target: {metadata['target_name']}")
    print(f"Dimensions: {summary['n_rows']} rows x {summary['n_features']} features")
    print(
        f"Features (first {len(feature_preview)} of {summary['n_features']}): "
        f"{feature_preview}"
    )
    print(
        "Feature types: "
        f"{summary['numeric_feature_count']} numeric, "
        f"{summary['categorical_feature_count']} categorical/other"
    )
    print(f"Feature dtype counts: {X.dtypes.astype(str).value_counts().to_dict()}")
    print(f"Target dtype: {y.dtype}")
    print(
        f"Missing values: X={summary['missing_values_X']}, "
        f"y={summary['missing_values_y']}"
    )
    print(f"Duplicate feature rows: {summary['duplicate_rows_X']}")
    print(f"Total memory: {summary['total_memory_mb']:.2f} MiB")
    _display_mapping("Target counts", summary["target_value_counts"])
    _display_mapping("Target proportions", summary["target_proportions"])


if __name__ == "__main__":
    main()

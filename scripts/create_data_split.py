"""Create and validate the shared split, exporting metadata only."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data import load_dataset, memory_usage_mb  # noqa: E402
from src.validation import (  # noqa: E402
    create_train_test_split,
    validate_train_test_split,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON suitable for review and reproducibility checks."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Build the common float32 split and save no row-level data."""
    config = load_config()
    output_directory = PROJECT_ROOT / config["paths"]["tables"]
    output_directory.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_dataset(optimize_memory=True)
    partitions = create_train_test_split(X, y)
    X_train, X_test, y_train, y_test = partitions
    validation = validate_train_test_split(X, y, *partitions)

    train_feature_memory = memory_usage_mb(X_train)
    test_feature_memory = memory_usage_mb(X_test)
    train_target_memory = memory_usage_mb(y_train)
    test_target_memory = memory_usage_mb(y_test)
    report: dict[str, Any] = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "openml_id": metadata["openml_id"],
        "project_dataset_name": metadata["project_dataset_name"],
        "openml_dataset_name": metadata["openml_dataset_name"],
        "target_name": metadata["target_name"],
        "total_dimensions": [int(X.shape[0]), int(X.shape[1])],
        "train_dimensions": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_dimensions": [int(X_test.shape[0]), int(X_test.shape[1])],
        "random_state": validation["split_random_state"],
        "test_size": validation["split_test_size"],
        "stratified": validation["stratified"],
        "original_target_distribution": validation[
            "original_target_distribution"
        ],
        "train_target_distribution": validation["train_target_distribution"],
        "test_target_distribution": validation["test_target_distribution"],
        "maximum_target_proportion_difference": validation[
            "maximum_target_proportion_difference"
        ],
        "train_indices_sha256": validation["train_indices_sha256"],
        "test_indices_sha256": validation["test_indices_sha256"],
        "train_feature_memory_mb": float(train_feature_memory),
        "test_feature_memory_mb": float(test_feature_memory),
        "train_target_memory_mb": float(train_target_memory),
        "test_target_memory_mb": float(test_target_memory),
        "train_total_memory_mb": float(train_feature_memory + train_target_memory),
        "test_total_memory_mb": float(test_feature_memory + test_target_memory),
        "overlap_count": validation["overlap_count"],
        "no_index_overlap": validation["overlap_count"] == 0,
    }
    output_path = output_directory / "train_test_split_summary.json"
    _write_json(output_path, report)

    print(f"Train rows: {validation['n_train']}")
    print(f"Test rows: {validation['n_test']}")
    print(f"Original target distribution: {validation['original_target_distribution']}")
    print(f"Train target distribution: {validation['train_target_distribution']}")
    print(f"Test target distribution: {validation['test_target_distribution']}")
    print(
        "Maximum target proportion difference: "
        f"{validation['maximum_target_proportion_difference']:.12g}"
    )
    print(f"Overlap count: {validation['overlap_count']}")
    print(f"Train fingerprint: {validation['train_indices_sha256']}")
    print(f"Test fingerprint: {validation['test_indices_sha256']}")
    print(f"Train total memory: {report['train_total_memory_mb']:.2f} MiB")
    print(f"Test total memory: {report['test_total_memory_mb']:.2f} MiB")
    print(f"Metadata written to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

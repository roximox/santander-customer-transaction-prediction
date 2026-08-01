"""Run the reproducible raw-data and float32 memory audit."""

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
from src.data import (  # noqa: E402
    audit_numeric_features,
    compare_numeric_precision,
    convert_numeric_dtype,
    get_dataset_summary,
    load_dataset,
    validate_dtype_conversion,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict, human-readable JSON with no non-standard NaN values."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Audit raw OpenML data and export reports without saving the dataset."""
    config = load_config()
    target_dtype = str(config["data"]["numeric_dtype"])
    output_directory = PROJECT_ROOT / config["paths"]["tables"]
    output_directory.mkdir(parents=True, exist_ok=True)

    X_raw, y, metadata = load_dataset(optimize_memory=False)
    summary = get_dataset_summary(X_raw, y, metadata)
    feature_audit = audit_numeric_features(X_raw)
    X_optimized = convert_numeric_dtype(X_raw, dtype=target_dtype, copy=True)
    validate_dtype_conversion(X_raw, X_optimized, target_dtype=target_dtype)
    comparison = compare_numeric_precision(X_raw, X_optimized)

    audit_summary: dict[str, Any] = {
        "execution_date": datetime.now(timezone.utc).date().isoformat(),
        "openml_id": metadata["openml_id"],
        "project_dataset_name": metadata["project_dataset_name"],
        "openml_dataset_name": metadata["openml_dataset_name"],
        "target_name": metadata["target_name"],
        "n_rows": summary["n_rows"],
        "n_features": summary["n_features"],
        "target_value_counts": summary["target_value_counts"],
        "target_proportions": summary["target_proportions"],
        "missing_values_X": summary["missing_values_X"],
        "missing_values_y": summary["missing_values_y"],
        "duplicate_rows_X": summary["duplicate_rows_X"],
        "duplicate_column_name_count": summary["duplicate_column_name_count"],
        "constant_feature_count": summary["constant_feature_count"],
        "quasi_constant_feature_count": summary["quasi_constant_feature_count"],
        "quasi_constant_threshold": summary["quasi_constant_threshold"],
        "infinity_count": summary["infinity_count"],
        "non_numeric_feature_count": summary["non_numeric_feature_count"],
        "index_is_unique": summary["index_is_unique"],
        "index_matches_target": summary["index_matches_target"],
        "raw_total_memory_mb": summary["total_memory_mb"],
        "raw_feature_memory_mb": comparison["original_memory_mb"],
        "optimized_feature_memory_mb": comparison["converted_memory_mb"],
        "memory_saved_mb": comparison["memory_saved_mb"],
        "memory_reduction_percentage": comparison[
            "memory_reduction_percentage"
        ],
        "maximum_absolute_error": comparison["maximum_absolute_error"],
        "mean_absolute_error": comparison["mean_absolute_error"],
        "maximum_relative_error": comparison["maximum_relative_error"],
        "mean_relative_error": comparison["mean_relative_error"],
        "changed_value_count": comparison["changed_value_count"],
        "changed_value_percentage": comparison["changed_value_percentage"],
        "missing_values_preserved": comparison["missing_values_preserved"],
        "infinities_preserved": comparison["infinities_preserved"],
    }

    summary_path = output_directory / "data_audit_summary.json"
    feature_path = output_directory / "feature_audit.csv"
    comparison_path = output_directory / "dtype_comparison.json"
    _write_json(summary_path, audit_summary)
    feature_audit.to_csv(feature_path, index=False)
    _write_json(comparison_path, comparison)

    print(
        f"Audited {metadata['project_dataset_name']} from OpenML dataset "
        f"{metadata['openml_dataset_name']} (ID {metadata['openml_id']})."
    )
    print(f"Dimensions: {summary['n_rows']} rows x {summary['n_features']} features")
    print(
        f"Feature memory: {comparison['original_memory_mb']:.2f} MiB "
        f"-> {comparison['converted_memory_mb']:.2f} MiB"
    )
    print(
        f"Saved: {comparison['memory_saved_mb']:.2f} MiB "
        f"({comparison['memory_reduction_percentage']:.2f}%)"
    )
    print(
        "Numeric error: "
        f"max abs={comparison['maximum_absolute_error']:.12g}, "
        f"mean abs={comparison['mean_absolute_error']:.12g}, "
        f"max relative={comparison['maximum_relative_error']:.12g}, "
        f"mean relative={comparison['mean_relative_error']:.12g}"
    )
    print(
        f"Special values preserved: missing={comparison['missing_values_preserved']}, "
        f"infinities={comparison['infinities_preserved']}"
    )
    print(f"Reports written under: {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

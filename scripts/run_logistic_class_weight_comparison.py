"""Run the controlled M01-LR-002 balanced class-weight comparison."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "santander-cache"))

from src.data import load_dataset  # noqa: E402
from src.experiments import run_and_save_experiment  # noqa: E402
from src.logistic_class_weight import (  # noqa: E402
    BALANCED_ID,
    BALANCED_MODEL_NAME,
    BASELINE_ID,
    build_class_weight_comparison,
    create_balanced_logistic_pipeline,
    save_class_weight_comparison,
    save_class_weight_cv_figure,
    save_class_weight_metrics_figure,
)
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports/experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "experiment_registry.csv"
COMPARISON_CSV = PROJECT_ROOT / "reports/tables/logistic_class_weight_comparison.csv"
COMPARISON_JSON = PROJECT_ROOT / "reports/tables/logistic_class_weight_comparison.json"
METRICS_FIGURE = PROJECT_ROOT / "reports/figures/logistic_class_weight_metrics.pdf"
CV_FIGURE = PROJECT_ROOT / "reports/figures/logistic_class_weight_cv.pdf"


def refuse_existing_outputs() -> None:
    """Refuse only M01-LR-002 and its comparison artifacts without overwriting."""
    requested = [
        EXPERIMENTS_DIR / f"{BALANCED_ID}_fold_results.csv",
        EXPERIMENTS_DIR / f"{BALANCED_ID}_summary.json",
        COMPARISON_CSV, COMPARISON_JSON, METRICS_FIGURE, CV_FIGURE,
    ]
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in requested if path.exists()]
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH)
        if "experiment_id" not in registry.columns:
            raise ValueError("Existing experiment registry lacks experiment_id column.")
        if BALANCED_ID in registry["experiment_id"].astype(str).tolist():
            existing.append(f"registry:{BALANCED_ID}")
    if existing:
        details = "\n- ".join(existing)
        raise FileExistsError(
            f"{BALANCED_ID} outputs already exist and were not overwritten. "
            f"Remove only the targeted balanced-experiment artifacts manually:\n- {details}"
        )


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reference file is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run, register, compare, and visualize the single-factor experiment."""
    refuse_existing_outputs()
    baseline_summary = _load_json(EXPERIMENTS_DIR / f"{BASELINE_ID}_summary.json")
    baseline_folds = pd.read_csv(EXPERIMENTS_DIR / f"{BASELINE_ID}_fold_results.csv")
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_hash = split_fingerprint(X_train.index)
    reserved_hash = split_fingerprint(X_reserved.index)
    if train_hash != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(f"Train fingerprint mismatch: {train_hash}; experiment stopped.")
    if reserved_hash != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(f"Reserved test fingerprint mismatch: {reserved_hash}; experiment stopped.")
    del X_reserved, y_reserved

    pipeline = create_balanced_logistic_pipeline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        balanced_folds, balanced_summary = run_and_save_experiment(
            pipeline, X_train, y_train,
            experiment_id=BALANCED_ID,
            model_name=BALANCED_MODEL_NAME,
            member="Member 01",
            branch="feature/data_processing",
            n_jobs=1,
            output_dir=EXPERIMENTS_DIR,
            registry_path=REGISTRY_PATH,
        )
    convergence_messages = [
        str(item.message) for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    if bool(convergence_messages) != bool(balanced_summary["convergence_warning_detected"]):
        raise RuntimeError("Convergence warning capture disagrees with saved summary.")

    comparison = build_class_weight_comparison(baseline_summary, balanced_summary)
    save_class_weight_comparison(
        comparison, csv_path=COMPARISON_CSV, json_path=COMPARISON_JSON
    )
    save_class_weight_metrics_figure(comparison, METRICS_FIGURE)
    save_class_weight_cv_figure(baseline_folds, balanced_folds, CV_FIGURE)

    print("Controlled class-weight comparison (M01-LR-002 minus M01-LR-001):")
    print(comparison.to_string(index=False))
    print("\nM01-LR-002 validation scores by fold:")
    print(
        balanced_folds[
            ["fold", "validation_roc_auc", "validation_average_precision", "validation_recall", "validation_f1"]
        ].to_string(index=False)
    )
    balanced_row = comparison.iloc[1]
    print("\nBalanced-minus-unweighted deltas:")
    for column in comparison.columns:
        if column.startswith("delta_"):
            print(f"  {column}: {balanced_row[column]:.6f}")
    print(f"  generalization_gap: {balanced_row['generalization_gap']:.6f}")
    if convergence_messages:
        print("Convergence warning detected; M01-LR-002 was not altered.")
        for message in convergence_messages:
            print(f"  {message}")
    else:
        print("No ConvergenceWarning detected.")
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

"""Run the Member 4 training-only HistGradientBoosting learning curve."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset  # noqa: E402
from src.gradient_boosting_learning_curve import (  # noqa: E402
    compute_hist_gradient_boosting_learning_curve,
    save_hist_gradient_boosting_learning_curve_figure,
    save_hist_gradient_boosting_learning_curve_results,
)
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"


def print_learning_curve_summary(results: pd.DataFrame) -> None:
    """Print the requested training-only metrics and generalization gaps."""
    print("HistGradientBoosting learning-curve results:")
    print(results.to_string(index=False))
    print("\nLearning-curve summary:")
    for row in results.itertuples(index=False):
        roc_auc_gap = row.train_roc_auc_mean - row.validation_roc_auc_mean
        average_precision_gap = (
            row.train_average_precision_mean - row.validation_average_precision_mean
        )
        print(f"  Train size: {row.train_size}")
        print(f"    Train ROC-AUC mean: {row.train_roc_auc_mean:.6f}")
        print(f"    Validation ROC-AUC mean: {row.validation_roc_auc_mean:.6f}")
        print(f"    ROC-AUC generalization gap: {roc_auc_gap:.6f}")
        print(f"    Train Average Precision mean: {row.train_average_precision_mean:.6f}")
        print(
            "    Validation Average Precision mean: "
            f"{row.validation_average_precision_mean:.6f}"
        )
        print(
            "    Average Precision generalization gap: "
            f"{average_precision_gap:.6f}"
        )


def main() -> None:
    """Compute the learning curve using only the verified training partition."""
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_fingerprint = split_fingerprint(X_train.index)
    reserved_fingerprint = split_fingerprint(X_reserved.index)
    if train_fingerprint != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(
            f"Train fingerprint mismatch: {train_fingerprint}; learning curve stopped."
        )
    if reserved_fingerprint != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(
            "Reserved test fingerprint mismatch: "
            f"{reserved_fingerprint}; learning curve stopped."
        )
    del X_reserved, y_reserved, X, y

    results = compute_hist_gradient_boosting_learning_curve(
        X_train,
        y_train,
        n_jobs=1,
    )
    csv_path = save_hist_gradient_boosting_learning_curve_results(results)
    pdf_path = save_hist_gradient_boosting_learning_curve_figure(results)
    print_learning_curve_summary(results)
    print(f"Saved CSV path: {csv_path}")
    print(f"Saved PDF path: {pdf_path}")
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

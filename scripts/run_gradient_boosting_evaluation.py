"""Run Member 4 tuned-HGB training-only out-of-fold evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset  # noqa: E402
from src.gradient_boosting_evaluation import (  # noqa: E402
    compute_hist_gradient_boosting_oof_metrics,
    compute_hist_gradient_boosting_oof_precision_recall_curve,
    compute_hist_gradient_boosting_oof_predictions,
    compute_hist_gradient_boosting_oof_roc_curve,
    save_hist_gradient_boosting_oof_confusion_matrix_figure,
    save_hist_gradient_boosting_oof_metrics,
    save_hist_gradient_boosting_oof_precision_recall_curve,
    save_hist_gradient_boosting_oof_precision_recall_figure,
    save_hist_gradient_boosting_oof_predictions,
    save_hist_gradient_boosting_oof_roc_curve,
    save_hist_gradient_boosting_oof_roc_figure,
)
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"


def print_oof_summary(
    metrics: dict[str, float | int],
    predictions: pd.DataFrame,
    roc_curve_data: pd.DataFrame,
    precision_recall_curve_data: pd.DataFrame,
) -> None:
    """Print a concise summary of diagnostics derived from one OOF result set."""
    print("Tuned HistGradientBoosting OOF evaluation summary:")
    for metric_name, label in (
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average Precision"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ):
        print(f"  {label}: {metrics[metric_name]:.6f}")
    for metric_name, label in (
        ("true_negatives", "True Negatives"),
        ("false_positives", "False Positives"),
        ("false_negatives", "False Negatives"),
        ("true_positives", "True Positives"),
    ):
        print(f"  {label}: {metrics[metric_name]}")
    print(f"  OOF prediction rows: {len(predictions)}")
    print(f"  ROC curve points: {len(roc_curve_data)}")
    print(f"  Precision-Recall curve points: {len(precision_recall_curve_data)}")


def main() -> None:
    """Compute all tuned-HGB diagnostics from training-only OOF predictions."""
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_fingerprint = split_fingerprint(X_train.index)
    reserved_fingerprint = split_fingerprint(X_reserved.index)
    if train_fingerprint != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(
            f"Train fingerprint mismatch: {train_fingerprint}; evaluation stopped."
        )
    if reserved_fingerprint != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(
            "Reserved test fingerprint mismatch: "
            f"{reserved_fingerprint}; evaluation stopped."
        )
    del X_reserved, y_reserved, X, y

    predictions = compute_hist_gradient_boosting_oof_predictions(
        X_train,
        y_train,
        n_jobs=1,
    )
    metrics = compute_hist_gradient_boosting_oof_metrics(predictions)
    roc_curve_data = compute_hist_gradient_boosting_oof_roc_curve(predictions)
    precision_recall_curve_data = (
        compute_hist_gradient_boosting_oof_precision_recall_curve(predictions)
    )
    predictions_path = save_hist_gradient_boosting_oof_predictions(predictions)
    metrics_path = save_hist_gradient_boosting_oof_metrics(metrics)
    roc_data_path = save_hist_gradient_boosting_oof_roc_curve(roc_curve_data)
    pr_data_path = save_hist_gradient_boosting_oof_precision_recall_curve(
        precision_recall_curve_data
    )
    roc_figure_path = save_hist_gradient_boosting_oof_roc_figure(
        roc_curve_data,
        metrics,
    )
    pr_figure_path = save_hist_gradient_boosting_oof_precision_recall_figure(
        precision_recall_curve_data,
        metrics,
    )
    confusion_matrix_figure_path = (
        save_hist_gradient_boosting_oof_confusion_matrix_figure(metrics)
    )
    print_oof_summary(
        metrics,
        predictions,
        roc_curve_data,
        precision_recall_curve_data,
    )
    print(f"Predictions CSV path: {predictions_path}")
    print(f"Metrics JSON path: {metrics_path}")
    print(f"ROC-data CSV path: {roc_data_path}")
    print(f"PR-data CSV path: {pr_data_path}")
    print(f"ROC PDF path: {roc_figure_path}")
    print(f"PR PDF path: {pr_figure_path}")
    print(f"Confusion-matrix PDF path: {confusion_matrix_figure_path}")
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

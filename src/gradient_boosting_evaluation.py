"""Training-only out-of-fold evaluation helpers for Member 4 HGB."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict

from src.config import get_project_root
from src.evaluation import validate_evaluation_inputs
from src.gradient_boosting_learning_curve import (
    create_tuned_hist_gradient_boosting_estimator,
)
from src.validation import create_stratified_cv
from src.visualization import save_figure


_OOF_REQUIRED_COLUMNS = {
    "row_index",
    "true_target",
    "predicted_class",
    "positive_class_probability",
}
_OOF_METRIC_NAMES = {
    "roc_auc",
    "average_precision",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "balanced_accuracy",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
}


def _resolve_positive_label(y: pd.Series | np.ndarray) -> Any:
    """Choose the Member 4 positive label using the shared project convention."""
    labels = list(pd.unique(y))
    for preferred in ("True", True, 1):
        if preferred in labels:
            return preferred
    return sorted(labels, key=lambda value: (type(value).__name__, repr(value)))[-1]


def _validate_binary_training_data(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Validate caller-provided, index-aligned binary training data."""
    validate_evaluation_inputs(
        X_train,
        y_train,
        model_name="tuned_hist_gradient_boosting",
        experiment_id="M04-HGB-OOF-001",
    )
    if y_train.nunique(dropna=True) != 2:
        raise ValueError("OOF prediction helpers require a binary target.")


def _validate_predictions(predictions: pd.DataFrame) -> None:
    """Validate the minimum OOF prediction schema used by metric helpers."""
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")
    missing_columns = sorted(_OOF_REQUIRED_COLUMNS - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"predictions lack required columns: {missing_columns}.")
    if predictions.empty:
        raise ValueError("predictions must not be empty.")
    if predictions["true_target"].nunique(dropna=True) != 2:
        raise ValueError("predictions must contain exactly two target classes.")
    if predictions["positive_class_probability"].isna().any():
        raise ValueError("positive_class_probability must not contain missing values.")


def _resolve_new_output_path(output_path: str | Path) -> Path:
    """Resolve a new output path inside the project root without overwriting."""
    project_root = get_project_root().resolve()
    path = Path(output_path)
    resolved_path = (path if path.is_absolute() else project_root / path).resolve()
    try:
        resolved_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("output_path must resolve inside the project root.") from error
    if resolved_path.exists():
        raise FileExistsError(f"Output file already exists: {resolved_path}")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _serializable_metric_value(value: Any) -> Any:
    """Convert NumPy metric scalars while preserving strict JSON validation."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_curve_data(curve_data: pd.DataFrame, required_columns: set[str]) -> None:
    """Validate curve data frames before their numerical values are persisted."""
    if not isinstance(curve_data, pd.DataFrame):
        raise TypeError("curve_data must be a pandas DataFrame.")
    missing_columns = sorted(required_columns - set(curve_data.columns))
    if missing_columns:
        raise ValueError(f"curve_data lacks required columns: {missing_columns}.")


def _validate_figure_metrics(metrics: Mapping[str, Any], required_names: set[str]) -> None:
    """Validate supplied aggregate metrics required by an OOF figure."""
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping.")
    missing_metrics = sorted(required_names - set(metrics))
    if missing_metrics:
        raise ValueError(f"metrics lack required values: {missing_metrics}.")


def _positive_target_vector(predictions: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Return the shared positive label and binary target vector for predictions."""
    positive = _resolve_positive_label(predictions["true_target"])
    return positive, np.asarray(predictions["true_target"] == positive, dtype=int)


def compute_hist_gradient_boosting_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_jobs: int | None = 1,
) -> pd.DataFrame:
    """Generate index-aligned tuned-HGB OOF predictions from training data only.

    Each row is predicted by a model fitted on the other four configured
    stratified folds. This function has no final-test-set argument and performs
    neither persistence nor any threshold or hyperparameter optimization.
    """
    _validate_binary_training_data(X_train, y_train)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, type(None))):
        raise ValueError("n_jobs must be an integer or None.")

    cv = create_stratified_cv()
    positive = _resolve_positive_label(y_train)
    classes = sorted(pd.unique(y_train), key=lambda value: (type(value).__name__, repr(value)))
    positive_position = classes.index(positive)

    probabilities = cross_val_predict(
        create_tuned_hist_gradient_boosting_estimator(),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
    )
    positive_probabilities = np.asarray(probabilities)[:, positive_position]
    negative = next(label for label in pd.unique(y_train) if label != positive)
    predicted_classes = np.where(positive_probabilities >= 0.5, positive, negative)

    fold_membership = np.empty(len(X_train), dtype=int)
    for fold, (_, validation_positions) in enumerate(cv.split(X_train, y_train), start=1):
        fold_membership[validation_positions] = fold

    return pd.DataFrame(
        {
            "row_index": X_train.index.to_numpy(copy=True),
            "fold": fold_membership,
            "true_target": y_train.to_numpy(copy=True),
            "predicted_class": predicted_classes,
            "positive_class_probability": positive_probabilities,
        },
        index=X_train.index.copy(),
    )


def compute_hist_gradient_boosting_oof_metrics(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    """Compute aggregate, JSON-compatible binary metrics from HGB OOF predictions."""
    _validate_predictions(predictions)
    positive, y_true_binary = _positive_target_vector(predictions)
    y_predicted = predictions["predicted_class"]
    probability_scores = predictions["positive_class_probability"]
    negative = next(label for label in pd.unique(predictions["true_target"]) if label != positive)
    tn, fp, fn, tp = confusion_matrix(
        predictions["true_target"], y_predicted, labels=[negative, positive]
    ).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true_binary, probability_scores)),
        "average_precision": float(average_precision_score(y_true_binary, probability_scores)),
        "f1": float(f1_score(predictions["true_target"], y_predicted, pos_label=positive, zero_division=0)),
        "precision": float(precision_score(predictions["true_target"], y_predicted, pos_label=positive, zero_division=0)),
        "recall": float(recall_score(predictions["true_target"], y_predicted, pos_label=positive, zero_division=0)),
        "accuracy": float(accuracy_score(predictions["true_target"], y_predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(predictions["true_target"], y_predicted)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def compute_hist_gradient_boosting_oof_roc_curve(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return numerical ROC-curve points derived from HGB OOF probabilities."""
    _validate_predictions(predictions)
    _, y_true_binary = _positive_target_vector(predictions)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true_binary, predictions["positive_class_probability"]
    )
    return pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "threshold": thresholds,
        }
    )


def compute_hist_gradient_boosting_oof_precision_recall_curve(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return numerical precision-recall points derived from HGB OOF probabilities."""
    _validate_predictions(predictions)
    _, y_true_binary = _positive_target_vector(predictions)
    precision, recall, thresholds = precision_recall_curve(
        y_true_binary, predictions["positive_class_probability"]
    )
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    )


def save_hist_gradient_boosting_oof_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path = "reports/tables/M04-HGB-OOF-001_predictions.csv",
) -> Path:
    """Save previously computed HGB OOF predictions without their pandas index."""
    _validate_predictions(predictions)
    resolved_path = _resolve_new_output_path(output_path)
    predictions.to_csv(resolved_path, index=False, encoding="utf-8")
    return resolved_path


def save_hist_gradient_boosting_oof_metrics(
    metrics: Mapping[str, Any],
    output_path: str | Path = "reports/tables/M04-HGB-OOF-001_metrics.json",
) -> Path:
    """Save previously computed aggregate HGB OOF metrics as strict JSON."""
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping.")
    missing_metrics = sorted(_OOF_METRIC_NAMES - set(metrics))
    if missing_metrics:
        raise ValueError(f"metrics lack required values: {missing_metrics}.")
    payload = {str(key): _serializable_metric_value(value) for key, value in metrics.items()}
    try:
        encoded_metrics = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metrics must contain strict JSON-compatible values.") from error

    resolved_path = _resolve_new_output_path(output_path)
    resolved_path.write_text(encoded_metrics + "\n", encoding="utf-8")
    return resolved_path


def save_hist_gradient_boosting_oof_roc_curve(
    roc_curve_data: pd.DataFrame,
    output_path: str | Path = "reports/tables/M04-HGB-OOF-001_roc_curve.csv",
) -> Path:
    """Save previously computed HGB OOF ROC numerical data."""
    _validate_curve_data(
        roc_curve_data,
        {"false_positive_rate", "true_positive_rate", "threshold"},
    )
    resolved_path = _resolve_new_output_path(output_path)
    roc_curve_data.to_csv(resolved_path, index=False, encoding="utf-8")
    return resolved_path


def save_hist_gradient_boosting_oof_precision_recall_curve(
    precision_recall_curve_data: pd.DataFrame,
    output_path: str | Path = "reports/tables/M04-HGB-OOF-001_precision_recall_curve.csv",
) -> Path:
    """Save HGB OOF precision-recall data, allowing sklearn's final NaN threshold."""
    _validate_curve_data(
        precision_recall_curve_data,
        {"precision", "recall", "threshold"},
    )
    resolved_path = _resolve_new_output_path(output_path)
    precision_recall_curve_data.to_csv(resolved_path, index=False, encoding="utf-8")
    return resolved_path


def save_hist_gradient_boosting_oof_roc_figure(
    roc_curve_data: pd.DataFrame,
    metrics: Mapping[str, Any],
    output_path: str | Path = "reports/figures/M04-HGB-OOF-001_roc_curve.pdf",
) -> Path:
    """Save a ROC figure using supplied tuned-HGB OOF diagnostics only."""
    _validate_curve_data(
        roc_curve_data,
        {"false_positive_rate", "true_positive_rate", "threshold"},
    )
    _validate_figure_metrics(metrics, {"roc_auc"})
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    try:
        axis.plot(
            roc_curve_data["false_positive_rate"],
            roc_curve_data["true_positive_rate"],
            label=f"Tuned HGB (ROC-AUC = {float(metrics['roc_auc']):.4f})",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.4", label="Random classifier")
        axis.set(
            title="Tuned HistGradientBoosting OOF ROC Curve",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend(loc="lower right")
        return save_figure(figure, _resolve_new_output_path(output_path))
    finally:
        plt.close(figure)


def save_hist_gradient_boosting_oof_precision_recall_figure(
    precision_recall_curve_data: pd.DataFrame,
    metrics: Mapping[str, Any],
    output_path: str | Path = "reports/figures/M04-HGB-OOF-001_precision_recall_curve.pdf",
) -> Path:
    """Save a precision-recall figure using supplied tuned-HGB OOF data only."""
    _validate_curve_data(
        precision_recall_curve_data,
        {"precision", "recall", "threshold"},
    )
    _validate_figure_metrics(metrics, {"average_precision"})
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    try:
        axis.plot(
            precision_recall_curve_data["recall"],
            precision_recall_curve_data["precision"],
            label=(
                "Tuned HGB "
                f"(Average Precision = {float(metrics['average_precision']):.4f})"
            ),
        )
        axis.set(
            title="Tuned HistGradientBoosting OOF Precision-Recall Curve",
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend(loc="lower left")
        return save_figure(figure, _resolve_new_output_path(output_path))
    finally:
        plt.close(figure)


def save_hist_gradient_boosting_oof_confusion_matrix_figure(
    metrics: Mapping[str, Any],
    output_path: str | Path = "reports/figures/M04-HGB-OOF-001_confusion_matrix.pdf",
) -> Path:
    """Save a confusion-matrix figure from supplied tuned-HGB OOF metric counts."""
    _validate_figure_metrics(
        metrics,
        {"true_negatives", "false_positives", "false_negatives", "true_positives"},
    )
    matrix = np.array(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    try:
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis, label="Count")
        axis.set(
            title="Tuned HistGradientBoosting OOF Confusion Matrix",
            xlabel="Predicted class",
            ylabel="Actual class",
            xticks=(0, 1),
            xticklabels=("Negative", "Positive"),
            yticks=(0, 1),
            yticklabels=("Negative", "Positive"),
        )
        for row, column in np.ndindex(matrix.shape):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
        return save_figure(figure, _resolve_new_output_path(output_path))
    finally:
        plt.close(figure)

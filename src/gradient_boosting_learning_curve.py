"""Training-only learning-curve utilities for Member 4 HistGradientBoosting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import learning_curve

from src.config import get_project_root
from src.evaluation import get_scoring_metrics
from src.modeling import create_hist_gradient_boosting_classifier
from src.validation import create_stratified_cv

HIST_GRADIENT_BOOSTING_LEARNING_CURVE_TRAIN_SIZES = [0.10, 0.25, 0.50, 0.75, 1.00]
_LEARNING_CURVE_METRICS = ("roc_auc", "average_precision")
_LEARNING_CURVE_RESULT_COLUMNS = (
    "train_size",
    "train_roc_auc_mean",
    "train_roc_auc_std",
    "validation_roc_auc_mean",
    "validation_roc_auc_std",
    "train_average_precision_mean",
    "train_average_precision_std",
    "validation_average_precision_mean",
    "validation_average_precision_std",
)


def create_tuned_hist_gradient_boosting_estimator() -> HistGradientBoostingClassifier:
    """Return the unfitted, frozen M04-HGB-SEARCH-001 best HGB estimator."""
    estimator = create_hist_gradient_boosting_classifier(
        learning_rate=0.05,
        max_iter=700,
        max_leaf_nodes=31,
        l2_regularization=10.0,
        random_state=42,
    )
    return estimator.set_params(min_samples_leaf=100)


def compute_hist_gradient_boosting_learning_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_jobs: int | None = 1,
) -> pd.DataFrame:
    """Compute M04 HGB learning curves from caller-provided training data only.

    The estimator is fitted solely inside scikit-learn's ``learning_curve``
    routine on subsets of ``X_train``. No dataset loading, persistence, or final
    test-partition access is performed here.
    """
    cv = create_stratified_cv()
    scoring_metrics = get_scoring_metrics()
    metric_results: dict[str, tuple[object, object, object]] = {}

    for metric_name in _LEARNING_CURVE_METRICS:
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=create_tuned_hist_gradient_boosting_estimator(),
            X=X_train,
            y=y_train,
            train_sizes=HIST_GRADIENT_BOOSTING_LEARNING_CURVE_TRAIN_SIZES,
            cv=cv,
            scoring=scoring_metrics[metric_name],
            n_jobs=n_jobs,
            shuffle=True,
            random_state=42,
            return_times=False,
        )
        metric_results[metric_name] = (train_sizes, train_scores, validation_scores)

    train_sizes, roc_auc_train, roc_auc_validation = metric_results["roc_auc"]
    _, average_precision_train, average_precision_validation = metric_results[
        "average_precision"
    ]
    return pd.DataFrame(
        {
            "train_size": train_sizes.astype(int),
            "train_roc_auc_mean": roc_auc_train.mean(axis=1),
            "train_roc_auc_std": roc_auc_train.std(axis=1),
            "validation_roc_auc_mean": roc_auc_validation.mean(axis=1),
            "validation_roc_auc_std": roc_auc_validation.std(axis=1),
            "train_average_precision_mean": average_precision_train.mean(axis=1),
            "train_average_precision_std": average_precision_train.std(axis=1),
            "validation_average_precision_mean": average_precision_validation.mean(axis=1),
            "validation_average_precision_std": average_precision_validation.std(axis=1),
        }
    )


def save_hist_gradient_boosting_learning_curve_results(
    results: pd.DataFrame,
    output_path: str | Path = "reports/tables/M04-HGB-learning-curve.csv",
) -> Path:
    """Save computed M04 HGB learning-curve results without overwriting files."""
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")

    project_root = get_project_root().resolve()
    path = Path(output_path)
    resolved_path = (path if path.is_absolute() else project_root / path).resolve()
    try:
        resolved_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("output_path must resolve inside the project root.") from error
    if resolved_path.exists():
        raise FileExistsError(f"Learning-curve results already exist: {resolved_path}")

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(resolved_path, index=False, encoding="utf-8")
    return resolved_path


def save_hist_gradient_boosting_learning_curve_figure(
    results: pd.DataFrame,
    output_path: str | Path = "reports/figures/M04-HGB-learning-curve.pdf",
) -> Path:
    """Save a two-panel HGB learning-curve figure from supplied results only."""
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    missing_columns = sorted(set(_LEARNING_CURVE_RESULT_COLUMNS) - set(results.columns))
    if missing_columns:
        raise ValueError(f"Learning-curve results lack required columns: {missing_columns}.")

    project_root = get_project_root().resolve()
    path = Path(output_path)
    resolved_path = (path if path.is_absolute() else project_root / path).resolve()
    try:
        resolved_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("output_path must resolve inside the project root.") from error
    if resolved_path.exists():
        raise FileExistsError(f"Learning-curve figure already exists: {resolved_path}")

    figure, axes = plt.subplots(ncols=2, figsize=(12, 5.2), sharex=True)
    try:
        train_size = results["train_size"]
        panels = (
            (
                axes[0],
                "ROC-AUC Learning Curve",
                "ROC-AUC",
                "train_roc_auc",
                "validation_roc_auc",
            ),
            (
                axes[1],
                "Average Precision Learning Curve",
                "Average Precision",
                "train_average_precision",
                "validation_average_precision",
            ),
        )
        for axis, title, ylabel, train_prefix, validation_prefix in panels:
            train_mean = results[f"{train_prefix}_mean"]
            train_std = results[f"{train_prefix}_std"]
            validation_mean = results[f"{validation_prefix}_mean"]
            validation_std = results[f"{validation_prefix}_std"]
            axis.plot(train_size, train_mean, marker="o", label="Train mean")
            axis.fill_between(
                train_size,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.18,
                label="Train ±1 std",
            )
            axis.plot(train_size, validation_mean, marker="s", label="Validation mean")
            axis.fill_between(
                train_size,
                validation_mean - validation_std,
                validation_mean + validation_std,
                alpha=0.18,
                label="Validation ±1 std",
            )
            axis.set_title(title)
            axis.set_xlabel("Training size")
            axis.set_ylabel(ylabel)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.25, linestyle="--")
            axis.legend(frameon=False)

        figure.tight_layout()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_path, format="pdf", bbox_inches="tight")
    finally:
        plt.close(figure)
    return resolved_path

"""Training-only, stratified learning-curve utilities.

The public API deliberately has no final-test argument.  Callers must pass the
shared training partition and every validation score is produced inside CV.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.evaluation import get_scoring_metrics, validate_evaluation_inputs

METRIC_NAMES = (
    "roc_auc", "average_precision", "f1", "precision", "recall",
    "accuracy", "balanced_accuracy",
)


def validate_train_size_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize strictly increasing fractions in ``(0, 1]``."""
    if isinstance(fractions, (str, bytes)) or not isinstance(fractions, Sequence):
        raise ValueError("train_size_fractions must be a non-empty sequence.")
    if not fractions:
        raise ValueError("train_size_fractions must not be empty.")
    normalized: list[float] = []
    for value in fractions:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Every train-size fraction must be numeric.")
        fraction = float(value)
        if not np.isfinite(fraction) or fraction <= 0 or fraction > 1:
            raise ValueError("Every train-size fraction must be in (0, 1].")
        normalized.append(fraction)
    if len(set(normalized)) != len(normalized):
        raise ValueError("train_size_fractions must not contain duplicates.")
    if normalized != sorted(normalized):
        raise ValueError("train_size_fractions must be in increasing order.")
    return tuple(normalized)


def create_stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    fraction: float,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return an index-preserving stratified subset without mutating inputs."""
    validated = validate_train_size_fractions([fraction])[0]
    validate_evaluation_inputs(X, y, model_name="subsample", experiment_id="subsample")
    if validated == 1.0:
        return X.copy(), y.copy()
    minimum_size = y.nunique() * 2
    requested_size = int(np.floor(len(y) * validated))
    if requested_size < minimum_size:
        raise ValueError(
            f"fraction={validated:g} yields {requested_size} rows, too few for "
            "a stratified subset containing every class."
        )
    positions = np.arange(len(y))
    selected, _ = train_test_split(
        positions,
        train_size=requested_size,
        stratify=y.to_numpy(),
        random_state=random_state,
        shuffle=True,
    )
    return X.iloc[selected].copy(), y.iloc[selected].copy()


def _iteration_count(estimator: BaseEstimator) -> float:
    fitted = estimator
    if hasattr(estimator, "named_steps"):
        fitted = estimator.named_steps.get("classifier", estimator)
    values = getattr(fitted, "n_iter_", None)
    return float(np.max(values)) if values is not None else float("nan")


def compute_learning_curve(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    configuration_id: str,
    train_size_fractions: Sequence[float],
    cv: StratifiedKFold,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit one cloned estimator per fold/fraction and return detailed scores."""
    validate_evaluation_inputs(
        X, y, model_name=type(estimator).__name__, experiment_id=configuration_id
    )
    fractions = validate_train_size_fractions(train_size_fractions)
    if not isinstance(cv, StratifiedKFold):
        raise ValueError("cv must be a StratifiedKFold instance.")
    scoring = get_scoring_metrics()
    rows: list[dict[str, Any]] = []
    for fold, (train_positions, validation_positions) in enumerate(cv.split(X, y), 1):
        X_fold, y_fold = X.iloc[train_positions], y.iloc[train_positions]
        X_validation, y_validation = X.iloc[validation_positions], y.iloc[validation_positions]
        for fraction in fractions:
            X_subset, y_subset = create_stratified_subsample(
                X_fold, y_fold, fraction, random_state=random_state + fold
            )
            fitted = clone(estimator)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                fitted.fit(X_subset, y_subset)
            fit_time = time.perf_counter() - started
            score_started = time.perf_counter()
            metric_values: dict[str, float] = {}
            for name, scorer in scoring.items():
                metric_values[f"train_{name}"] = float(scorer(fitted, X_subset, y_subset))
                metric_values[f"validation_{name}"] = float(
                    scorer(fitted, X_validation, y_validation)
                )
            score_time = time.perf_counter() - score_started
            positive_label = sorted(pd.unique(y), key=lambda item: (type(item).__name__, repr(item)))[-1]
            for preferred in ("True", True, 1):
                if preferred in set(pd.unique(y)):
                    positive_label = preferred
                    break
            rows.append({
                "configuration_id": configuration_id,
                "fold": fold,
                "train_fraction": fraction,
                "train_size": len(X_subset),
                "validation_size": len(X_validation),
                "positive_train_count": int((y_subset == positive_label).sum()),
                "positive_validation_count": int((y_validation == positive_label).sum()),
                "fit_time": fit_time,
                "score_time": score_time,
                **metric_values,
                "convergence_warning": any(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                ),
                "n_iter": _iteration_count(fitted),
            })
    return pd.DataFrame(rows)


def summarize_learning_curve(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate folds by configuration and fraction using population std."""
    required = {"configuration_id", "train_fraction", "train_size", "fit_time", "convergence_warning", "n_iter"}
    required.update(f"{split}_{metric}" for split in ("train", "validation") for metric in METRIC_NAMES)
    missing = sorted(required.difference(fold_results.columns))
    if missing:
        raise ValueError(f"Learning-curve results lack required columns: {missing}.")
    rows: list[dict[str, Any]] = []
    for (configuration_id, fraction), group in fold_results.groupby(
        ["configuration_id", "train_fraction"], sort=True
    ):
        row: dict[str, Any] = {
            "configuration_id": configuration_id,
            "train_fraction": float(fraction),
            "train_size_mean": float(group["train_size"].mean()),
        }
        for metric in METRIC_NAMES:
            train = group[f"train_{metric}"]
            validation = group[f"validation_{metric}"]
            row[f"train_{metric}_mean"] = float(train.mean())
            row[f"train_{metric}_std"] = float(train.std(ddof=0))
            row[f"validation_{metric}_mean"] = float(validation.mean())
            row[f"validation_{metric}_std"] = float(validation.std(ddof=0))
        row["roc_auc_generalization_gap"] = row["train_roc_auc_mean"] - row["validation_roc_auc_mean"]
        row["average_precision_generalization_gap"] = row["train_average_precision_mean"] - row["validation_average_precision_mean"]
        row["fit_time_mean"] = float(group["fit_time"].mean())
        row["fit_time_std"] = float(group["fit_time"].std(ddof=0))
        row["score_time_mean"] = float(group["score_time"].mean())
        row["convergence_rate"] = float((~group["convergence_warning"].astype(bool)).mean())
        row["mean_n_iter"] = float(group["n_iter"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def create_learning_curve_figures(
    summary: pd.DataFrame,
    output_directory: str | Path,
) -> list[Path]:
    """Create the four required vector figures from a saved/derived summary."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    styles = [("o", "tab:blue"), ("s", "tab:orange")]

    def metric_figure(metric: str, title: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(9.5, 6))
        for (configuration, group), (marker, color) in zip(summary.groupby("configuration_id"), styles):
            group = group.sort_values("train_size_mean")
            for split, linestyle in (("train", "--"), ("validation", "-")):
                ax.errorbar(group["train_size_mean"], group[f"{split}_{metric}_mean"],
                            yerr=group[f"{split}_{metric}_std"], marker=marker,
                            linestyle=linestyle, color=color, capsize=3,
                            label=f"{configuration} — {split}")
        ax.set(title=title, xlabel="Actual training observations", ylabel=metric.replace("_", " ").title())
        ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); path = output / filename; fig.savefig(path, format="pdf", bbox_inches="tight"); plt.close(fig)
        return path

    paths = [
        metric_figure("roc_auc", "Logistic Regression Learning Curve — ROC-AUC", "logistic_learning_curve_roc_auc.pdf"),
        metric_figure("average_precision", "Logistic Regression Learning Curve — Average Precision", "logistic_learning_curve_average_precision.pdf"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for (configuration, group), (marker, color) in zip(summary.groupby("configuration_id"), styles):
        group = group.sort_values("train_size_mean")
        ax.errorbar(group["train_size_mean"], group["fit_time_mean"], yerr=group["fit_time_std"], marker=marker, color=color, capsize=3, label=configuration)
    ax.set(title="Logistic Regression Learning Curve — Fit Time", xlabel="Actual training observations", ylabel="Mean fit time (seconds)")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    path = output / "logistic_learning_curve_fit_time.pdf"; fig.savefig(path, format="pdf", bbox_inches="tight"); plt.close(fig); paths.append(path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, metric in zip(axes.ravel(), ("precision", "recall", "f1", "balanced_accuracy")):
        for (configuration, group), (marker, color) in zip(summary.groupby("configuration_id"), styles):
            group = group.sort_values("train_size_mean")
            ax.errorbar(group["train_size_mean"], group[f"validation_{metric}_mean"], yerr=group[f"validation_{metric}_std"], marker=marker, color=color, capsize=3, label=configuration)
        ax.set(title=metric.replace("_", " ").title(), ylabel="Validation score"); ax.grid(alpha=.25)
    for ax in axes[-1]: ax.set_xlabel("Actual training observations")
    axes[0, 0].legend(frameon=False, fontsize=8); fig.suptitle("Validation Threshold Metrics by Training Size"); fig.tight_layout()
    path = output / "logistic_learning_curve_threshold_metrics.pdf"; fig.savefig(path, format="pdf", bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths

"""Controlled reporting helpers for the Logistic Regression class-weight study."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_project_root
from src.modeling import create_logistic_regression_pipeline

BASELINE_ID = "M01-LR-001"
BALANCED_ID = "M01-LR-002"
BALANCED_MODEL_NAME = "Logistic Regression L2 Balanced"

METRIC_NAMES = (
    "roc_auc", "average_precision", "precision", "recall", "f1", "accuracy",
    "balanced_accuracy",
)
DELTA_COLUMNS = tuple(f"delta_{name}" for name in METRIC_NAMES)
COMPARISON_COLUMNS = (
    "experiment_id", "class_weight", "roc_auc_mean", "roc_auc_std",
    "average_precision_mean", "average_precision_std", "precision_mean",
    "recall_mean", "f1_mean", "accuracy_mean", "balanced_accuracy_mean",
    "train_roc_auc_mean", "validation_roc_auc_mean", "generalization_gap",
    "fit_time_mean", *DELTA_COLUMNS,
)


def create_balanced_logistic_pipeline() -> Any:
    """Return the exact unfitted M01-LR-002 Pipeline with one changed factor."""
    return create_logistic_regression_pipeline(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )


def build_class_weight_comparison(
    baseline_summary: Mapping[str, Any],
    balanced_summary: Mapping[str, Any],
) -> pd.DataFrame:
    """Build two controlled rows and balanced-minus-baseline metric deltas."""
    summaries = (baseline_summary, balanced_summary)
    expected_ids = (BASELINE_ID, BALANCED_ID)
    if tuple(summary.get("experiment_id") for summary in summaries) != expected_ids:
        raise ValueError(f"Summaries must be ordered as {expected_ids}.")
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        metrics = summary["metrics"]
        train_auc = float(metrics["roc_auc"]["train_mean"])
        validation_auc = float(metrics["roc_auc"]["validation_mean"])
        parameters = summary.get("estimator_parameters", {})
        rows.append(
            {
                "experiment_id": summary["experiment_id"],
                "class_weight": parameters.get("classifier__class_weight"),
                "roc_auc_mean": validation_auc,
                "roc_auc_std": metrics["roc_auc"]["validation_std"],
                "average_precision_mean": metrics["average_precision"]["validation_mean"],
                "average_precision_std": metrics["average_precision"]["validation_std"],
                "precision_mean": metrics["precision"]["validation_mean"],
                "recall_mean": metrics["recall"]["validation_mean"],
                "f1_mean": metrics["f1"]["validation_mean"],
                "accuracy_mean": metrics["accuracy"]["validation_mean"],
                "balanced_accuracy_mean": metrics["balanced_accuracy"]["validation_mean"],
                "train_roc_auc_mean": train_auc,
                "validation_roc_auc_mean": validation_auc,
                "generalization_gap": train_auc - validation_auc,
                "fit_time_mean": summary["fit_time_mean"],
            }
        )
    table = pd.DataFrame(rows)
    # Some pandas versions infer a string dtype for mixed None/string values and
    # replace None with NaN. Preserve the semantic JSON null explicitly so strict
    # serialization remains portable across the supported environments.
    table["class_weight"] = pd.Series(
        [None, "balanced"], index=table.index, dtype=object
    )
    baseline, balanced = table.iloc[0], table.iloc[1]
    for metric, delta_column in zip(METRIC_NAMES, DELTA_COLUMNS):
        source = f"{metric}_mean"
        delta = float(balanced[source] - baseline[source])
        table[delta_column] = [0.0, delta]
    return table.loc[:, COMPARISON_COLUMNS]


def _project_path(path: str | Path) -> Path:
    root = get_project_root().resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Class-weight outputs must remain inside the project.") from exc
    return candidate


def _has_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_has_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_absolute_path(item) for item in value)
    return False


def save_class_weight_comparison(
    table: pd.DataFrame,
    *,
    csv_path: str | Path = "reports/tables/logistic_class_weight_comparison.csv",
    json_path: str | Path = "reports/tables/logistic_class_weight_comparison.json",
) -> tuple[Path, Path]:
    """Save the controlled comparison as CSV and strict JSON without overwrite."""
    missing = set(COMPARISON_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Comparison lacks columns: {sorted(missing)}.")
    csv_file, json_file = _project_path(csv_path), _project_path(json_path)
    if csv_file.exists() or json_file.exists():
        raise FileExistsError("Logistic class-weight comparison already exists.")
    records = table.loc[:, COMPARISON_COLUMNS].to_dict(orient="records")
    if _has_absolute_path(records):
        raise ValueError("Comparison must not contain absolute paths.")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, COMPARISON_COLUMNS].to_csv(csv_file, index=False)
    json_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_file, json_file


def save_class_weight_metrics_figure(
    table: pd.DataFrame,
    path: str | Path = "reports/figures/logistic_class_weight_metrics.pdf",
) -> Path:
    """Compare all requested validation metrics for unweighted and balanced L2."""
    output = _project_path(path)
    if output.exists():
        raise FileExistsError(f"Figure already exists: {output.name}.")
    plot = table.set_index("experiment_id")[[f"{name}_mean" for name in METRIC_NAMES]]
    plot.index = ["L2 unweighted", "L2 balanced"]
    plot.columns = [
        "ROC-AUC", "Average Precision", "Precision", "Recall", "F1",
        "Accuracy", "Balanced Accuracy",
    ]
    figure, axis = plt.subplots(figsize=(11, 6))
    plot.T.plot(kind="bar", ax=axis, width=0.75)
    axis.set_title("Effect of Balanced Class Weighting on Logistic Regression L2")
    axis.set_xlabel("Validation metric")
    axis.set_ylabel("Mean validation score")
    axis.set_ylim(0, 1)
    axis.legend(title="Experiment", frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output


def save_class_weight_cv_figure(
    baseline_folds: pd.DataFrame,
    balanced_folds: pd.DataFrame,
    path: str | Path = "reports/figures/logistic_class_weight_cv.pdf",
) -> Path:
    """Compare four validation metrics fold by fold for both experiments."""
    output = _project_path(path)
    if output.exists():
        raise FileExistsError(f"Figure already exists: {output.name}.")
    metrics = (
        ("validation_roc_auc", "ROC-AUC"),
        ("validation_average_precision", "Average Precision"),
        ("validation_recall", "Recall"),
        ("validation_f1", "F1"),
    )
    required = {"fold", *(name for name, _ in metrics)}
    for folds in (baseline_folds, balanced_folds):
        if not required <= set(folds.columns):
            raise ValueError(f"Fold results lack columns: {sorted(required)}.")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, (column, title) in zip(axes.flat, metrics):
        axis.plot(baseline_folds["fold"], baseline_folds[column], marker="o", label="L2 unweighted")
        axis.plot(balanced_folds["fold"], balanced_folds[column], marker="s", label="L2 balanced")
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_xticks(baseline_folds["fold"])
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Validation score")
    axes[1, 0].set_ylabel("Validation score")
    axes[1, 0].set_xlabel("Cross-validation fold")
    axes[1, 1].set_xlabel("Cross-validation fold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Logistic Regression Class Weighting — Fold Stability", y=0.98)
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=2, frameon=False,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output

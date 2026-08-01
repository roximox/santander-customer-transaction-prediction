"""Reporting helpers for the single Logistic Regression L2 baseline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_project_root
from src.modeling import create_logistic_regression_pipeline

LOGISTIC_EXPERIMENT_ID = "M01-LR-001"
LOGISTIC_MODEL_NAME = "Logistic Regression L2 Baseline"

COMPARISON_COLUMNS = (
    "experiment_id", "model_name", "roc_auc_mean", "roc_auc_std",
    "average_precision_mean", "average_precision_std", "f1_mean", "f1_std",
    "precision_mean", "recall_mean", "accuracy_mean",
    "balanced_accuracy_mean", "fit_time_mean", "score_time_mean",
    "train_roc_auc_mean", "validation_roc_auc_mean",
    "roc_auc_generalization_gap",
)


def create_logistic_baseline_pipeline() -> Any:
    """Return the exact unfitted Pipeline specified for M01-LR-001."""
    return create_logistic_regression_pipeline(
        penalty="l2",
        C=1.0,
        class_weight=None,
        max_iter=1000,
        random_state=42,
        with_mean=True,
        with_std=True,
    )


def build_logistic_comparison(
    summaries: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a common metric table from Dummy and Logistic summaries."""
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        metrics = summary["metrics"]
        train_auc = metrics["roc_auc"].get("train_mean")
        validation_auc = metrics["roc_auc"]["validation_mean"]
        if train_auc is None:
            raise ValueError("Summary must contain train ROC-AUC for gap calculation.")
        rows.append(
            {
                "experiment_id": summary["experiment_id"],
                "model_name": summary["model_name"],
                "roc_auc_mean": validation_auc,
                "roc_auc_std": metrics["roc_auc"]["validation_std"],
                "average_precision_mean": metrics["average_precision"]["validation_mean"],
                "average_precision_std": metrics["average_precision"]["validation_std"],
                "f1_mean": metrics["f1"]["validation_mean"],
                "f1_std": metrics["f1"]["validation_std"],
                "precision_mean": metrics["precision"]["validation_mean"],
                "recall_mean": metrics["recall"]["validation_mean"],
                "accuracy_mean": metrics["accuracy"]["validation_mean"],
                "balanced_accuracy_mean": metrics["balanced_accuracy"]["validation_mean"],
                "fit_time_mean": summary["fit_time_mean"],
                "score_time_mean": summary["score_time_mean"],
                "train_roc_auc_mean": train_auc,
                "validation_roc_auc_mean": validation_auc,
                "roc_auc_generalization_gap": float(train_auc) - float(validation_auc),
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def calculate_baseline_improvements(
    comparison: pd.DataFrame,
    *,
    logistic_id: str = LOGISTIC_EXPERIMENT_ID,
    reference_id: str = "M01-DUMMY-001",
) -> dict[str, float]:
    """Calculate factual Logistic improvements over the majority reference."""
    indexed = comparison.set_index("experiment_id")
    if logistic_id not in indexed.index or reference_id not in indexed.index:
        raise ValueError("Comparison must contain Logistic and reference experiment IDs.")
    logistic = indexed.loc[logistic_id]
    reference = indexed.loc[reference_id]
    reference_ap = float(reference["average_precision_mean"])
    ap_absolute = float(logistic["average_precision_mean"] - reference_ap)
    return {
        "roc_auc_absolute_improvement": float(
            logistic["roc_auc_mean"] - reference["roc_auc_mean"]
        ),
        "average_precision_absolute_improvement": ap_absolute,
        "average_precision_relative_improvement": ap_absolute / reference_ap,
        "balanced_accuracy_difference": float(
            logistic["balanced_accuracy_mean"] - reference["balanced_accuracy_mean"]
        ),
        "roc_auc_generalization_gap": float(logistic["roc_auc_generalization_gap"]),
        "logistic_to_dummy_fit_time_ratio": float(
            logistic["fit_time_mean"] / reference["fit_time_mean"]
        ),
        "logistic_roc_auc_std": float(logistic["roc_auc_std"]),
    }


def _project_path(path: str | Path) -> Path:
    root = get_project_root().resolve()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Logistic baseline outputs must remain inside the project.") from exc
    return resolved


def _has_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_has_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_absolute_path(item) for item in value)
    return False


def save_logistic_comparison(
    comparison: pd.DataFrame,
    *,
    csv_path: str | Path = "reports/tables/logistic_baseline_comparison.csv",
    json_path: str | Path = "reports/tables/logistic_baseline_comparison.json",
) -> tuple[Path, Path]:
    """Save comparison CSV and strict JSON without overwriting."""
    missing = set(COMPARISON_COLUMNS) - set(comparison.columns)
    if missing:
        raise ValueError(f"Comparison lacks required columns: {sorted(missing)}.")
    csv_file, json_file = _project_path(csv_path), _project_path(json_path)
    if csv_file.exists() or json_file.exists():
        raise FileExistsError("Logistic baseline comparison already exists.")
    records = comparison.loc[:, COMPARISON_COLUMNS].to_dict(orient="records")
    if _has_absolute_path(records):
        raise ValueError("Comparison content must not contain absolute paths.")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    comparison.loc[:, COMPARISON_COLUMNS].to_csv(csv_file, index=False)
    json_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_file, json_file


def save_logistic_vs_dummy_figure(
    comparison: pd.DataFrame,
    path: str | Path = "reports/figures/logistic_vs_dummy_metrics.pdf",
) -> Path:
    """Plot comparable Dummy and Logistic validation metrics from zero to one."""
    output = _project_path(path)
    if output.exists():
        raise FileExistsError(f"Figure already exists: {output.name}.")
    selected_ids = ["M01-DUMMY-001", "M01-DUMMY-003", "M01-DUMMY-004", LOGISTIC_EXPERIMENT_ID]
    labels = ["most_frequent", "stratified", "uniform", "Logistic L2"]
    indexed = comparison.set_index("experiment_id").loc[selected_ids]
    plot = indexed[
        ["roc_auc_mean", "average_precision_mean", "accuracy_mean", "balanced_accuracy_mean"]
    ].copy()
    plot.index = labels
    plot.columns = ["ROC-AUC", "Average Precision", "Accuracy", "Balanced Accuracy"]
    figure, axis = plt.subplots(figsize=(10, 5.8))
    plot.plot(kind="bar", ax=axis, width=0.78)
    axis.set_title("Logistic Regression L2 vs Naive Baselines")
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean validation score")
    axis.set_ylim(0, 1)
    axis.legend(title="Metric", frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output


def save_logistic_cv_figure(
    fold_results: pd.DataFrame,
    path: str | Path = "reports/figures/logistic_cv_scores.pdf",
) -> Path:
    """Plot Logistic validation ROC-AUC and Average Precision for every fold."""
    output = _project_path(path)
    if output.exists():
        raise FileExistsError(f"Figure already exists: {output.name}.")
    required = {"fold", "validation_roc_auc", "validation_average_precision"}
    if not required <= set(fold_results.columns):
        raise ValueError(f"Fold results lack required columns: {sorted(required)}.")
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(fold_results["fold"], fold_results["validation_roc_auc"], marker="o", label="ROC-AUC")
    axis.plot(
        fold_results["fold"], fold_results["validation_average_precision"],
        marker="s", label="Average Precision",
    )
    axis.set_title("Logistic Regression L2 — Validation Scores by Fold")
    axis.set_xlabel("Cross-validation fold")
    axis.set_ylabel("Validation score")
    axis.set_xticks(fold_results["fold"])
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output

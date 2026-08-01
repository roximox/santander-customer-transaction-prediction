"""Definitions and reporting helpers for the four scientific Dummy baselines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from src.config import get_project_root
from src.modeling import create_dummy_classifier

DUMMY_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {"experiment_id": "M01-DUMMY-001", "strategy": "most_frequent"},
    {"experiment_id": "M01-DUMMY-002", "strategy": "prior"},
    {"experiment_id": "M01-DUMMY-003", "strategy": "stratified"},
    {"experiment_id": "M01-DUMMY-004", "strategy": "uniform"},
)

COMPARISON_COLUMNS = (
    "experiment_id", "strategy", "model_name", "primary_metric",
    "roc_auc_mean", "roc_auc_std", "average_precision_mean",
    "average_precision_std", "f1_mean", "precision_mean", "recall_mean",
    "accuracy_mean", "balanced_accuracy_mean", "fit_time_mean",
    "score_time_mean",
)


def build_dummy_classifiers(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Construct the exact four baseline strategies with deterministic randomness."""
    return {
        "most_frequent": create_dummy_classifier(strategy="most_frequent"),
        "prior": create_dummy_classifier(strategy="prior"),
        "stratified": create_dummy_classifier(
            strategy="stratified", random_state=random_state
        ),
        "uniform": create_dummy_classifier(
            strategy="uniform", random_state=random_state
        ),
    }


def build_comparison_table(
    summaries: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a compact comparison from evaluated experiment summaries."""
    by_id = {item["experiment_id"]: item for item in DUMMY_EXPERIMENTS}
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        experiment_id = str(summary["experiment_id"])
        if experiment_id not in by_id:
            raise ValueError(f"Unexpected Dummy experiment ID {experiment_id!r}.")
        metrics = summary["metrics"]
        rows.append(
            {
                "experiment_id": experiment_id,
                "strategy": by_id[experiment_id]["strategy"],
                "model_name": summary["model_name"],
                "primary_metric": summary["primary_metric"],
                "roc_auc_mean": metrics["roc_auc"]["validation_mean"],
                "roc_auc_std": metrics["roc_auc"]["validation_std"],
                "average_precision_mean": metrics["average_precision"]["validation_mean"],
                "average_precision_std": metrics["average_precision"]["validation_std"],
                "f1_mean": metrics["f1"]["validation_mean"],
                "precision_mean": metrics["precision"]["validation_mean"],
                "recall_mean": metrics["recall"]["validation_mean"],
                "accuracy_mean": metrics["accuracy"]["validation_mean"],
                "balanced_accuracy_mean": metrics["balanced_accuracy"]["validation_mean"],
                "fit_time_mean": summary["fit_time_mean"],
                "score_time_mean": summary["score_time_mean"],
            }
        )
    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    return table.sort_values("experiment_id", ignore_index=True)


def _resolve_project_path(path: str | Path) -> Path:
    root = get_project_root().resolve()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Baseline output paths must remain inside the project.") from exc
    return resolved


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def save_comparison_table(
    table: pd.DataFrame,
    *,
    csv_path: str | Path = "reports/tables/dummy_baseline_comparison.csv",
    json_path: str | Path = "reports/tables/dummy_baseline_comparison.json",
) -> tuple[Path, Path]:
    """Save CSV and strict JSON comparison files without overwriting."""
    missing = set(COMPARISON_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Comparison table lacks columns: {sorted(missing)}.")
    csv_file = _resolve_project_path(csv_path)
    json_file = _resolve_project_path(json_path)
    if csv_file.exists() or json_file.exists():
        raise FileExistsError("Dummy baseline comparison output already exists.")
    records = table.loc[:, COMPARISON_COLUMNS].to_dict(orient="records")
    if _contains_absolute_path(records):
        raise ValueError("Comparison output must not contain absolute paths.")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, COMPARISON_COLUMNS].to_csv(csv_file, index=False)
    json_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_file, json_file


def save_metrics_figure(
    table: pd.DataFrame,
    path: str | Path = "reports/figures/dummy_baseline_metrics.pdf",
) -> Path:
    """Plot four comparable validation metrics on an honest zero-to-one scale."""
    output = _resolve_project_path(path)
    if output.exists():
        raise FileExistsError(f"Dummy baseline figure already exists: {output.name}.")
    plot = table.set_index("strategy")[
        ["roc_auc_mean", "average_precision_mean", "accuracy_mean", "balanced_accuracy_mean"]
    ].rename(
        columns={
            "roc_auc_mean": "ROC-AUC",
            "average_precision_mean": "Average Precision",
            "accuracy_mean": "Accuracy",
            "balanced_accuracy_mean": "Balanced Accuracy",
        }
    )
    figure, axis = plt.subplots(figsize=(10, 5.8))
    plot.plot(kind="bar", ax=axis, width=0.78)
    axis.set_title("DummyClassifier Baselines — Cross-Validation Metrics")
    axis.set_xlabel("Strategy")
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


def factual_interpretation(table: pd.DataFrame, positive_prevalence: float) -> str:
    """Describe calculated baseline facts without claiming discriminative ability."""
    indexed = table.set_index("strategy")
    majority = indexed.loc["most_frequent"]
    random_rows = indexed.loc[["stratified", "uniform"]]
    return (
        "The most_frequent and prior strategies predict the majority class. "
        f"Their accuracy is {majority['accuracy_mean']:.4f}, while balanced "
        f"accuracy is {majority['balanced_accuracy_mean']:.4f} and ROC-AUC is "
        f"{majority['roc_auc_mean']:.4f}. Average Precision for the naive "
        f"majority baseline is {majority['average_precision_mean']:.4f}, compared "
        f"with positive prevalence {positive_prevalence:.4f}; its F1 and recall "
        f"are {majority['f1_mean']:.4f} and {majority['recall_mean']:.4f}. "
        "The stratified and uniform strategies have different threshold-dependent "
        f"metrics (accuracy range {random_rows['accuracy_mean'].min():.4f}–"
        f"{random_rows['accuracy_mean'].max():.4f}) but ROC-AUC remains near 0.5, "
        "so these Dummy classifiers provide no learned discriminative signal."
    )

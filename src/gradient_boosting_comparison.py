"""Result-consolidation helpers for Member 4 HistGradientBoosting comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import get_project_root
from src.visualization import save_figure


HIST_GRADIENT_BOOSTING_COMPARISON_ID = "M04-HGB-COMP-001"
HIST_GRADIENT_BOOSTING_BASELINE_ID = "M04-HGB-001"
HIST_GRADIENT_BOOSTING_TUNED_SOURCE_ID = "M04-HGB-OOF-001"
DEFAULT_BASELINE_SUMMARY_PATH = "reports/experiments/M04-HGB-001_summary.json"
DEFAULT_TUNED_OOF_METRICS_PATH = "reports/tables/M04-HGB-OOF-001_metrics.json"
DEFAULT_TUNED_SEARCH_SUMMARY_PATH = "reports/searches/M04-HGB-SEARCH-001_summary.json"
DEFAULT_COMPARISON_TABLE_PATH = "reports/tables/M04-HGB-model-comparison.csv"
DEFAULT_COMPARISON_SUMMARY_PATH = "reports/tables/M04-HGB-model-comparison.json"
DEFAULT_COMPARISON_FIGURE_PATH = "reports/figures/M04-HGB-model-comparison.pdf"

_METRICS = (
    ("roc_auc", "ROC-AUC"),
    ("average_precision", "Average Precision"),
    ("f1", "F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Accuracy"),
)
_COMPARISON_COLUMNS = ("metric", "baseline_value", "tuned_value", "absolute_change")


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve a path inside the project root."""
    project_root = get_project_root().resolve()
    candidate = Path(path)
    resolved_path = (candidate if candidate.is_absolute() else project_root / candidate).resolve()
    try:
        resolved_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("path must resolve inside the project root.") from error
    return resolved_path


def _load_json_object(path: str | Path) -> dict[str, Any]:
    """Load an existing project-local JSON object."""
    resolved_path = _resolve_project_path(path)
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required comparison input does not exist: {resolved_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Comparison input is not valid JSON: {resolved_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Comparison input must be a JSON object: {resolved_path}")
    return payload


def _validate_metric_values(metrics: Mapping[str, Any], *, source: str) -> dict[str, float]:
    """Validate and return the seven finite comparison metrics."""
    values: dict[str, float] = {}
    for metric_key, _ in _METRICS:
        value = metrics.get(metric_key)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{source} lacks a numeric {metric_key!r} metric.")
        value_as_float = float(value)
        if not np.isfinite(value_as_float):
            raise ValueError(f"{source} metric {metric_key!r} must be finite.")
        values[metric_key] = value_as_float
    return values


def load_hist_gradient_boosting_baseline_metrics(
    summary_path: str | Path = DEFAULT_BASELINE_SUMMARY_PATH,
) -> dict[str, float]:
    """Load baseline five-fold CV validation means from M04-HGB-001."""
    summary = _load_json_object(summary_path)
    if summary.get("experiment_id") != HIST_GRADIENT_BOOSTING_BASELINE_ID:
        raise ValueError("Baseline summary does not correspond to M04-HGB-001.")
    raw_metrics = summary.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("Baseline summary must contain a metrics mapping.")
    validation_means: dict[str, Any] = {}
    for metric_key, _ in _METRICS:
        metric_result = raw_metrics.get(metric_key)
        if not isinstance(metric_result, Mapping) or "validation_mean" not in metric_result:
            raise ValueError(
                f"Baseline summary lacks validation_mean for {metric_key!r}."
            )
        validation_means[metric_key] = metric_result["validation_mean"]
    return _validate_metric_values(validation_means, source="Baseline summary")


def load_hist_gradient_boosting_tuned_oof_metrics(
    metrics_path: str | Path = DEFAULT_TUNED_OOF_METRICS_PATH,
) -> dict[str, float]:
    """Load aggregate tuned-HGB OOF metrics from M04-HGB-OOF-001."""
    return _validate_metric_values(
        _load_json_object(metrics_path),
        source="Tuned OOF metrics",
    )


def load_hist_gradient_boosting_tuned_parameters(
    summary_path: str | Path = DEFAULT_TUNED_SEARCH_SUMMARY_PATH,
) -> dict[str, Any]:
    """Load selected tuned HGB parameters from M04-HGB-SEARCH-001."""
    summary = _load_json_object(summary_path)
    if summary.get("search_id") != "M04-HGB-SEARCH-001":
        raise ValueError("Search summary does not correspond to M04-HGB-SEARCH-001.")
    parameters = summary.get("best_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Search summary must contain a best_parameters mapping.")
    return dict(parameters)


def build_hist_gradient_boosting_comparison(
    baseline_metrics: Mapping[str, Any],
    tuned_metrics: Mapping[str, Any],
) -> pd.DataFrame:
    """Build a baseline-versus-tuned, training-only HGB comparison DataFrame."""
    baseline = _validate_metric_values(baseline_metrics, source="Baseline metrics")
    tuned = _validate_metric_values(tuned_metrics, source="Tuned OOF metrics")
    return pd.DataFrame(
        [
            {
                "metric": metric_label,
                "baseline_value": baseline[metric_key],
                "tuned_value": tuned[metric_key],
                "absolute_change": tuned[metric_key] - baseline[metric_key],
            }
            for metric_key, metric_label in _METRICS
        ],
        columns=_COMPARISON_COLUMNS,
    )


def _validate_comparison(comparison: pd.DataFrame) -> None:
    """Validate the required comparison-table schema."""
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame.")
    missing_columns = sorted(set(_COMPARISON_COLUMNS) - set(comparison.columns))
    if missing_columns:
        raise ValueError(f"comparison lacks required columns: {missing_columns}.")
    if len(comparison) != len(_METRICS):
        raise ValueError("comparison must contain exactly seven metric rows.")


def _resolve_new_output_path(output_path: str | Path) -> Path:
    """Resolve a new output under the project root without overwriting it."""
    resolved_path = _resolve_project_path(output_path)
    if resolved_path.exists():
        raise FileExistsError(f"Comparison output already exists: {resolved_path}")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def save_hist_gradient_boosting_comparison_table(
    comparison: pd.DataFrame,
    output_path: str | Path = DEFAULT_COMPARISON_TABLE_PATH,
) -> Path:
    """Save the comparison DataFrame as an UTF-8 CSV without its pandas index."""
    _validate_comparison(comparison)
    resolved_path = _resolve_new_output_path(output_path)
    comparison.to_csv(resolved_path, index=False, encoding="utf-8")
    return resolved_path


def save_hist_gradient_boosting_comparison_summary(
    comparison: pd.DataFrame,
    tuned_parameters: Mapping[str, Any],
    output_path: str | Path = DEFAULT_COMPARISON_SUMMARY_PATH,
) -> Path:
    """Save strict JSON describing the training-only HGB comparison."""
    _validate_comparison(comparison)
    if not isinstance(tuned_parameters, Mapping):
        raise TypeError("tuned_parameters must be a mapping.")
    metric_records = [
        {
            "metric": str(row.metric),
            "baseline_value": float(row.baseline_value),
            "tuned_value": float(row.tuned_value),
            "absolute_change": float(row.absolute_change),
        }
        for row in comparison.itertuples(index=False)
    ]
    payload = {
        "comparison_id": HIST_GRADIENT_BOOSTING_COMPARISON_ID,
        "baseline_id": HIST_GRADIENT_BOOSTING_BASELINE_ID,
        "tuned_source": HIST_GRADIENT_BOOSTING_TUNED_SOURCE_ID,
        "baseline_source_type": "five_fold_validation_mean",
        "tuned_source_type": "aggregated_oof_predictions",
        "final_test_used": False,
        "tuned_parameters": dict(tuned_parameters),
        "metrics": metric_records,
    }
    encoded_payload = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    resolved_path = _resolve_new_output_path(output_path)
    resolved_path.write_text(encoded_payload + "\n", encoding="utf-8")
    return resolved_path


def save_hist_gradient_boosting_comparison_figure(
    comparison: pd.DataFrame,
    output_path: str | Path = DEFAULT_COMPARISON_FIGURE_PATH,
) -> Path:
    """Save a grouped-bar figure comparing all seven baseline and tuned metrics."""
    _validate_comparison(comparison)
    positions = np.arange(len(comparison))
    width = 0.38
    figure, axis = plt.subplots(figsize=(13, 6))
    try:
        axis.bar(
            positions - width / 2,
            comparison["baseline_value"],
            width,
            label="Baseline HGB",
        )
        axis.bar(
            positions + width / 2,
            comparison["tuned_value"],
            width,
            label="Tuned HGB",
        )
        axis.set(
            title="Member 4 HistGradientBoosting: Baseline vs Tuned OOF",
            xlabel="Metric",
            ylabel="Score",
            xticks=positions,
            xticklabels=comparison["metric"],
            ylim=(0, 1),
        )
        axis.grid(axis="y", alpha=0.25, linestyle="--")
        axis.legend()
        figure.tight_layout()
        return save_figure(figure, _resolve_new_output_path(output_path))
    finally:
        plt.close(figure)

"""Reusable, training-only hyperparameter-search utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.utils.validation import check_is_fitted

from src.config import get_project_root, load_config
from src.evaluation import get_primary_metric_name, get_scoring_metrics
from src.validation import create_stratified_cv

METRIC_NAMES = tuple(get_scoring_metrics())


def _serializable(value: Any) -> Any:
    """Convert scientific Python objects to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_serializable(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_serializable(item) for item in value]
    return repr(value)


def create_logistic_parameter_grid() -> dict[str, list[Any]]:
    """Return a fresh copy of the predeclared 20-candidate logistic grid."""
    return {
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__class_weight": [None, "balanced"],
    }


def create_grid_search(
    estimator: BaseEstimator,
    *,
    parameter_grid: Mapping[str, Sequence[Any]],
    scoring: Mapping[str, Any] | None = None,
    refit: str | None = None,
    cv: BaseCrossValidator | None = None,
    n_jobs: int | None = None,
    return_train_score: bool = True,
    verbose: int = 1,
    error_score: str | float = "raise",
) -> GridSearchCV:
    """Create an unfitted GridSearchCV; this API accepts training data nowhere."""
    resolved_scoring = dict(get_scoring_metrics() if scoring is None else scoring)
    resolved_refit = get_primary_metric_name() if refit is None else refit
    if resolved_refit not in resolved_scoring:
        raise ValueError(
            f"refit metric {resolved_refit!r} is absent from scoring: "
            f"{sorted(resolved_scoring)}."
        )
    resolved_cv = create_stratified_cv() if cv is None else cv
    if n_jobs is None:
        configured = load_config().get("experiments", {}).get("default_n_jobs", -1)
        n_jobs = int(configured)
    return GridSearchCV(
        estimator=estimator,
        param_grid=dict(parameter_grid),
        scoring=resolved_scoring,
        refit=resolved_refit,
        cv=resolved_cv,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        verbose=verbose,
        error_score=error_score,
    )


def grid_search_results_to_dataframe(search: GridSearchCV) -> pd.DataFrame:
    """Return readable, serializable candidate-level results from a fitted search."""
    check_is_fitted(search, attributes=["cv_results_", "best_index_"])
    raw = pd.DataFrame(search.cv_results_)
    results = pd.DataFrame(index=raw.index)
    results["candidate_id"] = [f"candidate_{index + 1:03d}" for index in raw.index]
    for name in ("penalty", "C", "class_weight"):
        column = f"param_classifier__{name}"
        if column in raw:
            results[name] = pd.Series(raw[column].tolist(), dtype=object)

    for metric in search.scoring:
        rank = f"rank_test_{metric}"
        if rank in raw:
            results[f"rank_{metric}"] = raw[rank].astype(int)
            if metric == "roc_auc":
                results[rank] = raw[rank].astype(int)
        for source, target in (
            (f"mean_test_{metric}", f"validation_{metric}_mean"),
            (f"std_test_{metric}", f"validation_{metric}_std"),
            (f"mean_train_{metric}", f"train_{metric}_mean"),
            (f"std_train_{metric}", f"train_{metric}_std"),
        ):
            if source in raw:
                results[target] = raw[source].astype(float)
        train = f"train_{metric}_mean"
        validation = f"validation_{metric}_mean"
        if train in results and validation in results:
            results[f"{metric}_generalization_gap"] = results[train] - results[validation]

    for timing in ("mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"):
        results[timing] = raw[timing].astype(float)
    for split in range(search.n_splits_):
        for metric in ("roc_auc", "average_precision"):
            source = f"split{split}_test_{metric}"
            if source in raw:
                results[f"split{split}_validation_{metric}"] = raw[source].astype(float)

    return results.map(_serializable)


def summarize_grid_search(
    search: GridSearchCV,
    results: pd.DataFrame,
    *,
    search_id: str,
    member: str,
    branch: str,
) -> dict[str, Any]:
    """Build a strict-JSON candidate-selection summary without storing a model."""
    check_is_fitted(search, attributes=["cv_results_", "best_index_", "best_score_"])
    if len(results) != len(search.cv_results_["params"]):
        raise ValueError("results must contain exactly one row per search candidate.")
    best = results.iloc[int(search.best_index_)]
    metrics: dict[str, Any] = {}
    for metric in search.scoring:
        metrics[metric] = {
            "validation_mean": best.get(f"validation_{metric}_mean"),
            "validation_std": best.get(f"validation_{metric}_std"),
            "train_mean": best.get(f"train_{metric}_mean"),
            "generalization_gap": best.get(f"{metric}_generalization_gap"),
        }
    estimator = search.estimator
    steps = [name for name, _ in getattr(estimator, "steps", [])]
    summary = {
        "search_id": search_id,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "member": member,
        "branch": branch,
        "estimator_class": estimator.__class__.__name__,
        "pipeline_steps": steps,
        "n_candidates": len(results),
        "n_splits": int(search.n_splits_),
        "total_fits": len(results) * int(search.n_splits_),
        "primary_metric": str(search.refit),
        "best_index": int(search.best_index_),
        "best_score": float(search.best_score_),
        "best_parameters": search.best_params_,
        "best_candidate_id": best["candidate_id"],
        "target_metric_summaries": metrics,
        "search_space": search.param_grid,
        "status": "completed",
        "final_test_used": False,
    }
    return _serializable(summary)


def save_grid_search_results(
    results: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save candidate CSV and strict JSON using relative, overwrite-safe paths."""
    search_id = summary.get("search_id")
    if not isinstance(search_id, str) or not search_id:
        raise ValueError("summary must contain a non-empty search_id.")
    relative = Path("reports/searches") if output_dir is None else Path(output_dir)
    if relative.is_absolute() or PureWindowsPath(str(output_dir or "")).is_absolute():
        raise ValueError("output_dir must be relative to the project root.")
    directory = get_project_root() / relative
    csv_path = directory / f"{search_id}_candidates.csv"
    json_path = directory / f"{search_id}_summary.json"
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing:
        raise FileExistsError("Grid-search artifact already exists: " + ", ".join(path.name for path in existing))
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False, encoding="utf-8")
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(_serializable(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return csv_path, json_path

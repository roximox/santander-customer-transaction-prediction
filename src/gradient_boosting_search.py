"""Reusable, training-only search setup for Member 4 HistGradientBoosting."""

from __future__ import annotations

from datetime import datetime, timezone
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted

from src.evaluation import get_primary_metric_name, get_scoring_metrics
from src.gradient_boosting import create_hist_gradient_boosting_baseline
from src.validation import create_stratified_cv

HIST_GRADIENT_BOOSTING_SEARCH_ID = "M04-HGB-SEARCH-001"


def create_hist_gradient_boosting_search_space() -> dict[str, list[int | float]]:
    """Return a fresh, bounded hyperparameter space for HistGradientBoosting."""
    return {
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "max_iter": [150, 300, 500, 700],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    }


def create_hist_gradient_boosting_randomized_search(
    *,
    n_iter: int = 20,
    cv=None,
    n_jobs: int | None = None,
) -> RandomizedSearchCV:
    """Create an unfitted deterministic HGB RandomizedSearchCV instance."""
    if isinstance(n_iter, bool) or not isinstance(n_iter, Integral) or n_iter < 1:
        raise ValueError(f"n_iter must be an integer >= 1; got {n_iter!r}.")
    if n_jobs is not None and (isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral)):
        raise ValueError(f"n_jobs must be an integer or None; got {n_jobs!r}.")

    estimator: HistGradientBoostingClassifier = create_hist_gradient_boosting_baseline()
    resolved_cv = create_stratified_cv() if cv is None else cv

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=create_hist_gradient_boosting_search_space(),
        n_iter=int(n_iter),
        scoring=dict(get_scoring_metrics()),
        refit=get_primary_metric_name(),
        cv=resolved_cv,
        random_state=42,
        n_jobs=None if n_jobs is None else int(n_jobs),
        return_train_score=True,
        error_score="raise",
    )


def _serializable(value: Any) -> Any:
    """Convert NumPy values to strict JSON-compatible Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_serializable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    return repr(value)


def hist_gradient_boosting_search_results_to_dataframe(
    search: RandomizedSearchCV,
) -> pd.DataFrame:
    """Return serializable, candidate-level results from a fitted HGB search."""
    if not isinstance(search, RandomizedSearchCV):
        raise TypeError("search must be a RandomizedSearchCV instance.")
    check_is_fitted(search, attributes=["cv_results_", "best_index_"])
    raw = pd.DataFrame(search.cv_results_)
    results = pd.DataFrame(index=raw.index)
    results["candidate_id"] = [f"candidate_{index + 1:03d}" for index in raw.index]

    for parameter in (
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    ):
        column = f"param_{parameter}"
        if column in raw:
            results[parameter] = pd.Series(raw[column].tolist(), dtype=object)

    for metric in search.scoring:
        rank = f"rank_test_{metric}"
        if rank in raw:
            results[f"rank_{metric}"] = raw[rank].astype(int)
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
    for fold in range(search.n_splits_):
        for metric in ("roc_auc", "average_precision"):
            source = f"split{fold}_test_{metric}"
            if source in raw:
                results[f"fold_{fold + 1}_{metric}"] = raw[source].astype(float)

    return results.map(_serializable)


def summarize_hist_gradient_boosting_search(
    search: RandomizedSearchCV,
    results: pd.DataFrame,
    *,
    search_id: str = HIST_GRADIENT_BOOSTING_SEARCH_ID,
    member: str,
    branch: str,
) -> dict[str, Any]:
    """Build a strict-JSON summary for a fitted HGB search without its model."""
    if not isinstance(search, RandomizedSearchCV):
        raise TypeError("search must be a RandomizedSearchCV instance.")
    check_is_fitted(search, attributes=["cv_results_", "best_index_", "best_score_"])
    if len(results) != len(search.cv_results_["params"]):
        raise ValueError("results must contain exactly one row per search candidate.")
    if not isinstance(search_id, str) or not search_id:
        raise ValueError("search_id must be a non-empty string.")

    best = results.iloc[int(search.best_index_)]
    metric_summaries: dict[str, dict[str, Any]] = {}
    for metric in search.scoring:
        metric_summaries[metric] = {
            "validation_mean": best.get(f"validation_{metric}_mean"),
            "validation_std": best.get(f"validation_{metric}_std"),
            "train_mean": best.get(f"train_{metric}_mean"),
            "train_std": best.get(f"train_{metric}_std"),
            "generalization_gap": best.get(f"{metric}_generalization_gap"),
        }

    return _serializable(
        {
            "search_id": search_id,
            "member": member,
            "branch": branch,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "n_candidates": len(results),
            "n_splits": int(search.n_splits_),
            "total_fits": len(results) * int(search.n_splits_),
            "primary_metric": str(search.refit),
            "best_index": int(search.best_index_),
            "best_score": float(search.best_score_),
            "best_parameters": search.best_params_,
            "best_candidate_id": best["candidate_id"],
            "target_metric_summaries": metric_summaries,
            "search_space": search.param_distributions,
            "final_test_used": False,
            "status": "completed",
        }
    )

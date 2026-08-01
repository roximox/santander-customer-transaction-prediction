"""Lightweight orchestration for evaluating, saving, and registering experiments.

This module composes the shared utilities in :mod:`src.evaluation`. It does not
implement cross-validation, metrics, preprocessing, or final-test evaluation.
Callers must provide a ready-to-evaluate estimator or scikit-learn Pipeline and
must select an explicit, unique experiment identifier.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

from src.config import get_project_root
from src.evaluation import (
    append_experiment_registry,
    evaluate_model_cv,
    save_experiment_results,
)

_SAFE_EXPERIMENT_ID = re.compile(r"^[A-Z0-9][A-Z0-9_-]*$")


def validate_experiment_metadata(
    *, experiment_id: str, model_name: str, member: str, branch: str
) -> None:
    """Validate required experiment metadata and a path-safe explicit ID."""
    values = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "member": member,
        "branch": branch,
    }
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")
        if value != value.strip():
            raise ValueError(f"{name} must not have leading or trailing spaces.")
    if not _SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id may contain only uppercase letters, digits, hyphens, "
            "and underscores, and must not contain spaces or path components."
        )


def _project_relative_path(path: str | Path) -> str:
    """Return a stable POSIX path relative to the project root."""
    root = get_project_root().resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Experiment path must be inside project root {root}.") from exc


def run_experiment(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    experiment_id: str,
    model_name: str,
    member: str,
    branch: str,
    cv: StratifiedKFold | None = None,
    n_jobs: int | None = None,
    return_train_score: bool = True,
    save_results: bool = False,
    register_experiment: bool = False,
    output_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate training data and optionally persist and register the result."""
    validate_experiment_metadata(
        experiment_id=experiment_id,
        model_name=model_name,
        member=member,
        branch=branch,
    )
    if register_experiment and not save_results:
        raise ValueError("register_experiment=True requires save_results=True.")

    fold_results, summary = evaluate_model_cv(
        estimator,
        X,
        y,
        experiment_id=experiment_id,
        model_name=model_name,
        member=member,
        branch=branch,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
    )
    if not save_results:
        summary.update({"saved": False, "registered": False})
        return fold_results, summary

    fold_path, summary_path = save_experiment_results(
        fold_results, summary, output_dir=output_dir
    )
    summary.update(
        {
            "fold_results_file": _project_relative_path(fold_path),
            "summary_file": _project_relative_path(summary_path),
            "saved": True,
            "registered": False,
        }
    )
    if register_experiment:
        try:
            saved_registry = append_experiment_registry(
                summary, registry_path=registry_path
            )
        except Exception as exc:
            raise RuntimeError(
                "Experiment results were saved, but registry update failed; "
                f"the result files remain at {summary['fold_results_file']!r} and "
                f"{summary['summary_file']!r}."
            ) from exc
        summary.update(
            {
                "registry_file": _project_relative_path(saved_registry),
                "registered": True,
            }
        )
    return fold_results, summary


def run_and_save_experiment(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    experiment_id: str,
    model_name: str,
    member: str,
    branch: str,
    cv: StratifiedKFold | None = None,
    n_jobs: int | None = None,
    return_train_score: bool = True,
    output_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    register_experiment: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run :func:`run_experiment` with result persistence enabled."""
    return run_experiment(
        estimator,
        X,
        y,
        experiment_id=experiment_id,
        model_name=model_name,
        member=member,
        branch=branch,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        save_results=True,
        register_experiment=register_experiment,
        output_dir=output_dir,
        registry_path=registry_path,
    )


def build_logbook_metadata(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Extract verified facts for manual Logbook writing without writing files."""
    fields = (
        "experiment_id",
        "model_name",
        "date_utc",
        "member",
        "branch",
        "n_samples",
        "n_features",
        "primary_metric",
        "primary_score_mean",
        "primary_score_std",
        "fit_time_mean",
        "summary_file",
    )
    return {field: summary[field] for field in fields if field in summary}

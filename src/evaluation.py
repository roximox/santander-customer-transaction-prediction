"""Shared, training-only cross-validation and experiment reporting utilities."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import get_project_root, load_config
from src.validation import create_stratified_cv, get_cv_split_fingerprints


def _positive_label(y: pd.Series | np.ndarray) -> Any:
    """Choose a deterministic positive label, supporting Santander string labels."""
    labels = list(pd.unique(y))
    for preferred in ("True", True, 1):
        if preferred in labels:
            return preferred
    return sorted(labels, key=lambda value: (type(value).__name__, repr(value)))[-1]


def _positive_scores(estimator: BaseEstimator, X: pd.DataFrame, positive: Any) -> np.ndarray:
    """Return scores oriented toward the selected positive class."""
    classes = list(getattr(estimator, "classes_", []))
    if positive not in classes:
        raise ValueError(f"Estimator classes do not contain positive label {positive!r}.")
    positive_position = classes.index(positive)
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))[:, positive_position]
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X))
        if scores.ndim == 1:
            return scores if positive_position == 1 else -scores
        return scores[:, positive_position]
    return np.asarray(estimator.predict(X) == positive, dtype=float)


def _roc_auc_scorer(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    positive = _positive_label(y)
    return float(roc_auc_score(np.asarray(y == positive, dtype=int), _positive_scores(estimator, X, positive)))


def _average_precision_scorer(
    estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series
) -> float:
    positive = _positive_label(y)
    return float(
        average_precision_score(
            np.asarray(y == positive, dtype=int),
            _positive_scores(estimator, X, positive),
        )
    )


def _prediction_metric(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    metric: Callable[..., float],
    *,
    uses_positive_label: bool,
) -> float:
    predictions = estimator.predict(X)
    kwargs = (
        {"pos_label": _positive_label(y), "zero_division": 0}
        if uses_positive_label
        else {}
    )
    return float(metric(y, predictions, **kwargs))


def get_scoring_metrics() -> dict[str, str | Callable[..., float]]:
    """Return common metrics with safe binary scorers for numeric or string labels."""
    return {
        "roc_auc": _roc_auc_scorer,
        "average_precision": _average_precision_scorer,
        "f1": lambda estimator, X, y: _prediction_metric(
            estimator, X, y, f1_score, uses_positive_label=True
        ),
        "precision": lambda estimator, X, y: _prediction_metric(
            estimator, X, y, precision_score, uses_positive_label=True
        ),
        "recall": lambda estimator, X, y: _prediction_metric(
            estimator, X, y, recall_score, uses_positive_label=True
        ),
        "accuracy": lambda estimator, X, y: _prediction_metric(
            estimator, X, y, accuracy_score, uses_positive_label=False
        ),
        "balanced_accuracy": lambda estimator, X, y: _prediction_metric(
            estimator, X, y, balanced_accuracy_score, uses_positive_label=False
        ),
    }


def get_primary_metric_name() -> str:
    """Read and validate the centrally configured primary metric name."""
    config = load_config()
    metrics_config = config.get("metrics")
    if not isinstance(metrics_config, Mapping):
        raise ValueError("Configuration must contain a 'metrics' mapping.")
    primary = metrics_config.get("primary")
    if not isinstance(primary, str) or primary not in get_scoring_metrics():
        raise ValueError(
            f"Configured primary metric {primary!r} is not available in shared scoring."
        )
    return primary


def validate_evaluation_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    experiment_id: str,
) -> None:
    """Validate training-only evaluation inputs without applying preprocessing."""
    if X.empty:
        raise ValueError("Evaluation feature frame X must not be empty.")
    if y.empty:
        raise ValueError("Evaluation target y must not be empty.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of rows; got {len(X)} and {len(y)}."
        )
    if not X.index.equals(y.index):
        raise ValueError("X and y indexes must be identical and in the same order.")
    if y.nunique(dropna=True) < 2:
        raise ValueError("Evaluation target y must contain at least two classes.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string.")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in experiment_id):
        raise ValueError("experiment_id may contain only letters, digits, hyphens, and underscores.")
    duplicated = X.columns[X.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Evaluation feature names must be unique; duplicates: {duplicated}.")


def _target_distribution(y: pd.Series) -> dict[Any, float]:
    """Return target proportions using plain scalar values."""
    return {
        key.item() if isinstance(key, np.generic) else key: float(value)
        for key, value in y.value_counts(dropna=False, normalize=True).items()
    }


def _serializable(value: Any) -> Any:
    """Recursively convert common scientific Python values to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_serializable(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_serializable(item) for item in value]
    return repr(value)


def evaluate_model_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    experiment_id: str,
    member: str = "unknown",
    branch: str = "unknown",
    cv: StratifiedKFold | None = None,
    n_jobs: int | None = None,
    return_train_score: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate an estimator using shared training-only stratified CV.

    This function must receive training data only; it has no test-set argument.
    Any learned preprocessing (scaling, imputation, PCA, or feature selection)
    must be included inside a scikit-learn ``Pipeline`` passed as ``estimator``.
    No transformation is fitted globally by this infrastructure.
    """
    validate_evaluation_inputs(
        X, y, model_name=model_name, experiment_id=experiment_id
    )
    if not callable(getattr(estimator, "fit", None)):
        raise ValueError("estimator must implement a callable fit method.")
    resolved_cv = create_stratified_cv() if cv is None else cv
    if not isinstance(resolved_cv, StratifiedKFold):
        raise ValueError("cv must be a StratifiedKFold instance.")
    config = load_config()
    experiments_config = config.get("experiments", {})
    if not isinstance(experiments_config, Mapping):
        raise ValueError("Configuration 'experiments' must be a mapping.")
    resolved_n_jobs = (
        experiments_config.get("default_n_jobs", -1) if n_jobs is None else n_jobs
    )
    if isinstance(resolved_n_jobs, bool) or not isinstance(resolved_n_jobs, int):
        raise ValueError(f"n_jobs must be an integer; got {resolved_n_jobs!r}.")

    scoring = get_scoring_metrics()
    fingerprints = get_cv_split_fingerprints(resolved_cv, X, y)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        raw_results = cross_validate(
            estimator,
            X,
            y,
            cv=resolved_cv,
            scoring=scoring,
            n_jobs=resolved_n_jobs,
            return_train_score=return_train_score,
            error_score="raise",
        )
    convergence_messages = [
        str(item.message)
        for item in caught_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]
    for message in convergence_messages:
        warnings.warn(message, ConvergenceWarning, stacklevel=2)

    fold_data: dict[str, Any] = {
        "experiment_id": [experiment_id] * resolved_cv.n_splits,
        "model_name": [model_name] * resolved_cv.n_splits,
        "fold": list(range(1, resolved_cv.n_splits + 1)),
        "fit_time": raw_results["fit_time"],
        "score_time": raw_results["score_time"],
        "train_size": [item["train_size"] for item in fingerprints],
        "validation_size": [item["validation_size"] for item in fingerprints],
    }
    for metric_name in scoring:
        if return_train_score:
            fold_data[f"train_{metric_name}"] = raw_results[f"train_{metric_name}"]
        fold_data[f"validation_{metric_name}"] = raw_results[f"test_{metric_name}"]
    fold_results = pd.DataFrame(fold_data)

    metric_summary: dict[str, Any] = {}
    for metric_name in scoring:
        validation_values = fold_results[f"validation_{metric_name}"].to_numpy()
        metric_values: dict[str, Any] = {
            "validation_mean": float(validation_values.mean()),
            "validation_std": float(validation_values.std(ddof=0)),
        }
        if return_train_score:
            train_values = fold_results[f"train_{metric_name}"].to_numpy()
            metric_values.update(
                {
                    "train_mean": float(train_values.mean()),
                    "train_std": float(train_values.std(ddof=0)),
                }
            )
        metric_summary[metric_name] = metric_values

    primary_metric = get_primary_metric_name()
    fit_times = fold_results["fit_time"].to_numpy()
    score_times = fold_results["score_time"].to_numpy()
    summary = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "member": member,
        "branch": branch,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_strategy": type(resolved_cv).__name__,
        "n_splits": int(resolved_cv.n_splits),
        "shuffle": bool(resolved_cv.shuffle),
        "random_state": (
            int(resolved_cv.random_state)
            if resolved_cv.random_state is not None
            else None
        ),
        "primary_metric": primary_metric,
        "primary_score_mean": metric_summary[primary_metric]["validation_mean"],
        "primary_score_std": metric_summary[primary_metric]["validation_std"],
        "metrics": metric_summary,
        "fit_time_mean": float(fit_times.mean()),
        "fit_time_std": float(fit_times.std(ddof=0)),
        "score_time_mean": float(score_times.mean()),
        "score_time_std": float(score_times.std(ddof=0)),
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "estimator_parameters": _serializable(estimator.get_params(deep=True)),
        "target_distribution": _target_distribution(y),
        "cv_fingerprints": fingerprints,
        "return_train_score": bool(return_train_score),
        "convergence_warning_detected": bool(convergence_messages),
        "convergence_warning_messages": convergence_messages,
        "status": "completed",
    }
    return fold_results, _serializable(summary)


def _has_absolute_path(value: Any) -> bool:
    """Detect absolute POSIX or Windows paths nested in report content."""
    if isinstance(value, str):
        return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_has_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_absolute_path(item) for item in value)
    return False


def _default_results_directory() -> Path:
    """Return the configured experiment output directory under the project root."""
    config = load_config()
    experiments = config.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ValueError("Configuration must contain an 'experiments' mapping.")
    configured = experiments.get("results_directory")
    if not isinstance(configured, str) or not configured:
        raise ValueError("experiments.results_directory must be a non-empty string.")
    return get_project_root() / configured


def save_experiment_results(
    fold_results: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save fold CSV and summary JSON without silently overwriting results."""
    experiment_id = summary.get("experiment_id")
    if not isinstance(experiment_id, str):
        raise ValueError("summary must contain a string experiment_id.")
    validate_evaluation_inputs(
        pd.DataFrame({"placeholder": [0, 1]}),
        pd.Series([0, 1]),
        model_name=str(summary.get("model_name", "")),
        experiment_id=experiment_id,
    )
    directory = _default_results_directory() if output_dir is None else Path(output_dir)
    if not directory.is_absolute():
        directory = get_project_root() / directory
    fold_path = directory / f"{experiment_id}_fold_results.csv"
    summary_path = directory / f"{experiment_id}_summary.json"
    if not overwrite and (fold_path.exists() or summary_path.exists()):
        raise FileExistsError(f"Experiment result already exists for {experiment_id!r}.")
    payload = _serializable(summary)
    if _has_absolute_path(payload):
        raise ValueError("Experiment summary must not contain absolute paths.")
    encoded_summary = json.dumps(
        payload, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(fold_path, index=False, encoding="utf-8")
    summary_path.write_text(encoded_summary, encoding="utf-8")
    return fold_path, summary_path


_REGISTRY_COLUMNS = [
    "experiment_id",
    "date_utc",
    "member",
    "branch",
    "model_name",
    "n_samples",
    "n_features",
    "cv_strategy",
    "primary_metric",
    "primary_score_mean",
    "primary_score_std",
    "fit_time_mean",
    "status",
    "summary_file",
]


def append_experiment_registry(
    summary: Mapping[str, Any],
    *,
    registry_path: str | Path | None = None,
) -> Path:
    """Append one unique experiment summary to the global CSV registry."""
    experiment_id = summary.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("summary must contain a non-empty experiment_id.")
    path = (
        _default_results_directory() / "experiment_registry.csv"
        if registry_path is None
        else Path(registry_path)
    )
    if not path.is_absolute():
        path = get_project_root() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        registry = pd.read_csv(path)
        if "experiment_id" not in registry.columns:
            raise ValueError("Existing experiment registry lacks experiment_id column.")
        if experiment_id in registry["experiment_id"].astype(str).tolist():
            raise ValueError(f"Experiment ID {experiment_id!r} already exists in registry.")
    else:
        registry = pd.DataFrame(columns=_REGISTRY_COLUMNS)

    row = {
        column: _serializable(summary.get(column)) for column in _REGISTRY_COLUMNS
    }
    row["summary_file"] = summary.get(
        "summary_file", f"{experiment_id}_summary.json"
    )
    if _has_absolute_path(row):
        raise ValueError("Experiment registry row must not contain absolute paths.")
    if registry.empty:
        updated = pd.DataFrame([row], columns=_REGISTRY_COLUMNS)
    else:
        updated = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(path, index=False, encoding="utf-8")
    return path

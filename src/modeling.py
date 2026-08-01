"""Reproducible, unfitted estimator and Pipeline factories for the project.

Factories in this module only construct scikit-learn objects. They never load
data, call ``fit``, run cross-validation, access the final test partition, or
persist models and experiment results. Defaults are shared starting points, not
claims of optimal performance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config

_DUMMY_STRATEGIES = frozenset(
    {"most_frequent", "prior", "stratified", "uniform", "constant"}
)
_LOGISTIC_PENALTIES = frozenset({"l1", "l2", "elasticnet", None})
_SOLVER_PENALTIES = {
    "lbfgs": frozenset({"l2", None}),
    "liblinear": frozenset({"l1", "l2"}),
    "newton-cg": frozenset({"l2", None}),
    "newton-cholesky": frozenset({"l2", None}),
    "sag": frozenset({"l2", None}),
    "saga": frozenset({"l1", "l2", "elasticnet", None}),
}


def _configured_random_state() -> int:
    """Return the validated project random state."""
    value = load_config().get("project", {}).get("random_state")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("project.random_state must be an integer.")
    return int(value)


def _random_state(value: int | None) -> int:
    if value is None:
        return _configured_random_state()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("random_state must be an integer or None.")
    return int(value)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(value)


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return float(value)


def _validate_class_weight(value: str | dict[Any, float] | None) -> None:
    if value is not None and value != "balanced" and not isinstance(value, dict):
        raise ValueError("class_weight must be None, 'balanced', or a dictionary.")


def _validate_n_jobs(value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral)):
        raise ValueError("n_jobs must be an integer or None.")


def create_dummy_classifier(
    *,
    strategy: str = "most_frequent",
    random_state: int | None = None,
    constant: Any | None = None,
) -> DummyClassifier:
    """Return a new unfitted DummyClassifier with a validated strategy."""
    if strategy not in _DUMMY_STRATEGIES:
        raise ValueError(
            f"Unsupported DummyClassifier strategy {strategy!r}; "
            f"choose one of {sorted(_DUMMY_STRATEGIES)}."
        )
    if strategy == "constant" and constant is None:
        raise ValueError("constant must be provided when strategy='constant'.")
    if strategy != "constant" and constant is not None:
        raise ValueError("constant is only valid when strategy='constant'.")
    seed = _random_state(random_state) if strategy in {"stratified", "uniform"} else None
    return DummyClassifier(strategy=strategy, random_state=seed, constant=constant)


def create_logistic_regression(
    *,
    penalty: str | None = "l2",
    C: float = 1.0,
    class_weight: str | dict[Any, float] | None = None,
    solver: str | None = None,
    max_iter: int = 1000,
    random_state: int | None = None,
    n_jobs: int | None = None,
    l1_ratio: float | None = None,
) -> LogisticRegression:
    """Return a validated, unfitted LogisticRegression starting point."""
    if penalty not in _LOGISTIC_PENALTIES:
        raise ValueError(f"Unsupported LogisticRegression penalty {penalty!r}.")
    resolved_C = _positive_real(C, "C")
    resolved_max_iter = _positive_integer(max_iter, "max_iter")
    _validate_class_weight(class_weight)
    _validate_n_jobs(n_jobs)
    if l1_ratio is not None and (
        isinstance(l1_ratio, bool)
        or not isinstance(l1_ratio, Real)
        or not 0 <= l1_ratio <= 1
    ):
        raise ValueError("l1_ratio must be between 0 and 1.")
    if penalty == "elasticnet" and l1_ratio is None:
        raise ValueError("penalty='elasticnet' requires an explicit l1_ratio.")
    if penalty != "elasticnet" and l1_ratio is not None:
        raise ValueError("l1_ratio is only valid with penalty='elasticnet'.")
    resolved_solver = (
        "saga" if penalty in {"l1", "elasticnet"} else "lbfgs"
    ) if solver is None else solver
    if resolved_solver not in _SOLVER_PENALTIES:
        raise ValueError(f"Unsupported LogisticRegression solver {resolved_solver!r}.")
    if penalty not in _SOLVER_PENALTIES[resolved_solver]:
        raise ValueError(
            f"solver={resolved_solver!r} is incompatible with penalty={penalty!r}."
        )
    return LogisticRegression(
        penalty=penalty,
        C=resolved_C,
        class_weight=class_weight,
        solver=resolved_solver,
        max_iter=resolved_max_iter,
        random_state=_random_state(random_state),
        n_jobs=n_jobs,
        l1_ratio=l1_ratio,
    )


def create_logistic_regression_pipeline(
    *,
    penalty: str | None = "l2",
    C: float = 1.0,
    class_weight: str | dict[Any, float] | None = None,
    solver: str | None = None,
    max_iter: int = 1000,
    random_state: int | None = None,
    n_jobs: int | None = None,
    l1_ratio: float | None = None,
    with_mean: bool = True,
    with_std: bool = True,
) -> Pipeline:
    """Return an unfitted StandardScaler → LogisticRegression Pipeline.

    Keeping scaling inside the Pipeline ensures StandardScaler is learned only
    from each cross-validation training fold. No imputation or feature selection
    is added because the completed data audit found no missing values.
    """
    classifier = create_logistic_regression(
        penalty=penalty, C=C, class_weight=class_weight, solver=solver,
        max_iter=max_iter, random_state=random_state, n_jobs=n_jobs,
        l1_ratio=l1_ratio,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
            ("classifier", classifier),
        ]
    )


def create_random_forest_classifier(
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | int | float | None = "sqrt",
    class_weight: str | dict[Any, float] | None = None,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> RandomForestClassifier:
    """Return an unfitted Random Forest starting point; defaults are not optimized."""
    _validate_class_weight(class_weight)
    _validate_n_jobs(n_jobs)
    return RandomForestClassifier(
        n_estimators=_positive_integer(n_estimators, "n_estimators"),
        max_depth=max_depth,
        min_samples_split=_positive_integer(min_samples_split, "min_samples_split"),
        min_samples_leaf=_positive_integer(min_samples_leaf, "min_samples_leaf"),
        max_features=max_features,
        class_weight=class_weight,
        random_state=_random_state(random_state),
        n_jobs=n_jobs,
    )


def create_extra_trees_classifier(
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | int | float | None = "sqrt",
    class_weight: str | dict[Any, float] | None = None,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> ExtraTreesClassifier:
    """Return an unfitted Extra Trees starting point; defaults are not optimized."""
    _validate_class_weight(class_weight)
    _validate_n_jobs(n_jobs)
    return ExtraTreesClassifier(
        n_estimators=_positive_integer(n_estimators, "n_estimators"),
        max_depth=max_depth,
        min_samples_split=_positive_integer(min_samples_split, "min_samples_split"),
        min_samples_leaf=_positive_integer(min_samples_leaf, "min_samples_leaf"),
        max_features=max_features,
        class_weight=class_weight,
        random_state=_random_state(random_state),
        n_jobs=n_jobs,
    )


def create_hist_gradient_boosting_classifier(
    *,
    learning_rate: float = 0.1,
    max_iter: int = 100,
    max_leaf_nodes: int | None = 31,
    l2_regularization: float = 0.0,
    random_state: int | None = None,
) -> HistGradientBoostingClassifier:
    """Return an unfitted histogram boosting starting point without scaling."""
    if max_leaf_nodes is not None:
        max_leaf_nodes = _positive_integer(max_leaf_nodes, "max_leaf_nodes")
    if (
        isinstance(l2_regularization, bool)
        or not isinstance(l2_regularization, Real)
        or l2_regularization < 0
    ):
        raise ValueError("l2_regularization must be a non-negative number.")
    return HistGradientBoostingClassifier(
        learning_rate=_positive_real(learning_rate, "learning_rate"),
        max_iter=_positive_integer(max_iter, "max_iter"),
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=float(l2_regularization),
        random_state=_random_state(random_state),
    )


def _json_value(value: Any) -> Any:
    """Convert estimator parameters to JSON-safe descriptions."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, BaseEstimator):
        return f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    return repr(value)


def describe_estimator(estimator: BaseEstimator) -> dict[str, Any]:
    """Return a JSON-serializable description containing configuration only."""
    if not isinstance(estimator, BaseEstimator):
        raise ValueError("estimator must be a scikit-learn BaseEstimator.")
    is_pipeline = isinstance(estimator, Pipeline)
    parameters = estimator.get_params(deep=True)
    random_state = parameters.get("random_state")
    if is_pipeline and random_state is None:
        random_state = parameters.get("classifier__random_state")
    return {
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "is_pipeline": is_pipeline,
        "pipeline_steps": [name for name, _ in estimator.steps] if is_pipeline else [],
        "parameters": _json_value(parameters),
        "random_state": _json_value(random_state),
    }

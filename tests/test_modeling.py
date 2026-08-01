"""Offline tests for unfitted shared model factories."""

from __future__ import annotations

import json

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling import (
    create_dummy_classifier,
    create_extra_trees_classifier,
    create_hist_gradient_boosting_classifier,
    create_logistic_regression,
    create_logistic_regression_pipeline,
    create_random_forest_classifier,
    describe_estimator,
)


@pytest.mark.parametrize("strategy", ["most_frequent", "prior", "stratified", "uniform"])
def test_create_dummy_classifier_supports_standard_strategies(strategy: str) -> None:
    estimator = create_dummy_classifier(strategy=strategy)
    assert isinstance(estimator, DummyClassifier)
    assert estimator.strategy == strategy
    assert not hasattr(estimator, "classes_")


def test_dummy_constant_requires_explicit_value_and_invalid_strategy_fails() -> None:
    assert create_dummy_classifier(strategy="constant", constant=1).constant == 1
    with pytest.raises(ValueError, match="constant must be provided"):
        create_dummy_classifier(strategy="constant")
    with pytest.raises(ValueError, match="Unsupported"):
        create_dummy_classifier(strategy="unknown")


def test_dummy_random_state_is_centralized_only_for_random_strategies() -> None:
    assert create_dummy_classifier(strategy="stratified").random_state == 42
    assert create_dummy_classifier(strategy="uniform").random_state == 42
    assert create_dummy_classifier(strategy="prior").random_state is None


def test_create_logistic_regression_defaults_are_explicit_and_unfitted() -> None:
    estimator = create_logistic_regression()
    assert isinstance(estimator, LogisticRegression)
    assert estimator.penalty == "l2"
    assert estimator.C == 1.0
    assert estimator.solver == "lbfgs"
    assert estimator.max_iter == 1000
    assert estimator.class_weight is None
    assert estimator.random_state == 42
    assert not hasattr(estimator, "coef_")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"C": 0}, "C"),
        ({"max_iter": 0}, "max_iter"),
        ({"penalty": "l1", "solver": "lbfgs"}, "incompatible"),
        ({"penalty": "elasticnet"}, "l1_ratio"),
        ({"penalty": "l2", "l1_ratio": 0.5}, "only valid"),
        ({"class_weight": "automatic"}, "class_weight"),
    ],
)
def test_logistic_regression_rejects_common_invalid_parameters(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_logistic_regression(**kwargs)


def test_logistic_options_accept_balancing_l1_and_elasticnet() -> None:
    assert create_logistic_regression(class_weight="balanced").class_weight == "balanced"
    assert create_logistic_regression(penalty="l1").solver == "saga"
    elastic = create_logistic_regression(penalty="elasticnet", l1_ratio=0.25)
    assert elastic.solver == "saga" and elastic.l1_ratio == pytest.approx(0.25)


def test_logistic_pipeline_has_stable_unfitted_steps_and_new_instances() -> None:
    first = create_logistic_regression_pipeline()
    second = create_logistic_regression_pipeline()
    assert isinstance(first, Pipeline)
    assert list(first.named_steps) == ["scaler", "classifier"]
    assert isinstance(first.named_steps["scaler"], StandardScaler)
    assert isinstance(first.named_steps["classifier"], LogisticRegression)
    assert first is not second
    assert first.named_steps["scaler"] is not second.named_steps["scaler"]
    assert not hasattr(first.named_steps["scaler"], "mean_")
    assert not hasattr(first.named_steps["classifier"], "coef_")


def test_tree_factories_return_unfitted_estimators_with_shared_seed() -> None:
    forest = create_random_forest_classifier()
    extra = create_extra_trees_classifier()
    boosting = create_hist_gradient_boosting_classifier()
    assert isinstance(forest, RandomForestClassifier)
    assert isinstance(extra, ExtraTreesClassifier)
    assert isinstance(boosting, HistGradientBoostingClassifier)
    assert forest.n_estimators == 300 and extra.n_estimators == 300
    assert forest.random_state == extra.random_state == boosting.random_state == 42
    assert not hasattr(forest, "estimators_")
    assert not hasattr(extra, "estimators_")
    assert not hasattr(boosting, "classes_")


@pytest.mark.parametrize(
    "factory",
    [create_random_forest_classifier, create_extra_trees_classifier],
)
def test_tree_factories_validate_estimator_count_and_class_weight(factory: object) -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        factory(n_estimators=0)
    with pytest.raises(ValueError, match="class_weight"):
        factory(class_weight="unsupported")


def test_hist_gradient_boosting_validates_core_parameters() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        create_hist_gradient_boosting_classifier(learning_rate=0)
    with pytest.raises(ValueError, match="max_iter"):
        create_hist_gradient_boosting_classifier(max_iter=0)
    with pytest.raises(ValueError, match="l2_regularization"):
        create_hist_gradient_boosting_classifier(l2_regularization=-1)


def test_describe_estimator_is_json_serializable_and_describes_pipeline() -> None:
    pipeline = create_logistic_regression_pipeline(C=0.5)
    description = describe_estimator(pipeline)
    assert description["is_pipeline"] is True
    assert description["pipeline_steps"] == ["scaler", "classifier"]
    assert description["random_state"] == 42
    assert description["parameters"]["classifier__C"] == pytest.approx(0.5)
    assert "sklearn.pipeline.Pipeline" == description["estimator_class"]
    json.dumps(description, allow_nan=False)


def test_factories_do_not_accept_data_or_test_arguments() -> None:
    import inspect

    factories = (
        create_dummy_classifier, create_logistic_regression,
        create_logistic_regression_pipeline, create_random_forest_classifier,
        create_extra_trees_classifier, create_hist_gradient_boosting_classifier,
    )
    for factory in factories:
        parameters = inspect.signature(factory).parameters
        assert not {"X", "y", "X_test", "y_test"} & set(parameters)

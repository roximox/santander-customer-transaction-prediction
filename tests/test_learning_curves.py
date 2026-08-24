"""Offline tests for the reusable training-only learning-curve framework."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.learning_curves import (
    compute_learning_curve,
    create_learning_curve_figures,
    create_stratified_subsample,
    summarize_learning_curve,
    validate_train_size_fractions,
)


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    values, target = make_classification(
        n_samples=240, n_features=8, weights=[.8, .2], random_state=42
    )
    index = pd.Index([f"row-{number}" for number in range(240)], name="row_id")
    return pd.DataFrame(values, index=index), pd.Series(target, index=index)


@pytest.fixture(scope="module")
def folds(sample: tuple[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    X, y = sample
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=500, random_state=42)),
    ])
    return compute_learning_curve(
        estimator, X, y, configuration_id="TEST-LR",
        train_size_fractions=[.25, .5, 1.0],
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
    )


def test_valid_fractions() -> None:
    assert validate_train_size_fractions([.05, .1, 1]) == (.05, .1, 1.0)


@pytest.mark.parametrize("fractions", [[0, .5], [.5, 1.1], [.5, .5], [.5, .25], []])
def test_invalid_fractions_are_rejected(fractions: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_train_size_fractions(fractions)


def test_subsample_is_stratified_reproducible_and_preserves_indices(sample: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = sample
    first_X, first_y = create_stratified_subsample(X, y, .5, random_state=7)
    second_X, second_y = create_stratified_subsample(X, y, .5, random_state=7)
    assert first_X.index.equals(first_y.index)
    assert first_X.index.equals(second_X.index) and first_y.equals(second_y)
    assert set(first_X.index) <= set(X.index)
    assert first_y.mean() == pytest.approx(y.mean(), abs=.01)


def test_fraction_one_and_inputs_are_not_modified(sample: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = sample
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    subset_X, subset_y = create_stratified_subsample(X, y, 1.0)
    assert subset_X.index.equals(X.index) and subset_y.index.equals(y.index)
    assert subset_X is not X and subset_y is not y
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_fold_fraction_count_metrics_and_convergence(folds: pd.DataFrame) -> None:
    assert len(folds) == 3 * 3
    expected = {
        f"{split}_{metric}" for split in ("train", "validation")
        for metric in ("roc_auc", "average_precision", "f1", "precision", "recall", "accuracy", "balanced_accuracy")
    }
    assert expected <= set(folds)
    assert folds["convergence_warning"].map(type).eq(bool).all()
    assert folds["n_iter"].ge(1).all()


def test_summary_mean_std_gap_and_json(folds: pd.DataFrame) -> None:
    summary = summarize_learning_curve(folds)
    assert len(summary) == 3
    first = summary.iloc[0]
    source = folds[folds["train_fraction"] == first["train_fraction"]]
    assert first["validation_roc_auc_mean"] == pytest.approx(source["validation_roc_auc"].mean())
    assert first["validation_roc_auc_std"] == pytest.approx(source["validation_roc_auc"].std(ddof=0))
    assert first["roc_auc_generalization_gap"] == pytest.approx(first["train_roc_auc_mean"] - first["validation_roc_auc_mean"])
    json.dumps(json.loads(summary.to_json(orient="records")), allow_nan=False)


def test_figures_are_creatable(folds: pd.DataFrame, tmp_path: Path) -> None:
    summary = summarize_learning_curve(folds)
    paths = create_learning_curve_figures(summary, tmp_path)
    assert len(paths) == 4
    assert all(path.read_bytes().startswith(b"%PDF") for path in paths)


def test_api_cannot_receive_final_test_data() -> None:
    parameters = set(inspect.signature(compute_learning_curve).parameters)
    assert not parameters & {"X_test", "y_test", "test_data", "final_test"}

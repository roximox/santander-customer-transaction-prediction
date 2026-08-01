"""Offline tests for reusable hyperparameter-search infrastructure."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from src.modeling import create_logistic_regression_pipeline
from src.search import (
    create_grid_search,
    create_logistic_parameter_grid,
    grid_search_results_to_dataframe,
    save_grid_search_results,
    summarize_grid_search,
)


@pytest.fixture
def fitted_search() -> GridSearchCV:
    values, target = make_classification(
        n_samples=90, n_features=6, n_informative=4, random_state=42
    )
    X = pd.DataFrame(values)
    y = pd.Series(target)
    search = create_grid_search(
        create_logistic_regression_pipeline(solver="saga", max_iter=300),
        parameter_grid={
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": [0.1],
            "classifier__class_weight": [None],
        },
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        n_jobs=1,
        verbose=0,
    )
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    search.fit(X, y)
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
    return search


def test_logistic_grid_has_exactly_twenty_fresh_combinations() -> None:
    first = create_logistic_parameter_grid()
    second = create_logistic_parameter_grid()
    assert len(list(ParameterGrid(first))) == 20
    assert set(first) == {
        "classifier__penalty", "classifier__C", "classifier__class_weight"
    }
    assert first is not second
    assert all(first[key] is not second[key] for key in first)


def test_create_grid_search_has_shared_defaults_and_is_unfitted() -> None:
    search = create_grid_search(
        create_logistic_regression_pipeline(solver="saga", max_iter=2000),
        parameter_grid=create_logistic_parameter_grid(),
        n_jobs=1,
        verbose=0,
    )
    assert isinstance(search, GridSearchCV)
    assert set(search.scoring) == {
        "roc_auc", "average_precision", "f1", "precision", "recall",
        "accuracy", "balanced_accuracy",
    }
    assert search.refit == "roc_auc"
    assert isinstance(search.cv, StratifiedKFold) and search.cv.n_splits == 5
    assert search.return_train_score is True
    with pytest.raises(Exception):
        check_is_fitted(search)


def test_refit_must_exist_in_scoring() -> None:
    with pytest.raises(ValueError, match="absent from scoring"):
        create_grid_search(
            create_logistic_regression_pipeline(),
            parameter_grid={"classifier__C": [1.0]},
            scoring={"accuracy": "accuracy"},
            refit="roc_auc",
        )


def test_results_have_rows_ranks_splits_and_correct_gap(fitted_search: GridSearchCV) -> None:
    results = grid_search_results_to_dataframe(fitted_search)
    assert len(results) == 2
    required = {
        "candidate_id", "rank_test_roc_auc", "rank_roc_auc",
        "rank_average_precision", "rank_f1", "rank_balanced_accuracy",
        "split0_validation_roc_auc", "split1_validation_average_precision",
        "validation_roc_auc_mean", "train_roc_auc_mean",
        "roc_auc_generalization_gap",
    }
    assert required <= set(results)
    expected = results["train_roc_auc_mean"] - results["validation_roc_auc_mean"]
    assert results["roc_auc_generalization_gap"].tolist() == pytest.approx(expected)


def test_results_reject_unfitted_search() -> None:
    search = create_grid_search(
        create_logistic_regression_pipeline(),
        parameter_grid={"classifier__C": [1.0]}, verbose=0,
    )
    with pytest.raises(Exception):
        grid_search_results_to_dataframe(search)


def test_summary_is_serializable_and_complete(fitted_search: GridSearchCV) -> None:
    results = grid_search_results_to_dataframe(fitted_search)
    summary = summarize_grid_search(
        fitted_search, results, search_id="TEST-SEARCH", member="Member 01", branch="test"
    )
    json.dumps(summary, allow_nan=False)
    assert summary["n_candidates"] == 2
    assert summary["total_fits"] == 4
    assert summary["best_parameters"] == fitted_search.best_params_
    assert summary["final_test_used"] is False
    assert "best_estimator" not in summary


def test_save_csv_json_is_relative_and_refuses_overwrite(
    fitted_search: GridSearchCV, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = grid_search_results_to_dataframe(fitted_search)
    summary = summarize_grid_search(
        fitted_search, results, search_id="TEST-SEARCH", member="M01", branch="test"
    )
    monkeypatch.setattr("src.search.get_project_root", lambda: tmp_path)
    csv_path, json_path = save_grid_search_results(results, summary, output_dir="outputs")
    assert csv_path.is_file() and json_path.is_file()
    assert not Path("outputs").is_absolute()
    assert json.loads(json_path.read_text(encoding="utf-8"))["n_candidates"] == 2
    with pytest.raises(FileExistsError):
        save_grid_search_results(results, summary, output_dir="outputs")
    with pytest.raises(ValueError, match="relative"):
        save_grid_search_results(results, summary, output_dir=tmp_path)


def test_public_search_api_has_no_final_test_argument() -> None:
    for function in (
        create_grid_search, grid_search_results_to_dataframe,
        summarize_grid_search, save_grid_search_results,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {"X_test", "y_test", "test_data", "final_test"}


def test_convergence_metadata_can_be_recorded(fitted_search: GridSearchCV) -> None:
    results = grid_search_results_to_dataframe(fitted_search)
    summary = summarize_grid_search(
        fitted_search, results, search_id="TEST", member="M01", branch="test"
    )
    summary.update({"convergence_warning_count": 1, "convergence_configurations": []})
    assert json.loads(json.dumps(summary))["convergence_warning_count"] == 1

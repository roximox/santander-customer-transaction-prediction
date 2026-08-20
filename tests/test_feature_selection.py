"""Offline tests for M03-FS-001 and M03-PCA-001 pipeline construction and reporting helpers."""

from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.feature_selection as fs_module
from scripts.run_feature_selection import main as fs_main
from scripts.run_feature_selection import refuse_existing_outputs as fs_refuse
from scripts.run_pca import main as pca_main
from scripts.run_pca import refuse_existing_outputs as pca_refuse
from src.experiments import run_experiment
from src.feature_selection import (
    COMPARISON_COLUMNS,
    FS_EXPERIMENT_ID,
    FS_MODEL_NAME,
    PCA_EXPERIMENT_ID,
    PCA_MODEL_NAME,
    build_fs_pca_comparison,
    create_feature_selection_pipeline,
    create_pca_pipeline,
    save_fs_cv_figure,
    save_fs_pca_comparison,
    save_fs_vs_pca_figure,
    save_pca_cv_figure,
)
from src.validation import create_stratified_cv


def _summary(experiment_id: str, model_name: str, auc: float, ap: float, fit_time: float = 1.0) -> dict[str, object]:
    metric_names = (
        "roc_auc", "average_precision", "f1", "precision", "recall",
        "accuracy", "balanced_accuracy",
    )
    metrics = {
        name: {
            "train_mean": auc + 0.02 if name == "roc_auc" else ap + 0.01,
            "train_std": 0.01,
            "validation_mean": auc if name == "roc_auc" else ap,
            "validation_std": 0.01,
        }
        for name in metric_names
    }
    return {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "metrics": metrics,
        "fit_time_mean": fit_time,
        "score_time_mean": 0.1,
    }


def _comparison() -> pd.DataFrame:
    return build_fs_pca_comparison(
        [
            _summary(FS_EXPERIMENT_ID, FS_MODEL_NAME, 0.82, 0.45, 3.0),
            _summary(PCA_EXPERIMENT_ID, PCA_MODEL_NAME, 0.80, 0.42, 2.5),
        ]
    )


def test_fs_pipeline_has_exact_unfitted_configuration() -> None:
    pipeline = create_feature_selection_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "selector", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["selector"], SelectFromModel)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.penalty == "l2"
    assert classifier.C == 1.0
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42
    assert not hasattr(classifier, "coef_")


def test_pca_pipeline_has_exact_unfitted_configuration() -> None:
    pipeline = create_pca_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "pca", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    pca = pipeline.named_steps["pca"]
    assert isinstance(pca, PCA)
    assert pca.n_components == 0.95
    assert pca.random_state == 42
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.penalty == "l2"
    assert classifier.C == 1.0
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42
    assert not hasattr(classifier, "coef_")


def test_comparison_columns_and_gap() -> None:
    comparison = _comparison()
    assert tuple(comparison.columns) == COMPARISON_COLUMNS
    assert set(comparison["experiment_id"]) == {FS_EXPERIMENT_ID, PCA_EXPERIMENT_ID}
    fs_row = comparison.set_index("experiment_id").loc[FS_EXPERIMENT_ID]
    assert fs_row["roc_auc_generalization_gap"] == pytest.approx(0.02)
    json.dumps(comparison.to_dict(orient="records"), allow_nan=False)


def test_comparison_save_is_relative_serializable_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fs_module, "get_project_root", lambda: tmp_path)
    csv_path, json_path = save_fs_pca_comparison(
        _comparison(), csv_path="comparison.csv", json_path="comparison.json"
    )
    assert csv_path.is_file() and json_path.is_file()
    assert str(tmp_path) not in json_path.read_text(encoding="utf-8")
    json.loads(json_path.read_text(encoding="utf-8"))
    with pytest.raises(FileExistsError, match="already exists"):
        save_fs_pca_comparison(
            _comparison(), csv_path="comparison.csv", json_path="comparison.json"
        )


def test_figures_are_created_in_temporary_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fs_module, "get_project_root", lambda: tmp_path)
    folds = pd.DataFrame(
        {
            "fold": [1, 2],
            "validation_roc_auc": [0.82, 0.83],
            "validation_average_precision": [0.45, 0.46],
        }
    )
    fs_figure = save_fs_cv_figure(folds, "fs_cv.pdf")
    pca_figure = save_pca_cv_figure(folds, "pca_cv.pdf")
    comparison_figure = save_fs_vs_pca_figure(_comparison(), "comparison.pdf")
    assert fs_figure.read_bytes().startswith(b"%PDF")
    assert pca_figure.read_bytes().startswith(b"%PDF")
    assert comparison_figure.read_bytes().startswith(b"%PDF")


def test_script_public_api_has_no_final_test_parameters() -> None:
    for function in (fs_main, fs_refuse, pca_main, pca_refuse):
        parameters = inspect.signature(function).parameters
        assert "X_test" not in parameters and "y_test" not in parameters


def test_convergence_warning_is_detected_and_not_silently_masked() -> None:
    values, labels = make_classification(
        n_samples=200, n_features=20, n_informative=10, random_state=42
    )
    X = pd.DataFrame(values)
    y = pd.Series(labels, index=X.index)
    # Force convergence failure by allowing only 1 iteration.
    from sklearn.pipeline import Pipeline as _Pipeline
    from sklearn.preprocessing import StandardScaler as _Scaler
    from sklearn.feature_selection import SelectFromModel as _SFM
    fast_estimator = _Pipeline([
        ("scaler", _Scaler()),
        ("selector", _SFM(LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=1, random_state=42))),
        ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=1, random_state=42)),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _, summary = run_experiment(
            fast_estimator, X, y,
            experiment_id="TECH-FS-CONVERGENCE",
            model_name="Technical FS convergence check",
            member="test", branch="test",
            cv=create_stratified_cv(n_splits=2), n_jobs=1,
        )
    assert summary["convergence_warning_detected"] is True
    assert summary["convergence_warning_messages"]
    assert any(issubclass(item.category, ConvergenceWarning) for item in caught)

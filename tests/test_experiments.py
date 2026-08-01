"""Offline tests for the shared experiment orchestrator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

import src.experiments as experiments
from src.config import get_project_root
from src.experiments import (
    build_logbook_metadata,
    run_and_save_experiment,
    run_experiment,
    validate_experiment_metadata,
)


def _data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": range(30), "b": [value % 4 for value in range(30)]})
    y = pd.Series([0] * 24 + [1] * 6, index=X.index, name="target")
    return X, y


def _kwargs(experiment_id: str = "TECH-ORCHESTRATOR-TEST") -> dict[str, str]:
    return {
        "experiment_id": experiment_id,
        "model_name": "Technical estimator",
        "member": "Member 01",
        "branch": "feature/data_processing",
    }


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def test_validate_experiment_metadata_accepts_valid_id() -> None:
    validate_experiment_metadata(**_kwargs("M02-RF_BASELINE-001"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", ""),
        ("experiment_id", "M01 LR 001"),
        ("experiment_id", "../../result"),
        ("model_name", " "),
        ("member", ""),
        ("branch", ""),
    ],
)
def test_validate_experiment_metadata_rejects_invalid_values(
    field: str, value: str
) -> None:
    values = _kwargs()
    values[field] = value
    with pytest.raises(ValueError, match=field):
        validate_experiment_metadata(**values)


def test_run_without_saving_returns_results_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    X, y = _data()
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    folds, summary = run_experiment(
        DummyClassifier(strategy="prior"),
        X,
        y,
        cv=_cv(),
        n_jobs=1,
        output_dir=tmp_path,
        **_kwargs(),
    )
    assert len(folds) == 3
    assert summary["experiment_id"] == "TECH-ORCHESTRATOR-TEST"
    assert summary["saved"] is False and summary["registered"] is False
    assert list(tmp_path.iterdir()) == []
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_register_without_saving_is_rejected(tmp_path: Path) -> None:
    X, y = _data()
    with pytest.raises(ValueError, match="requires save_results"):
        run_experiment(
            DummyClassifier(), X, y, register_experiment=True,
            output_dir=tmp_path, cv=_cv(), n_jobs=1, **_kwargs()
        )


def test_save_and_register_creates_files_with_relative_paths(tmp_path: Path) -> None:
    X, y = _data()
    work = get_project_root() / ".pytest_experiments" / tmp_path.name
    registry = work / "registry.csv"
    try:
        _, summary = run_and_save_experiment(
            DummyClassifier(strategy="prior"), X, y,
            output_dir=work, registry_path=registry, cv=_cv(), n_jobs=1,
            **_kwargs("TECH-SAVE-001"),
        )
        assert (work / "TECH-SAVE-001_fold_results.csv").is_file()
        assert (work / "TECH-SAVE-001_summary.json").is_file()
        assert registry.is_file()
        assert summary["saved"] is True and summary["registered"] is True
        for key in ("fold_results_file", "summary_file", "registry_file"):
            assert not Path(summary[key]).is_absolute()
            assert "/Users/" not in summary[key]
        row = pd.read_csv(registry).iloc[0]
        assert row["summary_file"] == summary["summary_file"]
        with pytest.raises(FileExistsError):
            run_and_save_experiment(
                DummyClassifier(), X, y, output_dir=work,
                registry_path=registry, cv=_cv(), n_jobs=1,
                **_kwargs("TECH-SAVE-001"),
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_run_and_save_delegates_to_run_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _data()
    sentinel = (pd.DataFrame(), {"ok": True})
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> tuple[pd.DataFrame, dict[str, bool]]:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(experiments, "run_experiment", fake_run)
    result = run_and_save_experiment(DummyClassifier(), X, y, **_kwargs())
    assert result is sentinel
    assert captured["save_results"] is True
    assert captured["register_experiment"] is True


def test_build_logbook_metadata_contains_only_verified_facts() -> None:
    allowed = {
        "experiment_id", "model_name", "date_utc", "member", "branch",
        "n_samples", "n_features", "primary_metric", "primary_score_mean",
        "primary_score_std", "fit_time_mean", "summary_file",
    }
    result = build_logbook_metadata({"experiment_id": "M01-X-001", "invented": "no"})
    assert set(result) <= allowed
    assert "invented" not in result


def test_evaluation_errors_are_not_hidden() -> None:
    X, y = _data()
    with pytest.raises(ValueError, match="at least two classes"):
        run_experiment(DummyClassifier(), X, y * 0, cv=_cv(), n_jobs=1, **_kwargs())


def test_saved_summary_contains_no_personal_absolute_path(tmp_path: Path) -> None:
    X, y = _data()
    work = get_project_root() / ".pytest_experiments" / tmp_path.name
    try:
        _, summary = run_and_save_experiment(
            DummyClassifier(), X, y, output_dir=work, register_experiment=False,
            cv=_cv(), n_jobs=1, **_kwargs("TECH-JSON-001"),
        )
        payload = json.dumps(summary)
        assert "/Users/admin" not in payload
    finally:
        shutil.rmtree(work, ignore_errors=True)

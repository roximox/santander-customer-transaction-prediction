"""Offline safeguards for the Member 01 Extra Trees experiment entry point."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_extra_trees_baseline as module


def test_extra_trees_experiment_metadata_and_outputs() -> None:
    assert module.EXPERIMENT_ID == "M01-ET-001"
    assert module.MODEL_NAME == "Extra Trees Baseline"
    assert [path.name for path in module.requested_outputs()] == [
        "M01-ET-001_fold_results.csv",
        "M01-ET-001_summary.json",
    ]


def test_extra_trees_entry_point_has_no_final_test_parameters() -> None:
    parameters = inspect.signature(module.main).parameters
    assert not {"X_test", "y_test", "test_data", "final_test"} & set(parameters)
    source = inspect.getsource(module.main)
    assert "X_train" in source and "y_train" in source
    assert "del X_reserved, y_reserved" in source


def test_refuse_existing_extra_trees_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments = tmp_path / "reports/experiments"
    experiments.mkdir(parents=True)
    registry = experiments / "experiment_registry.csv"
    pd.DataFrame({"experiment_id": [module.EXPERIMENT_ID]}).to_csv(registry, index=False)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "EXPERIMENTS_DIR", experiments)
    monkeypatch.setattr(module, "REGISTRY_PATH", registry)
    with pytest.raises(FileExistsError, match=module.EXPERIMENT_ID):
        module.refuse_existing_outputs()

"""Run an offline technical smoke test of experiment orchestration."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_project_root  # noqa: E402
from src.experiments import run_and_save_experiment, run_experiment  # noqa: E402
from src.validation import create_stratified_cv  # noqa: E402


def main() -> None:
    """Verify no-save and temporary save/register workflows."""
    values, labels = make_classification(
        n_samples=200, n_features=5, n_informative=3, n_redundant=0,
        weights=[0.9, 0.1], random_state=42,
    )
    X = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(labels, name="target")
    common = {
        "experiment_id": "TECH-ORCHESTRATOR-SMOKE",
        "model_name": "Technical prior estimator",
        "member": "Technical smoke test",
        "branch": "feature/data_processing",
        "cv": create_stratified_cv(),
        "n_jobs": 1,
    }
    with tempfile.TemporaryDirectory(
        prefix=".orchestrator-smoke-", dir=get_project_root()
    ) as directory:
        temporary = Path(directory)
        folds, unsaved = run_experiment(
            DummyClassifier(strategy="prior"), X, y,
            output_dir=temporary, save_results=False, **common,
        )
        assert not any(temporary.iterdir())
        _, saved = run_and_save_experiment(
            DummyClassifier(strategy="prior"), X, y,
            output_dir=temporary,
            registry_path=temporary / "experiment_registry.csv",
            **common,
        )
        assert (temporary / "TECH-ORCHESTRATOR-SMOKE_fold_results.csv").is_file()
        assert (temporary / "TECH-ORCHESTRATOR-SMOKE_summary.json").is_file()
        assert (temporary / "experiment_registry.csv").is_file()
        print("Technical smoke test only — not a Santander scientific experiment.")
        print(f"Folds: {len(folds)}")
        print(f"Unsaved run saved: {unsaved['saved']}")
        print(f"Temporary run registered: {saved['registered']}")
        print(f"Primary metric: {saved['primary_metric']}")
        print(f"Primary score mean: {saved['primary_score_mean']:.6f}")


if __name__ == "__main__":
    main()

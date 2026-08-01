"""Run and register the four scientific DummyClassifier baselines."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "santander-cache"))

from src.data import load_dataset  # noqa: E402
from src.dummy_baselines import (  # noqa: E402
    DUMMY_EXPERIMENTS,
    build_comparison_table,
    build_dummy_classifiers,
    factual_interpretation,
    save_comparison_table,
    save_metrics_figure,
)
from src.experiments import run_and_save_experiment, run_experiment  # noqa: E402
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN_FINGERPRINT = (
    "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
)
EXPECTED_TEST_FINGERPRINT = (
    "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
)
RESULTS_DIRECTORY = PROJECT_ROOT / "reports/experiments"
REGISTRY_PATH = RESULTS_DIRECTORY / "experiment_registry.csv"
COMPARISON_CSV = PROJECT_ROOT / "reports/tables/dummy_baseline_comparison.csv"
COMPARISON_JSON = PROJECT_ROOT / "reports/tables/dummy_baseline_comparison.json"
FIGURE_PATH = PROJECT_ROOT / "reports/figures/dummy_baseline_metrics.pdf"


def _refuse_existing_outputs() -> None:
    """Fail before loading data if any requested scientific artifact exists."""
    requested = [COMPARISON_CSV, COMPARISON_JSON, FIGURE_PATH]
    for definition in DUMMY_EXPERIMENTS:
        experiment_id = definition["experiment_id"]
        requested.extend(
            [
                RESULTS_DIRECTORY / f"{experiment_id}_fold_results.csv",
                RESULTS_DIRECTORY / f"{experiment_id}_summary.json",
            ]
        )
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in requested if path.exists()]
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH)
        if "experiment_id" not in registry.columns:
            raise ValueError("Existing experiment registry lacks experiment_id column.")
        requested_ids = {item["experiment_id"] for item in DUMMY_EXPERIMENTS}
        duplicate_ids = sorted(requested_ids & set(registry["experiment_id"].astype(str)))
        existing.extend(f"registry:{item}" for item in duplicate_ids)
    if existing:
        details = "\n- ".join(existing)
        raise FileExistsError(
            "Dummy baseline outputs already exist; nothing was overwritten. "
            "Review and remove only the intended four experiments manually before "
            f"rerunning:\n- {details}"
        )


def _assert_random_reproducibility(
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metadata: dict[str, str],
) -> None:
    """Evaluate a random Dummy strategy twice and compare every score exactly."""
    first, _ = run_experiment(estimator, X_train, y_train, n_jobs=1, **metadata)
    second, _ = run_experiment(estimator, X_train, y_train, n_jobs=1, **metadata)
    score_columns = [
        column for column in first.columns
        if column.startswith(("train_", "validation_"))
    ]
    pd.testing.assert_frame_equal(first[score_columns], second[score_columns])


def main() -> None:
    """Execute training-only baselines, reports, registry, and visualization."""
    _refuse_existing_outputs()
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_fingerprint = split_fingerprint(X_train.index)
    reserved_fingerprint = split_fingerprint(X_reserved.index)
    if train_fingerprint != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(
            f"Train split fingerprint mismatch: {train_fingerprint}; experiments stopped."
        )
    if reserved_fingerprint != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(
            f"Reserved test split fingerprint mismatch: {reserved_fingerprint}; "
            "experiments stopped."
        )
    del X_reserved, y_reserved

    classifiers = build_dummy_classifiers(random_state=42)
    summaries: list[dict[str, object]] = []
    for definition in DUMMY_EXPERIMENTS:
        strategy = definition["strategy"]
        metadata = {
            "experiment_id": definition["experiment_id"],
            "model_name": f"DummyClassifier ({strategy})",
            "member": "Member 01",
            "branch": "feature/data_processing",
        }
        if strategy in {"stratified", "uniform"}:
            _assert_random_reproducibility(
                classifiers[strategy], X_train, y_train, metadata
            )
        _, summary = run_and_save_experiment(
            classifiers[strategy], X_train, y_train,
            output_dir=RESULTS_DIRECTORY,
            registry_path=REGISTRY_PATH,
            n_jobs=1,
            **metadata,
        )
        summaries.append(summary)

    comparison = build_comparison_table(summaries)
    save_comparison_table(
        comparison, csv_path=COMPARISON_CSV, json_path=COMPARISON_JSON
    )
    save_metrics_figure(comparison, FIGURE_PATH)
    prevalence = float((y_train == "True").mean())
    print(comparison.to_string(index=False))
    print()
    print(factual_interpretation(comparison, prevalence))
    print("\nFinal test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

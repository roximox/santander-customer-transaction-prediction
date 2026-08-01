"""Run the single registered M01-LR-001 training-only scientific baseline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "santander-cache"))

from src.data import load_dataset  # noqa: E402
from src.experiments import run_and_save_experiment  # noqa: E402
from src.logistic_baseline import (  # noqa: E402
    LOGISTIC_EXPERIMENT_ID,
    LOGISTIC_MODEL_NAME,
    build_logistic_comparison,
    calculate_baseline_improvements,
    create_logistic_baseline_pipeline,
    save_logistic_comparison,
    save_logistic_cv_figure,
    save_logistic_vs_dummy_figure,
)
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports/experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "experiment_registry.csv"
COMPARISON_CSV = PROJECT_ROOT / "reports/tables/logistic_baseline_comparison.csv"
COMPARISON_JSON = PROJECT_ROOT / "reports/tables/logistic_baseline_comparison.json"
METRICS_FIGURE = PROJECT_ROOT / "reports/figures/logistic_vs_dummy_metrics.pdf"
CV_FIGURE = PROJECT_ROOT / "reports/figures/logistic_cv_scores.pdf"


def _requested_outputs() -> list[Path]:
    return [
        EXPERIMENTS_DIR / f"{LOGISTIC_EXPERIMENT_ID}_fold_results.csv",
        EXPERIMENTS_DIR / f"{LOGISTIC_EXPERIMENT_ID}_summary.json",
        COMPARISON_CSV,
        COMPARISON_JSON,
        METRICS_FIGURE,
        CV_FIGURE,
    ]


def refuse_existing_outputs() -> None:
    """Refuse duplicate Logistic artifacts without touching Dummy baselines."""
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in _requested_outputs() if path.exists()]
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH)
        if "experiment_id" not in registry.columns:
            raise ValueError("Existing experiment registry lacks experiment_id column.")
        if LOGISTIC_EXPERIMENT_ID in registry["experiment_id"].astype(str).tolist():
            existing.append(f"registry:{LOGISTIC_EXPERIMENT_ID}")
    if existing:
        details = "\n- ".join(existing)
        raise FileExistsError(
            f"{LOGISTIC_EXPERIMENT_ID} outputs already exist and were not overwritten. "
            "Review and remove only these targeted Logistic files manually before "
            f"rerunning:\n- {details}"
        )


def _load_dummy_summaries() -> list[dict[str, object]]:
    summaries = []
    for sequence in range(1, 5):
        path = EXPERIMENTS_DIR / f"M01-DUMMY-{sequence:03d}_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Required Dummy baseline summary is missing: {path.name}")
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def main() -> None:
    """Execute, register, compare, and plot the exact L2 baseline."""
    refuse_existing_outputs()
    dummy_summaries = _load_dummy_summaries()
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_hash = split_fingerprint(X_train.index)
    reserved_hash = split_fingerprint(X_reserved.index)
    if train_hash != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(f"Train fingerprint mismatch: {train_hash}; experiment stopped.")
    if reserved_hash != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(f"Reserved test fingerprint mismatch: {reserved_hash}; experiment stopped.")
    del X_reserved, y_reserved

    pipeline = create_logistic_baseline_pipeline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fold_results, summary = run_and_save_experiment(
            pipeline,
            X_train,
            y_train,
            experiment_id=LOGISTIC_EXPERIMENT_ID,
            model_name=LOGISTIC_MODEL_NAME,
            member="Member 01",
            branch="feature/data_processing",
            n_jobs=1,
            output_dir=EXPERIMENTS_DIR,
            registry_path=REGISTRY_PATH,
        )
    observed_convergence = [
        str(item.message) for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    if bool(observed_convergence) != bool(summary["convergence_warning_detected"]):
        raise RuntimeError("Convergence warning capture disagrees with saved summary.")

    comparison = build_logistic_comparison([*dummy_summaries, summary])
    save_logistic_comparison(
        comparison, csv_path=COMPARISON_CSV, json_path=COMPARISON_JSON
    )
    save_logistic_vs_dummy_figure(comparison, METRICS_FIGURE)
    save_logistic_cv_figure(fold_results, CV_FIGURE)
    improvements = calculate_baseline_improvements(comparison)

    metric_rows = {
        name: values for name, values in summary["metrics"].items()
    }
    print("M01-LR-001 mean and standard deviation by metric:")
    for name, values in metric_rows.items():
        print(
            f"  {name}: train={values.get('train_mean', float('nan')):.6f}, "
            f"validation={values['validation_mean']:.6f} ± {values['validation_std']:.6f}"
        )
    print("\nValidation scores by fold:")
    print(
        fold_results[
            ["fold", "validation_roc_auc", "validation_average_precision"]
        ].to_string(index=False)
    )
    print("\nComparison with Dummy baselines:")
    print(comparison.to_string(index=False))
    print("\nCalculated comparisons:")
    for name, value in improvements.items():
        print(f"  {name}: {value:.6f}")
    classifier_parameters = summary["estimator_parameters"]
    print(
        f"\nSolver={classifier_parameters['classifier__solver']}, "
        f"max_iter={classifier_parameters['classifier__max_iter']}"
    )
    if observed_convergence:
        print("Convergence warning detected; this experiment was not altered.")
        for message in observed_convergence:
            print(f"  {message}")
    else:
        print("No ConvergenceWarning detected.")
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

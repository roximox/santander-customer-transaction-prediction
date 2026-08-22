"""Run the registered M04-HGB-002 training-only tuned HGB experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset  # noqa: E402
from src.experiments import run_and_save_experiment  # noqa: E402
from src.gradient_boosting_learning_curve import (  # noqa: E402
    create_tuned_hist_gradient_boosting_estimator,
)
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPERIMENT_ID = "M04-HGB-002"
MODEL_NAME = "HistGradientBoosting Tuned"
EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports/experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "experiment_registry.csv"


def refuse_existing_outputs() -> None:
    """Refuse duplicate experiment outputs or a duplicate registry entry."""
    outputs = (
        EXPERIMENTS_DIR / f"{EXPERIMENT_ID}_fold_results.csv",
        EXPERIMENTS_DIR / f"{EXPERIMENT_ID}_summary.json",
    )
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in outputs if path.exists()]
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH)
        if "experiment_id" not in registry.columns:
            raise ValueError("Existing experiment registry lacks experiment_id column.")
        if EXPERIMENT_ID in registry["experiment_id"].astype(str).tolist():
            existing.append(f"registry:{EXPERIMENT_ID}")
    if existing:
        raise FileExistsError(
            f"{EXPERIMENT_ID} outputs already exist and were not overwritten. "
            "Remove only the targeted artifacts manually before rerunning:\n- "
            + "\n- ".join(existing)
        )


def print_result_summary(fold_results: pd.DataFrame, summary: dict[str, object]) -> None:
    """Print the requested concise training-only validation summary."""
    metrics = summary["metrics"]
    roc_auc = metrics["roc_auc"]
    average_precision = metrics["average_precision"]
    print(f"{EXPERIMENT_ID} validation summary:")
    print(f"  ROC-AUC: {roc_auc['validation_mean']:.6f} +/- {roc_auc['validation_std']:.6f}")
    print(
        "  Average Precision: "
        f"{average_precision['validation_mean']:.6f} +/- "
        f"{average_precision['validation_std']:.6f}"
    )
    for name, label in (
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ):
        print(f"  {label}: {metrics[name]['validation_mean']:.6f}")
    train_roc_auc = roc_auc["train_mean"]
    validation_roc_auc = roc_auc["validation_mean"]
    print(f"  Train ROC-AUC: {train_roc_auc:.6f}")
    print(f"  Validation ROC-AUC: {validation_roc_auc:.6f}")
    print(f"  Generalization gap: {train_roc_auc - validation_roc_auc:.6f}")
    print("\nPer-fold validation scores:")
    print(
        fold_results[["fold", "validation_roc_auc", "validation_average_precision"]]
        .to_string(index=False)
    )


def main() -> None:
    """Evaluate and register frozen tuned HGB on the verified training partition."""
    refuse_existing_outputs()
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_hash = split_fingerprint(X_train.index)
    reserved_hash = split_fingerprint(X_reserved.index)
    if train_hash != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(f"Train fingerprint mismatch: {train_hash}; experiment stopped.")
    if reserved_hash != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(f"Reserved test fingerprint mismatch: {reserved_hash}; experiment stopped.")
    del X_reserved, y_reserved

    estimator = create_tuned_hist_gradient_boosting_estimator()
    fold_results, summary = run_and_save_experiment(
        estimator,
        X_train,
        y_train,
        experiment_id=EXPERIMENT_ID,
        model_name=MODEL_NAME,
        member="Member 04",
        branch="feature/model-optimization",
        n_jobs=1,
        output_dir=EXPERIMENTS_DIR,
        registry_path=REGISTRY_PATH,
    )
    print_result_summary(fold_results, summary)
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

"""Run and register the Member 01 training-only Extra Trees baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset  # noqa: E402
from src.experiments import run_and_save_experiment  # noqa: E402
from src.modeling import create_extra_trees_classifier  # noqa: E402
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPERIMENT_ID = "M01-ET-001"
MODEL_NAME = "Extra Trees Baseline"
EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_RESERVED_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports/experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "experiment_registry.csv"


def requested_outputs() -> tuple[Path, Path]:
    """Return the two official outputs produced by this experiment."""
    return (
        EXPERIMENTS_DIR / f"{EXPERIMENT_ID}_fold_results.csv",
        EXPERIMENTS_DIR / f"{EXPERIMENT_ID}_summary.json",
    )


def refuse_existing_outputs() -> None:
    """Prevent artifact overwrites and duplicate registry entries."""
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in requested_outputs() if path.exists()]
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH)
        if "experiment_id" not in registry.columns:
            raise ValueError("Existing experiment registry lacks experiment_id column.")
        if EXPERIMENT_ID in registry["experiment_id"].astype(str).tolist():
            existing.append(f"registry:{EXPERIMENT_ID}")
    if existing:
        raise FileExistsError(
            f"{EXPERIMENT_ID} outputs already exist and were not overwritten:\n- "
            + "\n- ".join(existing)
        )


def main() -> None:
    """Evaluate Extra Trees using only the verified development partition."""
    refuse_existing_outputs()
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_fingerprint = split_fingerprint(X_train.index)
    reserved_fingerprint = split_fingerprint(X_reserved.index)
    if train_fingerprint != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(f"Train fingerprint mismatch: {train_fingerprint}")
    if reserved_fingerprint != EXPECTED_RESERVED_FINGERPRINT:
        raise RuntimeError(f"Reserved-test fingerprint mismatch: {reserved_fingerprint}")
    del X_reserved, y_reserved, X, y

    estimator = create_extra_trees_classifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    fold_results, summary = run_and_save_experiment(
        estimator,
        X_train,
        y_train,
        experiment_id=EXPERIMENT_ID,
        model_name=MODEL_NAME,
        member="Member 01",
        branch="develop",
        n_jobs=1,
        output_dir=EXPERIMENTS_DIR,
        registry_path=REGISTRY_PATH,
    )
    print(f"{EXPERIMENT_ID} — {MODEL_NAME}")
    print(fold_results[["fold", "validation_roc_auc", "validation_average_precision"]].to_string(index=False))
    for metric_name, values in summary["metrics"].items():
        print(
            f"  {metric_name}: validation={values['validation_mean']:.6f} "
            f"± {values['validation_std']:.6f}"
        )
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

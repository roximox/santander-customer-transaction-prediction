"""Run the registered M03-PCA-001 training-only PCA experiment."""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "santander-cache"))

import pandas as pd  # noqa: E402

from src.data import load_dataset  # noqa: E402
from src.experiments import run_and_save_experiment  # noqa: E402
from src.feature_selection import (  # noqa: E402
    PCA_EXPERIMENT_ID,
    PCA_MODEL_NAME,
    create_pca_pipeline,
    save_pca_cv_figure,
)
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
EXPERIMENTS_DIR = PROJECT_ROOT / "reports/experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "experiment_registry.csv"
CV_FIGURE = PROJECT_ROOT / "reports/figures/pca_cv_scores.pdf"


def _requested_outputs() -> list[Path]:
    return [
        EXPERIMENTS_DIR / f"{PCA_EXPERIMENT_ID}_fold_results.csv",
        EXPERIMENTS_DIR / f"{PCA_EXPERIMENT_ID}_summary.json",
        CV_FIGURE,
    ]


def refuse_existing_outputs() -> None:
    """Refuse duplicate PCA artifacts without touching other experiment results."""
    existing = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in _requested_outputs()
        if path.exists()
    ]
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH)
        if "experiment_id" not in registry.columns:
            raise ValueError("Existing experiment registry lacks experiment_id column.")
        if PCA_EXPERIMENT_ID in registry["experiment_id"].astype(str).tolist():
            existing.append(f"registry:{PCA_EXPERIMENT_ID}")
    if existing:
        details = "\n- ".join(existing)
        raise FileExistsError(
            f"{PCA_EXPERIMENT_ID} outputs already exist and were not overwritten. "
            "Review and remove only these targeted files manually before "
            f"rerunning:\n- {details}"
        )


def main() -> None:
    """Execute, register, and plot the PCA dimensionality reduction baseline."""
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

    pipeline = create_pca_pipeline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fold_results, summary = run_and_save_experiment(
            pipeline,
            X_train,
            y_train,
            experiment_id=PCA_EXPERIMENT_ID,
            model_name=PCA_MODEL_NAME,
            member="Member 03",
            branch="feature/pca",
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

    save_pca_cv_figure(fold_results, CV_FIGURE)

    metric_rows = {name: values for name, values in summary["metrics"].items()}
    print(f"{PCA_EXPERIMENT_ID} mean and standard deviation by metric:")
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
    if observed_convergence:
        print("Convergence warning detected.")
        for message in observed_convergence:
            print(f"  {message}")
    else:
        print("No ConvergenceWarning detected.")
    print("Final test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

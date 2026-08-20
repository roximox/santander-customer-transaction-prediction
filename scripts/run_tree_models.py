"""Run and register the Member 02 training-only tree model experiments."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "santander-cache")
)

import matplotlib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset  # noqa: E402
from src.experiments import run_and_save_experiment  # noqa: E402
from src.modeling import create_random_forest_classifier  # noqa: E402
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402
from src.visualization import save_figure  # noqa: E402

EXPECTED_TRAIN_FINGERPRINT = (
    "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
)
EXPECTED_TEST_FINGERPRINT = (
    "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
)
EXPERIMENT_IDS = ("M02-DT-001", "M02-RF-001")
EXPERIMENTS_DIR = PROJECT_ROOT / "reports/experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "experiment_registry.csv"
COMPARISON_CSV = PROJECT_ROOT / "reports/tables/tree_model_comparison.csv"
COMPARISON_JSON = PROJECT_ROOT / "reports/tables/tree_model_comparison.json"
METRICS_FIGURE = PROJECT_ROOT / "reports/figures/tree_model_metrics.pdf"


def requested_outputs() -> list[Path]:
    """Return every artifact produced by this exact experiment run."""
    experiment_files = [
        EXPERIMENTS_DIR / f"{experiment_id}_{suffix}"
        for experiment_id in EXPERIMENT_IDS
        for suffix in ("fold_results.csv", "summary.json")
    ]
    return [
        *experiment_files,
        COMPARISON_CSV,
        COMPARISON_JSON,
        METRICS_FIGURE,
    ]


def refuse_existing_outputs() -> None:
    """Prevent accidental overwrites or duplicate registry entries."""
    existing = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in requested_outputs()
        if path.exists()
    ]
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH)
        if "experiment_id" not in registry.columns:
            raise ValueError("Existing experiment registry lacks experiment_id column.")
        registered = set(registry["experiment_id"].astype(str))
        existing.extend(
            f"registry:{experiment_id}"
            for experiment_id in EXPERIMENT_IDS
            if experiment_id in registered
        )
    if existing:
        details = "\n- ".join(existing)
        raise FileExistsError(
            "Tree experiment outputs already exist and were not overwritten. "
            f"Review them before rerunning:\n- {details}"
        )


def build_comparison(summaries: list[dict[str, object]]) -> pd.DataFrame:
    """Build a compact comparison from saved shared-framework summaries."""
    rows: list[dict[str, object]] = []
    for summary in summaries:
        metrics = summary["metrics"]
        row: dict[str, object] = {
            "experiment_id": summary["experiment_id"],
            "model_name": summary["model_name"],
            "fit_time_mean": summary["fit_time_mean"],
        }
        for metric_name, values in metrics.items():
            row[f"validation_{metric_name}_mean"] = values["validation_mean"]
            row[f"validation_{metric_name}_std"] = values["validation_std"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "validation_roc_auc_mean", ascending=False
    )


def save_comparison(comparison: pd.DataFrame) -> None:
    """Save the comparison table and a primary/secondary metric figure."""
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_CSV, index=False)
    COMPARISON_JSON.write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    metric_columns = [
        "validation_roc_auc_mean",
        "validation_average_precision_mean",
        "validation_balanced_accuracy_mean",
    ]
    plot_data = comparison.set_index("model_name")[metric_columns]
    plot_data.columns = [
        "ROC-AUC",
        "Average Precision",
        "Balanced Accuracy",
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    plot_data.plot(kind="bar", ax=axis, rot=0)
    axis.set_ylabel("Mean validation score")
    axis.set_ylim(0, 1)
    axis.set_title("Tree models — shared five-fold cross-validation")
    axis.legend(loc="lower right")
    figure.tight_layout()
    save_figure(figure, METRICS_FIGURE)
    plt.close(figure)


def main() -> None:
    """Evaluate, register, compare, and report the two tree experiments."""
    refuse_existing_outputs()
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_hash = split_fingerprint(X_train.index)
    reserved_hash = split_fingerprint(X_reserved.index)
    if train_hash != EXPECTED_TRAIN_FINGERPRINT:
        raise RuntimeError(f"Train fingerprint mismatch: {train_hash}")
    if reserved_hash != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(f"Reserved-test fingerprint mismatch: {reserved_hash}")
    del X_reserved, y_reserved

    experiments = [
        (
            DecisionTreeClassifier(
                max_depth=5,
                class_weight="balanced",
                random_state=42,
            ),
            "M02-DT-001",
            "Decision Tree",
        ),
        (
            create_random_forest_classifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "M02-RF-001",
            "Random Forest",
        ),
    ]

    summaries: list[dict[str, object]] = []
    for estimator, experiment_id, model_name in experiments:
        fold_results, summary = run_and_save_experiment(
            estimator=estimator,
            X=X_train,
            y=y_train,
            experiment_id=experiment_id,
            model_name=model_name,
            member="Member 02",
            branch="feature/eda+tree_models",
            n_jobs=1,
            output_dir=EXPERIMENTS_DIR,
            registry_path=REGISTRY_PATH,
        )
        summaries.append(summary)
        print(f"\n{experiment_id} — {model_name}")
        print(
            fold_results[
                ["fold", "validation_roc_auc", "validation_average_precision"]
            ].to_string(index=False)
        )
        for metric_name, values in summary["metrics"].items():
            print(
                f"  {metric_name}: validation={values['validation_mean']:.6f} "
                f"± {values['validation_std']:.6f}"
            )

    comparison = build_comparison(summaries)
    save_comparison(comparison)
    print("\nTree-model comparison:")
    print(comparison.to_string(index=False))
    print("\nFinal test partition remained reserved and was not evaluated.")


if __name__ == "__main__":
    main()

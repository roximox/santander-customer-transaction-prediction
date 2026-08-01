"""Run the predeclared M01 Logistic Regression training-only grid search."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "santander-cache"))

import matplotlib.pyplot as plt  # noqa: E402

from src.data import load_dataset  # noqa: E402
from src.modeling import create_logistic_regression_pipeline  # noqa: E402
from src.search import (  # noqa: E402
    create_grid_search,
    create_logistic_parameter_grid,
    grid_search_results_to_dataframe,
    save_grid_search_results,
    summarize_grid_search,
)
from src.validation import create_train_test_split, split_fingerprint  # noqa: E402

SEARCH_ID = "M01-LR-SEARCH-001"
EXPECTED_TRAIN_FINGERPRINT = "61c403ec521d15ab9d6316606eba5acdfc22381cb764b6da76a65041ec11f477"
EXPECTED_TEST_FINGERPRINT = "bf7d43e492967dad1358676e9bf8355a910823077b59118014db12af3e26f586"
SEARCH_DIR = PROJECT_ROOT / "reports/searches"
TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures"
OUTPUTS = (
    SEARCH_DIR / f"{SEARCH_ID}_candidates.csv",
    SEARCH_DIR / f"{SEARCH_ID}_summary.json",
    TABLE_DIR / "logistic_grid_search_top_candidates.csv",
    TABLE_DIR / "logistic_grid_search_top_candidates.json",
    TABLE_DIR / "logistic_grid_search_decision_table.csv",
    FIGURE_DIR / "logistic_grid_search_roc_auc.pdf",
    FIGURE_DIR / "logistic_grid_search_tradeoff.pdf",
    FIGURE_DIR / "logistic_grid_search_train_validation.pdf",
)


def refuse_existing_outputs() -> None:
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError(
            f"{SEARCH_ID} outputs already exist and were not overwritten. Remove only "
            "the targeted artifacts manually if this search must truly be repeated:\n- "
            + "\n- ".join(existing)
        )


def _write_table(table: pd.DataFrame, csv_path: Path, json_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, encoding="utf-8")
    if json_path is not None:
        # pandas emits JSON null for missing class_weight values; round-tripping
        # avoids passing NaN to Python's strict JSON encoder.
        records = json.loads(table.to_json(orient="records"))
        json_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _label(row: pd.Series) -> str:
    weight = "balanced" if row["class_weight"] == "balanced" else "None"
    return f"{row['candidate_id']} ({row['penalty']}, C={row['C']:g}, {weight})"


def save_figures(results: pd.DataFrame, top: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    styles = {("l1", None): "o-", ("l1", "balanced"): "s-", ("l2", None): "^-", ("l2", "balanced"): "D-"}
    figure, axis = plt.subplots(figsize=(9, 5.8))
    for (penalty, weight), marker in styles.items():
        subset = results[(results["penalty"] == penalty) & (results["class_weight"].isna() if weight is None else results["class_weight"].eq(weight))].sort_values("C")
        axis.errorbar(subset["C"], subset["validation_roc_auc_mean"], yerr=subset["validation_roc_auc_std"], fmt=marker, capsize=3, label=f"{penalty.upper()} / {weight or 'None'}")
    axis.set_xscale("log")
    axis.set(title="Logistic Grid Search — Validation ROC-AUC", xlabel="Inverse regularization strength C (log scale)", ylabel="Mean validation ROC-AUC")
    axis.grid(alpha=0.25); axis.legend(frameon=False)
    figure.tight_layout(); figure.savefig(OUTPUTS[5], format="pdf", bbox_inches="tight"); plt.close(figure)

    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    plot = top.head(6).set_index("candidate_id")[[f"validation_{metric}_mean" for metric in metrics]]
    plot.columns = ["Precision", "Recall", "F1", "Balanced accuracy"]
    figure, axis = plt.subplots(figsize=(11, 6))
    plot.plot(kind="bar", ax=axis, width=0.78)
    axis.set(title="Threshold-Metric Trade-offs Among Top ROC-AUC Candidates", xlabel="Candidate", ylabel="Mean validation score", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25); axis.legend(frameon=False)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout(); figure.savefig(OUTPUTS[6], format="pdf", bbox_inches="tight"); plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.8))
    for (penalty, weight), marker in styles.items():
        subset = results[(results["penalty"] == penalty) & (results["class_weight"].isna() if weight is None else results["class_weight"].eq(weight))].sort_values("C")
        label = f"{penalty.upper()} / {weight or 'None'}"
        axis.plot(subset["C"], subset["train_roc_auc_mean"], marker[0] + "--", alpha=0.75, label=f"Train — {label}")
        axis.plot(subset["C"], subset["validation_roc_auc_mean"], marker, label=f"Validation — {label}")
    axis.set_xscale("log")
    axis.set(title="Train vs Validation ROC-AUC Across Regularization", xlabel="Inverse regularization strength C (log scale)", ylabel="Mean ROC-AUC")
    axis.grid(alpha=0.25); axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout(); figure.savefig(OUTPUTS[7], format="pdf", bbox_inches="tight"); plt.close(figure)


def build_decision_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = {
        "roc_auc": "best grid-search ROC-AUC",
        "average_precision": "best grid-search Average Precision",
        "f1": "best grid-search F1 at default threshold",
        "balanced_accuracy": "best grid-search Balanced Accuracy at default threshold",
    }
    for metric, reason in labels.items():
        row = results.sort_values([f"rank_{metric}", "candidate_id"]).iloc[0]
        rows.append({
            "configuration": _label(row), "penalty": row["penalty"], "C": row["C"],
            "class_weight": row["class_weight"], "ROC-AUC": row["validation_roc_auc_mean"],
            "Average Precision": row["validation_average_precision_mean"], "Precision": row["validation_precision_mean"],
            "Recall": row["validation_recall_mean"], "F1": row["validation_f1_mean"],
            "Accuracy": row["validation_accuracy_mean"], "Balanced Accuracy": row["validation_balanced_accuracy_mean"],
            "Generalization Gap": row["roc_auc_generalization_gap"], "Fit Time": row["mean_fit_time"], "selection_reason": reason,
        })
    for experiment_id in ("M01-LR-001", "M01-LR-002"):
        summary = json.loads((PROJECT_ROOT / f"reports/experiments/{experiment_id}_summary.json").read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        parameters = summary["estimator_parameters"]
        rows.append({
            "configuration": experiment_id, "penalty": parameters["classifier__penalty"], "C": parameters["classifier__C"],
            "class_weight": parameters["classifier__class_weight"], "ROC-AUC": metrics["roc_auc"]["validation_mean"],
            "Average Precision": metrics["average_precision"]["validation_mean"], "Precision": metrics["precision"]["validation_mean"],
            "Recall": metrics["recall"]["validation_mean"], "F1": metrics["f1"]["validation_mean"],
            "Accuracy": metrics["accuracy"]["validation_mean"], "Balanced Accuracy": metrics["balanced_accuracy"]["validation_mean"],
            "Generalization Gap": metrics["roc_auc"]["train_mean"] - metrics["roc_auc"]["validation_mean"],
            "Fit Time": summary["fit_time_mean"], "selection_reason": "registered reference experiment",
        })
    return pd.DataFrame(rows)


def main() -> None:
    refuse_existing_outputs()
    X, y, _ = load_dataset(optimize_memory=True)
    X_train, X_reserved, y_train, y_reserved = create_train_test_split(X, y)
    train_hash, test_hash = split_fingerprint(X_train.index), split_fingerprint(X_reserved.index)
    if train_hash != EXPECTED_TRAIN_FINGERPRINT or test_hash != EXPECTED_TEST_FINGERPRINT:
        raise RuntimeError(f"Shared split fingerprint mismatch: train={train_hash}, test={test_hash}")
    del X_reserved, y_reserved, X, y

    pipeline = create_logistic_regression_pipeline(
        penalty="l2", C=1.0, class_weight=None, solver="saga", max_iter=2000, random_state=42
    )
    grid = create_logistic_parameter_grid()
    search = create_grid_search(pipeline, parameter_grid=grid, refit="roc_auc")
    candidates, folds = 20, 5
    print(f"Planned search: {candidates} candidates, {folds} folds, {candidates * folds} fits")
    print(f"Training dimensions: {X_train.shape[0]} rows × {X_train.shape[1]} features")
    print(f"n_jobs: {search.n_jobs}")

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(X_train, y_train)
    duration = time.perf_counter() - started
    convergence_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    for message in convergence_messages:
        warnings.warn(message, ConvergenceWarning, stacklevel=2)

    results = grid_search_results_to_dataframe(search)
    summary = summarize_grid_search(
        search, results, search_id=SEARCH_ID, member="Member 01", branch="feature/data_processing"
    )
    summary.update({
        "n_samples": int(X_train.shape[0]), "n_features": int(X_train.shape[1]),
        "n_jobs": search.n_jobs, "duration_seconds": duration,
        "mean_seconds_per_candidate": duration / candidates,
        "mean_seconds_per_fit": duration / (candidates * folds),
        "convergence_warning_count": len(convergence_messages),
        "convergence_warning_messages": sorted(set(convergence_messages)),
        "convergence_configurations": [],
        "convergence_configuration_note": "Warnings are emitted per parallel fit and cannot be mapped reliably to a candidate by GridSearchCV.",
        "max_iter": 2000, "solver": "saga",
        "train_fingerprint": train_hash, "reserved_test_fingerprint_verified_only": test_hash,
    })
    save_grid_search_results(results, summary)
    top = results.sort_values(["rank_roc_auc", "candidate_id"]).head(10).copy()
    _write_table(top, OUTPUTS[2], OUTPUTS[3])
    decision = build_decision_table(results)
    _write_table(decision, OUTPUTS[4])
    save_figures(results, top)

    print(f"Total duration: {duration:.2f} seconds")
    print(f"ConvergenceWarning count: {len(convergence_messages)}")
    print(f"Best ROC-AUC: {search.best_score_:.9f}")
    print(f"Best parameters: {search.best_params_}")
    print("\nTop 10 by ROC-AUC:")
    print(top[["candidate_id", "penalty", "C", "class_weight", "rank_roc_auc", "validation_roc_auc_mean", "validation_average_precision_mean", "validation_f1_mean", "validation_balanced_accuracy_mean"]].to_string(index=False))
    for metric in ("average_precision", "f1", "balanced_accuracy"):
        winner = results.sort_values([f"rank_{metric}", "candidate_id"]).iloc[0]
        print(f"Best {metric}: {_label(winner)} = {winner[f'validation_{metric}_mean']:.9f}")
    print("Final test partition was fingerprint-verified only; score and predict were never called on it.")


def finalize_existing_results() -> None:
    """Finish post-processing after a fit-complete serialization interruption."""
    candidates_path, summary_path = OUTPUTS[:2]
    if not candidates_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Complete candidate and summary files are required for finalization.")
    results = pd.read_csv(candidates_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or len(results) != 20:
        raise ValueError("Only a completed 20-candidate search may be finalized.")
    top = results.sort_values(["rank_roc_auc", "candidate_id"]).head(10).copy()
    if not OUTPUTS[2].exists():
        top.to_csv(OUTPUTS[2], index=False, encoding="utf-8")
    if not OUTPUTS[3].exists():
        records = json.loads(top.to_json(orient="records"))
        OUTPUTS[3].write_text(
            json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not OUTPUTS[4].exists():
        _write_table(build_decision_table(results), OUTPUTS[4])
    if any(not path.exists() for path in OUTPUTS[5:]):
        if any(path.exists() for path in OUTPUTS[5:]):
            raise FileExistsError("Figure set is partial; remove only partial search figures manually.")
        save_figures(results, top)
    print(f"Finalized saved {SEARCH_ID} without rerunning GridSearchCV.")


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing_results()
    else:
        main()

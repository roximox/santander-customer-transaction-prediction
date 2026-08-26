"""Leakage-safe Feature Selection and PCA pipelines and reporting helpers for Member 03.

All pipeline factories in this module construct unfitted scikit-learn objects only.
They never load data, call ``fit``, run cross-validation, access the final test
partition, or persist models and experiment results.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; prevents Qt crash in CI and test environments.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import get_project_root, load_config

FS_EXPERIMENT_ID = "M03-FS-001"
FS_MODEL_NAME = "L1 Feature Selection + Logistic Regression"
PCA_EXPERIMENT_ID = "M03-PCA-001"
PCA_MODEL_NAME = "PCA + Logistic Regression"

COMPARISON_COLUMNS = (
    "experiment_id",
    "model_name",
    "roc_auc_mean",
    "roc_auc_std",
    "average_precision_mean",
    "average_precision_std",
    "f1_mean",
    "f1_std",
    "precision_mean",
    "recall_mean",
    "accuracy_mean",
    "balanced_accuracy_mean",
    "fit_time_mean",
    "score_time_mean",
    "train_roc_auc_mean",
    "validation_roc_auc_mean",
    "roc_auc_generalization_gap",
)


def _configured_random_state() -> int:
    """Return the validated project random state from shared configuration."""
    value = load_config().get("project", {}).get("random_state")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("project.random_state must be an integer.")
    return int(value)


def create_feature_selection_pipeline() -> Pipeline:
    """Return an unfitted StandardScaler → SelectFromModel(L1) → LogisticRegression(L2) Pipeline.

    The L1-penalized estimator inside SelectFromModel shrinks uninformative
    feature coefficients to exactly zero, dropping them before the L2 classifier
    is trained. Scaling is learned inside each cross-validation training fold to
    prevent leakage. These parameters are untuned baseline starting points.
    """
    random_state = _configured_random_state()
    selector = SelectFromModel(
        LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.1,
            max_iter=1000,
            random_state=random_state,
        )
    )
    classifier = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("classifier", classifier),
        ]
    )


def create_pca_pipeline() -> Pipeline:
    """Return an unfitted StandardScaler → PCA(95% variance) → LogisticRegression(L2) Pipeline.

    PCA is sensitive to feature scales, so StandardScaler is applied first.
    ``n_components=0.95`` automatically selects the minimum number of components
    that preserve 95% of the total variance. Scaling and PCA are learned inside
    each cross-validation training fold only. These parameters are untuned
    baseline starting points.
    """
    random_state = _configured_random_state()
    pca = PCA(n_components=0.95, random_state=random_state)
    classifier = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", pca),
            ("classifier", classifier),
        ]
    )


def build_fs_pca_comparison(
    summaries: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a common metric comparison table from FS and PCA experiment summaries."""
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        metrics = summary["metrics"]
        train_auc = metrics["roc_auc"].get("train_mean")
        validation_auc = metrics["roc_auc"]["validation_mean"]
        if train_auc is None:
            raise ValueError("Summary must contain train ROC-AUC for gap calculation.")
        rows.append(
            {
                "experiment_id": summary["experiment_id"],
                "model_name": summary["model_name"],
                "roc_auc_mean": validation_auc,
                "roc_auc_std": metrics["roc_auc"]["validation_std"],
                "average_precision_mean": metrics["average_precision"]["validation_mean"],
                "average_precision_std": metrics["average_precision"]["validation_std"],
                "f1_mean": metrics["f1"]["validation_mean"],
                "f1_std": metrics["f1"]["validation_std"],
                "precision_mean": metrics["precision"]["validation_mean"],
                "recall_mean": metrics["recall"]["validation_mean"],
                "accuracy_mean": metrics["accuracy"]["validation_mean"],
                "balanced_accuracy_mean": metrics["balanced_accuracy"]["validation_mean"],
                "fit_time_mean": summary["fit_time_mean"],
                "score_time_mean": summary["score_time_mean"],
                "train_roc_auc_mean": float(train_auc),
                "validation_roc_auc_mean": float(validation_auc),
                "roc_auc_generalization_gap": float(train_auc) - float(validation_auc),
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _project_path(path: str | Path) -> Path:
    """Resolve a path to an absolute location inside the project root."""
    root = get_project_root().resolve()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Feature selection outputs must remain inside the project."
        ) from exc
    return resolved


def _has_absolute_path(value: Any) -> bool:
    """Detect absolute POSIX or Windows paths nested in report content."""
    if isinstance(value, str):
        return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_has_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_absolute_path(item) for item in value)
    return False


def save_fs_pca_comparison(
    comparison: pd.DataFrame,
    *,
    csv_path: str | Path = "reports/tables/feature_selection_pca_comparison.csv",
    json_path: str | Path = "reports/tables/feature_selection_pca_comparison.json",
) -> tuple[Path, Path]:
    """Save FS vs PCA comparison as CSV and strict JSON without overwriting."""
    missing = set(COMPARISON_COLUMNS) - set(comparison.columns)
    if missing:
        raise ValueError(f"Comparison lacks required columns: {sorted(missing)}.")
    csv_file, json_file = _project_path(csv_path), _project_path(json_path)
    if csv_file.exists() or json_file.exists():
        raise FileExistsError("Feature selection and PCA comparison already exists.")
    records = comparison.loc[:, COMPARISON_COLUMNS].to_dict(orient="records")
    if _has_absolute_path(records):
        raise ValueError("Comparison content must not contain absolute paths.")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    comparison.loc[:, COMPARISON_COLUMNS].to_csv(csv_file, index=False)
    json_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_file, json_file


def save_fs_cv_figure(
    fold_results: pd.DataFrame,
    path: str | Path = "reports/figures/feature_selection_cv_scores.pdf",
) -> Path:
    """Plot Feature Selection validation ROC-AUC and Average Precision by fold."""
    output = _project_path(path)
    if output.exists():
        raise FileExistsError(f"Figure already exists: {output.name}.")
    required = {"fold", "validation_roc_auc", "validation_average_precision"}
    if not required <= set(fold_results.columns):
        raise ValueError(f"Fold results lack required columns: {sorted(required)}.")
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(
        fold_results["fold"], fold_results["validation_roc_auc"],
        marker="o", label="ROC-AUC",
    )
    axis.plot(
        fold_results["fold"], fold_results["validation_average_precision"],
        marker="s", label="Average Precision",
    )
    axis.set_title("L1 Feature Selection + Logistic Regression — Validation Scores by Fold")
    axis.set_xlabel("Cross-validation fold")
    axis.set_ylabel("Validation score")
    axis.set_xticks(fold_results["fold"])
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output


def save_pca_cv_figure(
    fold_results: pd.DataFrame,
    path: str | Path = "reports/figures/pca_cv_scores.pdf",
) -> Path:
    """Plot PCA validation ROC-AUC and Average Precision by fold."""
    output = _project_path(path)
    if output.exists():
        raise FileExistsError(f"Figure already exists: {output.name}.")
    required = {"fold", "validation_roc_auc", "validation_average_precision"}
    if not required <= set(fold_results.columns):
        raise ValueError(f"Fold results lack required columns: {sorted(required)}.")
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(
        fold_results["fold"], fold_results["validation_roc_auc"],
        marker="o", label="ROC-AUC",
    )
    axis.plot(
        fold_results["fold"], fold_results["validation_average_precision"],
        marker="s", label="Average Precision",
    )
    axis.set_title("PCA + Logistic Regression — Validation Scores by Fold")
    axis.set_xlabel("Cross-validation fold")
    axis.set_ylabel("Validation score")
    axis.set_xticks(fold_results["fold"])
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output


def save_fs_vs_pca_figure(
    comparison: pd.DataFrame,
    path: str | Path = "reports/figures/feature_selection_vs_pca_metrics.pdf",
) -> Path:
    """Plot FS and PCA validation metrics side by side from zero to one."""
    output = _project_path(path)
    if output.exists():
        raise FileExistsError(f"Figure already exists: {output.name}.")
    indexed = comparison.set_index("experiment_id")
    plot = indexed[
        ["roc_auc_mean", "average_precision_mean", "accuracy_mean", "balanced_accuracy_mean"]
    ].copy()
    plot.index = ["L1 Feature Selection", "PCA 95%"]
    plot.columns = ["ROC-AUC", "Average Precision", "Accuracy", "Balanced Accuracy"]
    figure, axis = plt.subplots(figsize=(10, 5.8))
    plot.plot(kind="bar", ax=axis, width=0.78)
    axis.set_title("L1 Feature Selection vs PCA — Validation Metrics")
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean validation score")
    axis.set_ylim(0, 1)
    axis.legend(title="Metric", frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output

"""Training-only convergence, sparsity, and coefficient-stability analysis."""

from __future__ import annotations

import json
import inspect
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import get_project_root
from src.evaluation import get_scoring_metrics
from src.validation import create_stratified_cv

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "LR-SELECTED-ROC": {"regularization": "l2", "l1_ratio": 0.0, "C": 0.01, "class_weight": None},
    "LR-SELECTED-AP": {"regularization": "l1", "l1_ratio": 1.0, "C": 0.1, "class_weight": None},
    "LR-SELECTED-BALANCED": {"regularization": "l2", "l1_ratio": 0.0, "C": 0.01, "class_weight": "balanced"},
    "LR-L1-WEAK-REG": {"regularization": "l1", "l1_ratio": 1.0, "C": 100.0, "class_weight": None},
}
EPSILON = float(np.finfo(float).eps)


def create_coefficient_audit_pipeline(
    *, l1_ratio: float, C: float, class_weight: str | None, max_iter: int = 2000
) -> Pipeline:
    """Build an equivalent scaler/logistic Pipeline across sklearn APIs."""
    if l1_ratio not in (0.0, 1.0):
        raise ValueError("This audit supports only l1_ratio=0 (L2) or 1 (L1).")
    logistic_parameters: dict[str, Any] = {
        "C": C, "class_weight": class_weight, "solver": "saga",
        "max_iter": max_iter, "random_state": 42,
    }
    penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
    if penalty_default == "deprecated":
        # scikit-learn 1.8+: l1_ratio replaces the removed penalty choice.
        logistic_parameters["l1_ratio"] = l1_ratio
    else:
        # scikit-learn <=1.7 ignores l1_ratio unless penalty='elasticnet'.
        logistic_parameters["penalty"] = "l1" if l1_ratio == 1.0 else "l2"
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**logistic_parameters)),
    ])


def audit_configurations_by_fold(
    X: pd.DataFrame, y: pd.Series, *, cv: Any | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit exactly four configurations on five training folds, never a test set."""
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    resolved_cv = create_stratified_cv() if cv is None else cv
    scorers = get_scoring_metrics()
    audit_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for configuration_id, parameters in CONFIGURATIONS.items():
        for fold, (train_positions, validation_positions) in enumerate(resolved_cv.split(X, y), 1):
            X_fold_train, y_fold_train = X.iloc[train_positions], y.iloc[train_positions]
            X_validation, y_validation = X.iloc[validation_positions], y.iloc[validation_positions]
            pipeline = create_coefficient_audit_pipeline(
                l1_ratio=parameters["l1_ratio"], C=parameters["C"],
                class_weight=parameters["class_weight"],
            )
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(X_fold_train, y_fold_train)
            fit_time = time.perf_counter() - started
            convergence = [
                str(item.message) for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            classifier = pipeline.named_steps["classifier"]
            n_iter = int(np.max(classifier.n_iter_))
            scores = {
                metric: float(scorer(pipeline, X_validation, y_validation))
                for metric, scorer in scorers.items()
            }
            audit_rows.append({
                "configuration_id": configuration_id, "fold": fold,
                "regularization": parameters["regularization"], "l1_ratio": parameters["l1_ratio"],
                "C": parameters["C"], "class_weight": parameters["class_weight"],
                "convergence_warning": bool(convergence),
                "convergence_warning_messages": " | ".join(convergence),
                "n_iter": n_iter, "max_iter": int(classifier.max_iter),
                "solver": classifier.solver, "fit_time": fit_time,
                **{f"validation_{metric}": value for metric, value in scores.items()},
            })
            coefficients = np.asarray(classifier.coef_).reshape(-1)
            if len(coefficients) != X.shape[1]:
                raise ValueError("Coefficient count does not match feature count.")
            converged = not convergence
            for feature, coefficient in zip(X.columns, coefficients):
                coefficient = float(coefficient)
                coefficient_rows.append({
                    "configuration_id": configuration_id, "fold": fold,
                    "feature": str(feature), "coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                    "sign": int(np.sign(coefficient)), "is_zero": coefficient == 0.0,
                    "n_iter": n_iter, "converged": converged,
                })
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
    return pd.DataFrame(audit_rows), pd.DataFrame(coefficient_rows)


def summarize_coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact-zero selection and sign stability across folds."""
    rows: list[dict[str, Any]] = []
    for (configuration_id, feature), group in coefficients.groupby(
        ["configuration_id", "feature"], sort=False
    ):
        values = group["coefficient"].to_numpy(dtype=float)
        positive = int((values > 0).sum()); negative = int((values < 0).sum()); zero = int((values == 0).sum())
        majority = max(positive, negative, zero)
        std = float(values.std(ddof=0)); mean = float(values.mean())
        rows.append({
            "configuration_id": configuration_id, "feature": feature,
            "mean_coefficient": mean, "std_coefficient": std,
            "mean_absolute_coefficient": float(np.abs(values).mean()),
            "min_coefficient": float(values.min()), "max_coefficient": float(values.max()),
            "positive_fold_count": positive, "negative_fold_count": negative,
            "zero_fold_count": zero, "sign_consistency": majority / len(values),
            "selection_frequency": (positive + negative) / len(values),
            "stability_ratio": abs(mean) / (std + EPSILON),
        })
    return pd.DataFrame(rows)


def summarize_l1_sparsity(coefficients: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize exact nonzero supports for both audited L1 configurations."""
    rows: list[dict[str, Any]] = []; overall: dict[str, Any] = {}
    for configuration_id in ("LR-SELECTED-AP", "LR-L1-WEAK-REG"):
        subset = coefficients[coefficients["configuration_id"] == configuration_id]
        supports: list[set[str]] = []
        for fold, group in subset.groupby("fold"):
            selected = set(group.loc[~group["is_zero"], "feature"])
            supports.append(selected)
            rows.append({
                "configuration_id": configuration_id, "fold": int(fold),
                "nonzero_coefficient_count": len(selected),
                "selected_feature_percentage": 100.0 * len(selected) / len(group),
            })
        intersection = set.intersection(*supports); union = set.union(*supports)
        frequencies = subset.groupby("feature")["is_zero"].apply(lambda values: float((~values).mean()))
        overall[configuration_id] = {
            "intersection_feature_count": len(intersection), "union_feature_count": len(union),
            "features_selected_all_folds": len(intersection),
            "intersection_features": sorted(intersection), "union_features": sorted(union),
            "selection_frequency": {str(key): float(value) for key, value in frequencies.items()},
        }
    return pd.DataFrame(rows), overall


def _output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = get_project_root() / candidate
    candidate = candidate.resolve()
    candidate.relative_to(get_project_root().resolve())
    return candidate


def save_analysis_figure(figure: Any, path: str | Path) -> Path:
    output = _output_path(path)
    if output.exists():
        raise FileExistsError(f"Figure already exists: {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(output, format="pdf", bbox_inches="tight"); plt.close(figure)
    return output


def create_top_coefficients_figure(stability: pd.DataFrame) -> Any:
    data = stability[stability["configuration_id"] == "LR-SELECTED-ROC"].nlargest(20, "mean_absolute_coefficient").sort_values("mean_coefficient")
    figure, axis = plt.subplots(figsize=(9, 7))
    colors = np.where(data["mean_coefficient"] >= 0, "#2878B5", "#D9534F")
    axis.barh(data["feature"], data["mean_coefficient"], xerr=data["std_coefficient"], color=colors, alpha=0.85, capsize=2)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(title="Top 20 Standardized Logistic Coefficients — LR-SELECTED-ROC", xlabel="Mean coefficient across folds (±1 SD)", ylabel="Feature")
    axis.grid(axis="x", alpha=0.2)
    return figure


def create_stability_figure(stability: pd.DataFrame) -> Any:
    data = stability[stability["configuration_id"] == "LR-SELECTED-ROC"].nlargest(20, "mean_absolute_coefficient").sort_values("mean_coefficient")
    figure, axis = plt.subplots(figsize=(9, 7))
    scatter = axis.scatter(data["mean_coefficient"], data["feature"], s=35 + 100 * data["sign_consistency"], c=data["sign_consistency"], cmap="viridis", vmin=0, vmax=1)
    axis.errorbar(data["mean_coefficient"], data["feature"], xerr=data["std_coefficient"], fmt="none", ecolor="gray", alpha=0.7)
    axis.axvline(0, color="black", linewidth=0.8); figure.colorbar(scatter, ax=axis, label="Sign consistency")
    axis.set(title="Coefficient Stability Across Folds — LR-SELECTED-ROC", xlabel="Mean standardized coefficient (±1 SD)", ylabel="Feature")
    return figure


def create_l1_sparsity_figure(sparsity: pd.DataFrame) -> Any:
    figure, axis = plt.subplots(figsize=(8, 5))
    for configuration_id, group in sparsity.groupby("configuration_id"):
        axis.plot(group["fold"], group["nonzero_coefficient_count"], marker="o", label=configuration_id)
    axis.set(title="Exact L1 Sparsity by Fold", xlabel="Fold", ylabel="Nonzero coefficients", xticks=sorted(sparsity["fold"].unique()))
    axis.grid(alpha=0.25); axis.legend(frameon=False)
    return figure


def create_convergence_figure(audit: pd.DataFrame) -> Any:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for configuration_id, group in audit.groupby("configuration_id"):
        axis.plot(group["fold"], group["n_iter"], marker="o", label=configuration_id)
    max_iter = int(audit["max_iter"].iloc[0])
    observed_max = float(audit["n_iter"].max())
    axis.set(
        title=f"Logistic Regression Iterations by Fold (max_iter={max_iter})",
        xlabel="Fold", ylabel="n_iter", xticks=sorted(audit["fold"].unique()),
        ylim=(0, max(10.0, observed_max * 1.2)),
    )
    axis.text(
        0.99, 0.96, f"All fits remained far below max_iter={max_iter}",
        transform=axis.transAxes, ha="right", va="top", fontsize=9,
    )
    axis.grid(alpha=0.25); axis.legend(frameon=False, fontsize=8)
    return figure


def strict_json_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Return records with pandas missing values represented as JSON null."""
    return json.loads(table.to_json(orient="records"))

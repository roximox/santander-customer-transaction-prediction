"""Cached, defensive loaders for historical project artifacts only.

This module deliberately contains no dataset loader, estimator, evaluation, or
experiment-writing imports. Missing optional artifacts return empty objects with
a concise error stored in ``DataFrame.attrs['error']`` where applicable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from src.model_selection import infer_model_family, normalize_experiment_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_COLUMNS = (
    "experiment_id", "member", "model_name", "model_family", "roc_auc_mean",
    "roc_auc_std", "average_precision_mean", "f1_mean", "precision_mean",
    "recall_mean", "balanced_accuracy_mean", "roc_auc_generalization_gap",
    "fit_time_mean", "source_type", "status", "final_test_used",
)


def _root(root: str | Path | None) -> Path:
    return PROJECT_ROOT if root is None else Path(root)


def _safe_path(relative_path: str | Path, root: str | Path | None = None) -> Path:
    base = _root(root).resolve()
    candidate = (base / relative_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("Dashboard artifact paths must stay inside the project root.") from exc
    return candidate


def _empty(error: str, columns: Iterable[str] = ()) -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(columns))
    frame.attrs["error"] = error
    return frame


@st.cache_data(show_spinner=False)
def load_csv(
    relative_path: str,
    required_columns: tuple[str, ...] = (),
    root: str | Path | None = None,
) -> pd.DataFrame:
    path = _safe_path(relative_path, root)
    if not path.is_file():
        return _empty(f"Artifact not available: {relative_path}", required_columns)
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        return _empty(f"Could not read {relative_path}: {exc}", required_columns)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        return _empty(f"{relative_path} is missing columns: {', '.join(missing)}", required_columns)
    return frame


@st.cache_data(show_spinner=False)
def load_json(relative_path: str, root: str | Path | None = None) -> dict[str, Any]:
    path = _safe_path(relative_path, root)
    if not path.is_file():
        return {"_error": f"Artifact not available: {relative_path}"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"_error": f"Could not read {relative_path}: {exc}"}
    return value if isinstance(value, dict) else {"_error": f"Expected a JSON object: {relative_path}"}


def load_registry(root: str | Path | None = None) -> pd.DataFrame:
    return load_csv(
        "reports/experiments/experiment_registry.csv",
        ("experiment_id", "member", "model_name", "status", "summary_file"),
        root,
    )


def load_experiment_summary(experiment_id: str, root: str | Path | None = None) -> dict[str, Any]:
    return load_json(f"reports/experiments/{experiment_id}_summary.json", root)


def load_fold_results(experiment_id: str, root: str | Path | None = None) -> pd.DataFrame:
    return load_csv(
        f"reports/experiments/{experiment_id}_fold_results.csv",
        ("experiment_id", "fold"),
        root,
    )


@st.cache_data(show_spinner=False)
def load_experiments(root: str | Path | None = None) -> pd.DataFrame:
    base = _root(root)
    directory = base / "reports/experiments"
    if not directory.is_dir():
        return _empty("Experiment directory is not available.", EXPERIMENT_COLUMNS)
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*_summary.json")):
        if any(token in path.name.lower() for token in ("smoke", "synthetic", "technical", "template", "incomplete")):
            continue
        raw = load_json(path.relative_to(base).as_posix(), base)
        if "_error" in raw:
            continue
        try:
            normalized = normalize_experiment_summary({**raw, "summary_file": path.relative_to(base).as_posix()})
        except (TypeError, ValueError):
            continue
        rows.append(normalized)
    if not rows:
        return _empty("No registered experiment summaries are available.", EXPERIMENT_COLUMNS)
    return pd.DataFrame(rows)


def load_model_comparison(eligible_only: bool = True, root: str | Path | None = None) -> pd.DataFrame:
    name = "model_comparison_eligible.csv" if eligible_only else "model_comparison_all.csv"
    return load_csv(
        f"reports/model_selection/{name}",
        ("experiment_id", "member", "model_name", "model_family", "roc_auc_mean"),
        root,
    )


def load_selection_outputs(root: str | Path | None = None) -> dict[str, Any]:
    return {
        "summary": load_json("reports/model_selection/model_selection_summary.json", root),
        "comparability": load_json("reports/model_selection/model_selection_comparability.json", root),
        "coverage": load_csv("reports/model_selection/model_selection_coverage.csv", ("member", "expected_family", "status"), root),
        "decision": load_csv("reports/model_selection/model_selection_decision.csv", ("selection_category", "experiment_id"), root),
    }


def load_dataset_audit(root: str | Path | None = None) -> dict[str, Any]:
    return {
        "audit": load_json("reports/tables/data_audit_summary.json", root),
        "dtype": load_json("reports/tables/dtype_comparison.json", root),
        "split": load_json("reports/tables/train_test_split_summary.json", root),
    }


def load_search_candidates(search_id: str, root: str | Path | None = None) -> pd.DataFrame:
    return load_csv(f"reports/searches/{search_id}_candidates.csv", ("candidate_id",), root)


def load_learning_curves(root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    return {
        "Logistic Regression": load_csv(
            "reports/tables/logistic_learning_curve_summary.csv",
            ("configuration_id", "train_size_mean", "train_roc_auc_mean", "validation_roc_auc_mean"),
            root,
        ),
        "HistGradientBoosting Tuned": load_csv(
            "reports/tables/M04-HGB-learning-curve.csv",
            ("train_size", "train_roc_auc_mean", "validation_roc_auc_mean"),
            root,
        ),
    }


def load_coefficient_artifacts(root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    return {
        "stability": load_csv(
            "reports/tables/logistic_coefficient_stability.csv",
            ("configuration_id", "feature", "mean_coefficient", "sign_consistency"),
            root,
        ),
        "sparsity": load_csv(
            "reports/tables/logistic_l1_sparsity_summary.csv",
            ("configuration_id", "nonzero_coefficient_count"),
            root,
        ),
    }


def filter_experiments(
    frame: pd.DataFrame,
    *,
    members: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
    experiment_ids: Iterable[str] | None = None,
    include_dummy: bool = True,
) -> pd.DataFrame:
    result = frame.copy()
    for column, values in (("member", members), ("model_family", families), ("experiment_id", experiment_ids)):
        selected = list(values or [])
        if selected and column in result:
            result = result[result[column].isin(selected)]
    if not include_dummy and "model_family" in result:
        result = result[result["model_family"].ne("DUMMY")]
    return result.reset_index(drop=True)


def infer_registry_families(registry: pd.DataFrame) -> pd.DataFrame:
    result = registry.copy()
    if not result.empty:
        result["model_family"] = result.apply(infer_model_family, axis=1)
    return result


def selection_status(root: str | Path | None = None) -> str:
    summary = load_selection_outputs(root)["summary"]
    return str(summary.get("selection_status", "not_available"))

"""Read-only, CV-only infrastructure for pre-final multi-model selection.

This module reads recorded metadata and scores.  It deliberately imports no
estimators, data loaders, or training orchestration code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import get_project_root

METRICS = ("roc_auc", "average_precision", "f1", "precision", "recall", "accuracy", "balanced_accuracy")
EXPECTED_FAMILIES = (
    ("Member 01", "LOGISTIC_REGRESSION"), ("Member 02", "RANDOM_FOREST"),
    ("Member 01", "EXTRA_TREES"), ("Member 03", "PCA"),
    ("Member 03", "FEATURE_SELECTION"), ("Member 04", "HIST_GRADIENT_BOOSTING"),
)


def discover_experiment_summaries(experiments_dir: str | Path | None = None) -> list[Path]:
    """Return deterministic non-technical summary paths without reading them."""
    directory = get_project_root() / "reports/experiments" if experiments_dir is None else Path(experiments_dir)
    if not directory.is_dir():
        return []
    paths = []
    for path in directory.glob("*_summary.json"):
        lowered = path.name.lower()
        if not any(token in lowered for token in ("smoke", "synthetic", "technical", "template", "incomplete")):
            paths.append(path)
    return sorted(paths, key=lambda item: item.name)


def load_experiment_summary(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one UTF-8 experiment summary."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Experiment summary does not exist: {source}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON experiment summary {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Experiment summary {source} must contain a JSON object.")
    required = ("experiment_id", "model_name", "member", "primary_metric", "primary_score_mean")
    missing = [name for name in required if name not in value or value[name] in (None, "")]
    if missing:
        raise ValueError(f"Experiment summary {source} lacks required fields: {missing}.")
    value = dict(value)
    value["summary_file"] = _relative_source(source)
    value.setdefault("source_type", "registered_experiment")
    return value


def _relative_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(get_project_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _metric(summary: Mapping[str, Any], name: str, field: str) -> float | None:
    containers = (summary.get("metrics"), summary.get("target_metric_summaries"))
    aliases = {"validation_mean": ("validation_mean", "mean"), "validation_std": ("validation_std", "std"), "train_mean": ("train_mean",)}
    for container in containers:
        if isinstance(container, Mapping) and isinstance(container.get(name), Mapping):
            metric = container[name]
            for key in aliases.get(field, (field,)):
                value = _number(metric.get(key))
                if value is not None:
                    return value
    direct = summary.get(f"{name}_{'mean' if field == 'validation_mean' else 'std' if field == 'validation_std' else field}")
    return _number(direct)


def infer_model_family(summary: Mapping[str, Any]) -> str:
    """Infer a conservative family from explicit identifiers and names."""
    text = " ".join(str(summary.get(key, "")) for key in ("model_name", "experiment_id", "estimator_class")).lower()
    if "dummy" in text: return "DUMMY"
    if "extra trees" in text or "extratrees" in text or "-et-" in text: return "EXTRA_TREES"
    if "random forest" in text or "randomforest" in text or "-rf-" in text: return "RANDOM_FOREST"
    if "histgradient" in text or "hist gradient" in text or "-hgb-" in text: return "HIST_GRADIENT_BOOSTING"
    if "feature selection" in text or "-fs-" in text: return "FEATURE_SELECTION"
    if "pca" in text: return "PCA"
    if "logistic" in text or "-lr-" in text: return "LOGISTIC_REGRESSION"
    return "OTHER"


def normalize_experiment_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize compatible experiment/search shapes without inventing values."""
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping.")
    roc = _metric(summary, "roc_auc", "validation_mean")
    roc_std = _metric(summary, "roc_auc", "validation_std")
    train_roc = _metric(summary, "roc_auc", "train_mean")
    if roc is None and str(summary.get("primary_metric", "")).lower() == "roc_auc":
        roc, roc_std = _number(summary.get("primary_score_mean")), _number(summary.get("primary_score_std"))
    gap = _number(summary.get("roc_auc_generalization_gap"))
    if gap is None and train_roc is not None and roc is not None:
        gap = train_roc - roc
    warning = summary.get("convergence_warning_detected")
    convergence = summary.get("convergence_status")
    if convergence is None and isinstance(warning, bool):
        convergence = "failed" if warning else "converged"
    estimator_parameters = summary.get("estimator_parameters", {})
    pipeline_steps = summary.get("pipeline_steps")
    if pipeline_steps is None and isinstance(estimator_parameters, Mapping) and any(str(key).startswith("classifier__") for key in estimator_parameters):
        pipeline_steps = ["scaler", "classifier"]
    normalized: dict[str, Any] = {
        "experiment_id": summary.get("experiment_id") or summary.get("candidate_id"),
        "member": summary.get("member"), "model_name": summary.get("model_name"),
        "model_family": infer_model_family(summary), "branch": summary.get("branch"),
        "date_utc": summary.get("date_utc"), "n_samples": summary.get("n_samples"),
        "n_features": summary.get("n_features"), "primary_metric": summary.get("primary_metric"),
        "roc_auc_mean": roc, "roc_auc_std": roc_std,
        "train_roc_auc_mean": train_roc, "roc_auc_generalization_gap": gap,
        "fit_time_mean": _number(summary.get("fit_time_mean") if "fit_time_mean" in summary else summary.get("mean_fit_time")),
        "fit_time_std": _number(summary.get("fit_time_std") if "fit_time_std" in summary else summary.get("std_fit_time")),
        "convergence_status": convergence, "estimator_class": summary.get("estimator_class"),
        "pipeline_steps": pipeline_steps, "summary_file": summary.get("summary_file"),
        "source_type": summary.get("source_type", "registered_experiment"),
        "status": summary.get("status"), "n_splits": summary.get("n_splits"),
        "cv_strategy": summary.get("cv_strategy"), "random_state": summary.get("random_state"),
        "target_distribution": summary.get("target_distribution"), "data_source": summary.get("data_source", summary.get("openml_id")),
        "final_test_used": summary.get("final_test_used", False),
    }
    for metric in METRICS[1:]:
        normalized[f"{metric}_mean"] = _metric(summary, metric, "validation_mean")
        normalized[f"{metric}_std"] = _metric(summary, metric, "validation_std")
    normalized["missing_metadata"] = [key for key in ("n_samples", "n_splits", "cv_strategy", "random_state", "target_distribution", "data_source") if normalized.get(key) is None]
    return normalized


def normalize_search_candidate(candidate: Mapping[str, Any], search_summary: Mapping[str, Any], *, summary_file: str) -> dict[str, Any]:
    """Normalize a selected recorded grid-search row as a virtual candidate."""
    candidate_id = str(candidate["candidate_id"])
    weight = candidate.get("class_weight")
    if pd.isna(weight): weight = None
    merged: dict[str, Any] = {
        "experiment_id": f"{search_summary['search_id']}::{candidate_id}",
        "candidate_id": candidate_id, "member": search_summary.get("member"),
        "branch": search_summary.get("branch"), "date_utc": search_summary.get("date_utc"),
        "model_name": f"Logistic Regression L2 C=0.01 ({'balanced' if weight == 'balanced' else 'unweighted'})",
        "estimator_class": "LogisticRegression", "pipeline_steps": search_summary.get("pipeline_steps"),
        "n_samples": search_summary.get("n_samples"), "n_features": search_summary.get("n_features"),
        "n_splits": search_summary.get("n_splits"), "cv_strategy": "StratifiedKFold",
        "random_state": 42, "primary_metric": search_summary.get("primary_metric"),
        "primary_score_mean": candidate.get("validation_roc_auc_mean"),
        "primary_score_std": candidate.get("validation_roc_auc_std"),
        "mean_fit_time": candidate.get("mean_fit_time"), "std_fit_time": candidate.get("std_fit_time"),
        "roc_auc_generalization_gap": candidate.get("roc_auc_generalization_gap"),
        "status": search_summary.get("status"), "final_test_used": search_summary.get("final_test_used", False),
        "summary_file": summary_file, "source_type": "grid_search_candidate",
    }
    merged["target_metric_summaries"] = {
        metric: {"validation_mean": candidate.get(f"validation_{metric}_mean"), "validation_std": candidate.get(f"validation_{metric}_std"), "train_mean": candidate.get(f"train_{metric}_mean")}
        for metric in METRICS
    }
    return normalize_experiment_summary(merged)


def enrich_shared_protocol_metadata(
    summaries: Sequence[Mapping[str, Any]],
    split_metadata: Mapping[str, Any],
    *,
    provenance_file: str,
) -> list[dict[str, Any]]:
    """Fill report-only provenance from the official split when protocols match.

    Recorded experiment summaries are never mutated. Metadata is inherited only
    when a candidate matches the authoritative development dimensions and shared
    random state; incompatible candidates retain their original missing values.
    """
    train_dimensions = split_metadata.get("train_dimensions")
    if not isinstance(train_dimensions, Sequence) or len(train_dimensions) != 2:
        raise ValueError("Official split metadata must contain train_dimensions.")
    target_distribution = split_metadata.get("train_target_distribution")
    openml_id = split_metadata.get("openml_id")
    random_state = split_metadata.get("random_state")
    if not isinstance(target_distribution, Mapping) or openml_id is None or random_state is None:
        raise ValueError("Official split metadata lacks provenance fields.")

    enriched: list[dict[str, Any]] = []
    for source in summaries:
        item = dict(source)
        matches_shared_split = (
            item.get("n_samples") == train_dimensions[0]
            and item.get("n_features") == train_dimensions[1]
            and item.get("random_state") == random_state
            and item.get("cv_strategy") == "StratifiedKFold"
        )
        inherited: list[str] = []
        if matches_shared_split and item.get("target_distribution") is None:
            item["target_distribution"] = dict(target_distribution)
            inherited.append("target_distribution")
        if matches_shared_split and item.get("data_source") is None:
            item["data_source"] = f"OpenML-{openml_id}"
            inherited.append("data_source")
        item["shared_metadata_provenance"] = provenance_file if inherited else None
        item["inherited_metadata_fields"] = inherited
        item["missing_metadata"] = [
            key for key in ("n_samples", "n_splits", "cv_strategy", "random_state", "target_distribution", "data_source")
            if item.get(key) is None
        ]
        enriched.append(item)
    return enriched


def build_model_comparison_table(normalized_summaries: Sequence[Mapping[str, Any]], *, include_dummy: bool = False) -> pd.DataFrame:
    """Build one non-mutating comparison row per normalized result."""
    table = pd.DataFrame([dict(item) for item in normalized_summaries])
    if table.empty: return table
    if not include_dummy: table = table[table["model_family"].ne("DUMMY")]
    return table.reset_index(drop=True)


def select_eligible_candidates(comparison: pd.DataFrame, *, required_primary_metric: str = "roc_auc") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate eligible candidates and transparent exclusion reasons."""
    seen: set[Any] = set(); eligible, excluded = [], []
    for _, row in comparison.iterrows():
        reasons = []
        experiment_id = row.get("experiment_id")
        if experiment_id in seen: reasons.append("duplicate experiment")
        seen.add(experiment_id)
        if row.get("model_family") == "DUMMY": reasons.append("Dummy baseline")
        if pd.isna(row.get("roc_auc_mean")): reasons.append("missing ROC-AUC")
        if str(row.get("status", "completed")).lower() not in ("completed", "nan", "none"): reasons.append("failed/incomplete experiment")
        if str(row.get("primary_metric", "")).lower() != required_primary_metric: reasons.append("non-comparable CV protocol")
        if bool(row.get("final_test_used", False)): reasons.append("final-test score prohibited")
        if str(row.get("convergence_status", "")).lower() == "failed": reasons.append("convergence failure explicitly known")
        item = row.to_dict()
        if reasons: item["exclusion_reason"] = "; ".join(reasons); excluded.append(item)
        else: eligible.append(item)
    return pd.DataFrame(eligible, columns=comparison.columns), pd.DataFrame(excluded)


def validate_candidate_comparability(candidates: pd.DataFrame) -> dict[str, Any]:
    """Audit protocol metadata, reporting unknown fields as not verifiable."""
    fields = ("n_samples", "n_splits", "cv_strategy", "random_state", "primary_metric", "target_distribution", "data_source")
    checks: dict[str, Any] = {}; incompatible = False; unknown = False
    for field in fields:
        values = [value for value in candidates.get(field, pd.Series(dtype=object)).tolist() if not (value is None or (not isinstance(value, (dict, list)) and pd.isna(value)))]
        canonical = [json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value) for value in values]
        missing = len(values) != len(candidates)
        consistent = len(set(canonical)) <= 1
        checks[field] = {"status": "not verifiable" if missing else "consistent" if consistent else "incompatible", "observed_values": sorted(set(canonical))}
        unknown |= missing; incompatible |= not consistent
    if "final_test_used" in candidates and candidates["final_test_used"].fillna(False).astype(bool).any():
        incompatible = True; checks["final_test_usage"] = {"status": "incompatible", "detail": "At least one candidate reports final-test usage."}
    else: checks["final_test_usage"] = {"status": "consistent", "detail": "No candidate reports final-test usage."}
    status = "incompatible" if incompatible else "partially_comparable" if unknown else "comparable"
    return {"comparability_status": status, "candidate_count": len(candidates), "checks": checks}


def add_model_rankings(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add dense ranks; performance descends and time ascends, with no composite."""
    ranked = candidates.copy(deep=True)
    for metric in ("roc_auc", "average_precision", "f1", "precision", "recall", "balanced_accuracy"):
        ranked[f"rank_{metric}"] = ranked[f"{metric}_mean"].rank(method="min", ascending=False, na_option="bottom")
    ranked["rank_fit_time"] = ranked["fit_time_mean"].rank(method="min", ascending=True, na_option="bottom")
    return ranked


def identify_competitive_candidates(candidates: pd.DataFrame, *, tolerance_std_multiplier: float = 1.0) -> pd.DataFrame:
    """Apply a CV-variability heuristic, not a formal significance test."""
    if tolerance_std_multiplier < 0: raise ValueError("tolerance_std_multiplier must be non-negative.")
    result = candidates.copy(deep=True)
    valid = result.dropna(subset=["roc_auc_mean"])
    if valid.empty: result["roc_auc_competitive"] = False; return result
    best = valid.sort_values(["roc_auc_mean", "experiment_id"], ascending=[False, True]).iloc[0]
    tolerance = (_number(best.get("roc_auc_std")) or 0.0) * tolerance_std_multiplier
    result["roc_auc_competitive"] = result["roc_auc_mean"].ge(float(best["roc_auc_mean"]) - tolerance).fillna(False)
    result["competitive_tolerance"] = tolerance
    result["competitive_method"] = "CV variability heuristic (not a formal statistical test)"
    return result


def build_selection_decision_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return transparent multi-criteria decision rows without a final winner."""
    categories = {
        "best_roc_auc": ("roc_auc_mean", False), "best_average_precision": ("average_precision_mean", False),
        "best_f1": ("f1_mean", False), "best_recall": ("recall_mean", False),
        "best_precision": ("precision_mean", False), "best_balanced_accuracy": ("balanced_accuracy_mean", False),
        "fastest": ("fit_time_mean", True), "lowest_generalization_gap": ("roc_auc_generalization_gap", True),
    }
    rows = []
    for category, (column, ascending) in categories.items():
        valid = candidates.dropna(subset=[column]).sort_values([column, "experiment_id"], ascending=[ascending, True])
        if not valid.empty: rows.append(_decision_row(valid.iloc[0], category, f"Best recorded {column}; transparent single-metric category."))
    for _, row in candidates[candidates.get("roc_auc_competitive", False).astype(bool)].iterrows():
        rows.append(_decision_row(row, "competitive_candidates", "Within one best-model CV standard deviation; heuristic, not a formal test."))
    return pd.DataFrame(rows)


def _decision_row(row: pd.Series, category: str, rationale: str) -> dict[str, Any]:
    return {"selection_category": category, "experiment_id": row.get("experiment_id"), "model_name": row.get("model_name"), "member": row.get("member"), "model_family": row.get("model_family"), "roc_auc": row.get("roc_auc_mean"), "average_precision": row.get("average_precision_mean"), "precision": row.get("precision_mean"), "recall": row.get("recall_mean"), "f1": row.get("f1_mean"), "balanced_accuracy": row.get("balanced_accuracy_mean"), "generalization_gap": row.get("roc_auc_generalization_gap"), "fit_time": row.get("fit_time_mean"), "selection_recommendation": "retain_for_final_group_review", "rationale": rationale}


def build_coverage_table(comparison: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Report expected group coverage without fabricating absent candidates."""
    rows = []
    for member, family in EXPECTED_FAMILIES:
        found = comparison[(comparison["member"] == member) & (comparison["model_family"] == family)]
        accepted = eligible[(eligible["member"] == member) & (eligible["model_family"] == family)]
        status = "available" if len(accepted) else "partially_available" if len(found) else "missing"
        rows.append({"member": member, "expected_family": family, "experiments_found": len(found), "eligible_candidates": len(accepted), "status": status})
    return pd.DataFrame(rows)


def create_selection_figures(candidates: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create four professional figures only from eligible recorded scores."""
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True); paths = []
    labels = candidates["experiment_id"].astype(str)
    for metric, filename, title in (("roc_auc", "final_model_comparison_roc_auc.pdf", "Pre-Final CV ROC-AUC"), ("average_precision", "final_model_comparison_average_precision.pdf", "Pre-Final CV Average Precision")):
        ordered = candidates.dropna(subset=[f"{metric}_mean"]).sort_values(f"{metric}_mean")
        fig, ax = plt.subplots(figsize=(10, max(4, .55 * len(ordered))))
        ax.barh(ordered["experiment_id"], ordered[f"{metric}_mean"], xerr=ordered[f"{metric}_std"].fillna(0), capsize=3)
        ax.set(title=title, xlabel=f"Mean validation {metric.replace('_', ' ').title()}", ylabel="Recorded candidate"); ax.grid(axis="x", alpha=.25); fig.tight_layout()
        path=directory/filename; fig.savefig(path, format="pdf", bbox_inches="tight"); plt.close(fig); paths.append(path)
    metrics=("precision", "recall", "f1", "balanced_accuracy"); plot=candidates.set_index("experiment_id")[[f"{m}_mean" for m in metrics]]; plot.columns=[m.replace("_", " ").title() for m in metrics]
    fig,ax=plt.subplots(figsize=(11,6)); plot.plot(kind="bar",ax=ax); ax.set(title="Pre-Final Validation Threshold Metrics",xlabel="Recorded candidate",ylabel="Mean validation score",ylim=(0,1)); ax.grid(axis="y",alpha=.25); ax.legend(frameon=False); ax.tick_params(axis="x",rotation=15); fig.tight_layout(); path=directory/"final_model_comparison_threshold_metrics.pdf"; fig.savefig(path,format="pdf",bbox_inches="tight"); plt.close(fig); paths.append(path)
    timed=candidates.dropna(subset=["fit_time_mean","roc_auc_mean"]); fig,ax=plt.subplots(figsize=(9,6)); ax.scatter(timed["fit_time_mean"],timed["roc_auc_mean"])
    for _,row in timed.iterrows(): ax.annotate(row["experiment_id"],(row["fit_time_mean"],row["roc_auc_mean"]),xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set(title="Pre-Final CV Performance vs Fit Time",xlabel="Mean fit time (seconds)",ylabel="Mean validation ROC-AUC"); ax.grid(alpha=.25); fig.tight_layout(); path=directory/"final_model_performance_vs_time.pdf"; fig.savefig(path,format="pdf",bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def build_portfolio_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """Create a concise portfolio view of competitive/top recorded candidates."""
    selected = candidates[candidates.get("roc_auc_competitive", False).astype(bool)].copy()
    if selected.empty: selected = candidates.nlargest(min(5, len(candidates)), "roc_auc_mean")
    return pd.DataFrame({"Model": selected["model_name"], "Member": selected["member"], "ROC-AUC mean ± std": selected.apply(lambda r: _mean_std(r,"roc_auc"),axis=1), "Average Precision mean ± std": selected.apply(lambda r: _mean_std(r,"average_precision"),axis=1), "F1": selected["f1_mean"], "Precision": selected["precision_mean"], "Recall": selected["recall_mean"], "Balanced Accuracy": selected["balanced_accuracy_mean"], "Train-validation gap": selected["roc_auc_generalization_gap"], "Fit time": selected["fit_time_mean"]})


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    def cell(value: Any) -> str:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return ""
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", " ")

    columns = [cell(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _mean_std(row: pd.Series, metric: str) -> str:
    mean, std = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
    return "not available" if pd.isna(mean) else f"{mean:.6f} ± {std:.6f}" if not pd.isna(std) else f"{mean:.6f} ± not available"


def validate_relative_output_path(path: str | Path) -> Path:
    """Reject absolute output paths for reproducible reports."""
    value = Path(path)
    if value.is_absolute() or PureWindowsPath(str(path)).is_absolute(): raise ValueError("Output path must be project-relative.")
    return value

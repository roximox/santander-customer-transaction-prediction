"""Synthetic, offline tests for pre-final model selection metadata processing."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from src.model_selection import (
    add_model_rankings, build_coverage_table, build_model_comparison_table,
    build_portfolio_table, build_selection_decision_table,
    dataframe_to_markdown,
    discover_experiment_summaries, identify_competitive_candidates,
    load_experiment_summary, normalize_experiment_summary,
    normalize_search_candidate, select_eligible_candidates,
    validate_candidate_comparability, validate_relative_output_path,
)


def _summary(identifier: str = "M02-RF-001", *, family_name: str = "Random Forest", roc: float = .86) -> dict:
    return {
        "experiment_id": identifier, "model_name": family_name, "member": "Member 02",
        "branch": "feature/rf", "date_utc": "2026-08-11T10:00:00+00:00",
        "n_samples": 160000, "n_features": 200, "n_splits": 5,
        "cv_strategy": "StratifiedKFold", "random_state": 42,
        "primary_metric": "roc_auc", "primary_score_mean": roc,
        "primary_score_std": .003, "fit_time_mean": 2.0, "fit_time_std": .1,
        "status": "completed", "final_test_used": False,
        "target_distribution": {"0": .9, "1": .1}, "data_source": "OpenML-45566",
        "metrics": {name: {"validation_mean": value, "validation_std": .01, "train_mean": value + .01}
                    for name, value in {"roc_auc": roc, "average_precision": .51, "f1": .42,
                    "precision": .4, "recall": .45, "accuracy": .91, "balanced_accuracy": .7}.items()},
        "estimator_class": "RandomForestClassifier", "summary_file": f"reports/experiments/{identifier}_summary.json",
    }


def test_discovery_ignores_smoke_and_is_sorted(tmp_path: Path) -> None:
    for name in ("B_summary.json", "A_summary.json", "smoke_summary.json", "technical_summary.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert [p.name for p in discover_experiment_summaries(tmp_path)] == ["A_summary.json", "B_summary.json"]


def test_load_valid_json_and_reject_invalid_or_missing(tmp_path: Path) -> None:
    valid = tmp_path / "valid_summary.json"; valid.write_text(json.dumps(_summary()), encoding="utf-8")
    assert load_experiment_summary(valid)["experiment_id"] == "M02-RF-001"
    invalid = tmp_path / "invalid_summary.json"; invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        load_experiment_summary(invalid)
    missing = tmp_path / "missing_summary.json"; missing.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="required fields"):
        load_experiment_summary(missing)


def test_normalization_and_missing_values_do_not_mutate_source() -> None:
    source = _summary(); original = deepcopy(source); result = normalize_experiment_summary(source)
    assert result["model_family"] == "RANDOM_FOREST" and result["roc_auc_mean"] == .86
    assert result["roc_auc_generalization_gap"] == pytest.approx(.01)
    assert source == original
    minimal = {"experiment_id": "X", "model_name": "Unknown", "member": "M", "primary_metric": "roc_auc", "primary_score_mean": .5}
    normalized = normalize_experiment_summary(minimal)
    assert normalized["average_precision_mean"] is None and "n_splits" in normalized["missing_metadata"]


def test_dummy_exclusion_default_and_optional_inclusion() -> None:
    normal = normalize_experiment_summary(_summary())
    dummy = normalize_experiment_summary(_summary("M01-DUMMY-001", family_name="DummyClassifier", roc=.5))
    assert len(build_model_comparison_table([normal, dummy])) == 1
    assert len(build_model_comparison_table([normal, dummy], include_dummy=True)) == 2


def test_eligibility_detects_dummy_duplicate_missing_and_failed() -> None:
    normal = normalize_experiment_summary(_summary())
    duplicate = dict(normal)
    dummy = normalize_experiment_summary(_summary("M01-DUMMY-001", family_name="DummyClassifier", roc=.5))
    missing = dict(normal, experiment_id="M03-PCA-001", roc_auc_mean=None)
    failed = dict(normal, experiment_id="M04-HGB-001", status="failed")
    comparison = build_model_comparison_table([normal, duplicate, dummy, missing, failed], include_dummy=True)
    eligible, excluded = select_eligible_candidates(comparison)
    assert eligible["experiment_id"].tolist() == ["M02-RF-001"]
    reasons = " ".join(excluded["exclusion_reason"])
    assert "duplicate experiment" in reasons and "Dummy baseline" in reasons and "missing ROC-AUC" in reasons


def test_rankings_descend_metrics_ascend_time_and_preserve_ties() -> None:
    table = pd.DataFrame([normalize_experiment_summary(_summary("A", roc=.8)), normalize_experiment_summary(_summary("B", roc=.9))])
    table.loc[0, "fit_time_mean"], table.loc[1, "fit_time_mean"] = 1, 2
    ranked = add_model_rankings(table)
    assert ranked.set_index("experiment_id").loc["B", "rank_roc_auc"] == 1
    assert ranked.set_index("experiment_id").loc["A", "rank_fit_time"] == 1
    assert not any("composite" in column for column in ranked.columns)


def test_comparability_compatible_incompatible_and_partial() -> None:
    base = pd.DataFrame([normalize_experiment_summary(_summary("A")), normalize_experiment_summary(_summary("B"))])
    assert validate_candidate_comparability(base)["comparability_status"] == "comparable"
    changed = base.copy(); changed.loc[1, "n_splits"] = 3
    assert validate_candidate_comparability(changed)["comparability_status"] == "incompatible"
    partial = base.copy(); partial.loc[1, "random_state"] = None
    assert validate_candidate_comparability(partial)["comparability_status"] == "partially_comparable"


def test_cv_variability_heuristic_and_decision_categories() -> None:
    candidates = pd.DataFrame([normalize_experiment_summary(_summary("A", roc=.86)), normalize_experiment_summary(_summary("B", roc=.858)), normalize_experiment_summary(_summary("C", roc=.84))])
    candidates = identify_competitive_candidates(add_model_rankings(candidates))
    assert candidates.set_index("experiment_id")["roc_auc_competitive"].to_dict() == {"A": True, "B": True, "C": False}
    decision = build_selection_decision_table(candidates)
    expected = {"best_roc_auc", "best_average_precision", "best_f1", "best_recall", "best_precision", "best_balanced_accuracy", "fastest", "lowest_generalization_gap", "competitive_candidates"}
    assert expected <= set(decision["selection_category"])
    assert decision["rationale"].str.contains("composite", case=False).sum() == 0


def test_coverage_reports_available_and_missing() -> None:
    rf = normalize_experiment_summary(_summary())
    comparison = pd.DataFrame([rf]); coverage = build_coverage_table(comparison, comparison)
    assert coverage.loc[coverage.expected_family.eq("RANDOM_FOREST"), "status"].item() == "available"
    assert coverage.loc[coverage.expected_family.eq("EXTRA_TREES"), "status"].item() == "missing"


def test_search_candidate_is_normalizable() -> None:
    candidate = {"candidate_id": "candidate_002", "class_weight": None, "validation_roc_auc_mean": .859, "validation_roc_auc_std": .003, "validation_average_precision_mean": .508, "mean_fit_time": 3.0}
    search = {"search_id": "M01-LR-SEARCH-001", "member": "Member 01", "n_samples": 160000, "n_features": 200, "n_splits": 5, "primary_metric": "roc_auc", "status": "completed", "final_test_used": False}
    result = normalize_search_candidate(candidate, search, summary_file="reports/searches/s.json")
    assert result["source_type"] == "grid_search_candidate" and result["model_family"] == "LOGISTIC_REGRESSION"


def test_portfolio_csv_markdown_json_and_relative_paths(tmp_path: Path) -> None:
    candidates = identify_competitive_candidates(pd.DataFrame([normalize_experiment_summary(_summary())]))
    portfolio = build_portfolio_table(candidates)
    csv_path, markdown_path, json_path = tmp_path/"p.csv", tmp_path/"p.md", tmp_path/"p.json"
    portfolio.to_csv(csv_path, index=False); markdown_path.write_text(dataframe_to_markdown(portfolio), encoding="utf-8"); json_path.write_text(json.dumps({"rows": json.loads(portfolio.to_json(orient="records"))}), encoding="utf-8")
    assert csv_path.is_file() and "ROC-AUC" in markdown_path.read_text() and json.loads(json_path.read_text())["rows"]
    assert validate_relative_output_path("reports/model_selection/a.csv").as_posix().startswith("reports/")
    with pytest.raises(ValueError): validate_relative_output_path("/tmp/a.csv")


def test_public_api_has_no_test_data_or_training() -> None:
    import src.model_selection as module
    sources = [
        inspect.getsource(module),
        (Path(__file__).resolve().parents[1] / "scripts/build_model_selection_report.py").read_text(encoding="utf-8"),
    ]
    for source in sources:
        assert not any(name in source for name in (
            "X_test", "y_test", "load_dataset", "fetch_openml", ".fit(",
            "requests.", "urllib.",
        ))

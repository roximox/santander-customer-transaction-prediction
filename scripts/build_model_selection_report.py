"""Build the CV-only ADA-ML-09 pre-final model-selection report."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "santander-matplotlib"))

from src.model_selection import (  # noqa: E402
    EXPECTED_FAMILIES, add_model_rankings, build_coverage_table,
    build_model_comparison_table, build_portfolio_table,
    build_selection_decision_table, create_selection_figures,
    dataframe_to_markdown,
    discover_experiment_summaries, identify_competitive_candidates,
    load_experiment_summary, normalize_experiment_summary,
    normalize_search_candidate, select_eligible_candidates,
    validate_candidate_comparability,
)

OUTPUT_DIR = PROJECT_ROOT / "reports/model_selection"
FIGURE_DIR = PROJECT_ROOT / "reports/figures"
OUTPUTS = tuple(OUTPUT_DIR / name for name in (
    "model_comparison_all.csv", "model_comparison_eligible.csv",
    "model_comparison_excluded.csv", "model_selection_decision.csv",
    "model_selection_coverage.csv", "model_selection_summary.json",
    "model_selection_comparability.json", "model_comparison_portfolio.csv",
    "model_comparison_portfolio.md", "group_model_selection_notes.md",
)) + tuple(FIGURE_DIR / name for name in (
    "final_model_comparison_roc_auc.pdf", "final_model_comparison_average_precision.pdf",
    "final_model_comparison_threshold_metrics.pdf", "final_model_performance_vs_time.pdf",
))


def refuse_existing_outputs() -> None:
    existing = [path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError("Model-selection outputs already exist and were not overwritten:\n- " + "\n- ".join(existing))


def selected_search_candidates() -> list[dict[str, Any]]:
    """Read exactly the two recorded M01 Logistic selection candidates."""
    summary_path = PROJECT_ROOT / "reports/searches/M01-LR-SEARCH-001_summary.json"
    candidates_path = PROJECT_ROOT / "reports/searches/M01-LR-SEARCH-001_candidates.csv"
    if not summary_path.is_file() or not candidates_path.is_file():
        return []
    search = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidates_path)
    selected = candidates[
        candidates["penalty"].eq("l2") & candidates["C"].eq(.01)
        & (candidates["class_weight"].isna() | candidates["class_weight"].eq("balanced"))
    ].sort_values("candidate_id")
    if len(selected) != 2:
        raise ValueError("Expected exactly the recorded L2/C=0.01 unweighted and balanced candidates.")
    source = summary_path.relative_to(PROJECT_ROOT).as_posix()
    return [normalize_search_candidate(row.to_dict(), search, summary_file=source) for _, row in selected.iterrows()]


def _strict(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, list): return [_strict(item) for item in value]
    if isinstance(value, (float,)) and pd.isna(value): return None
    if hasattr(value, "item"): return _strict(value.item())
    return value


def _best_by_metric(candidates: pd.DataFrame) -> dict[str, str | None]:
    result = {}
    for metric in ("roc_auc", "average_precision", "f1", "precision", "recall", "balanced_accuracy"):
        valid = candidates.dropna(subset=[f"{metric}_mean"])
        result[metric] = None if valid.empty else str(valid.sort_values([f"{metric}_mean", "experiment_id"], ascending=[False, True]).iloc[0]["experiment_id"])
    return result


def _notes(eligible: pd.DataFrame, coverage: pd.DataFrame, decision: pd.DataFrame, comparability: dict[str, Any]) -> str:
    available = "\n".join(f"- {row.experiment_id}: {row.model_name} ({row.member})" for row in eligible.itertuples()) or "- None"
    missing = "\n".join(f"- {row.member}: {row.expected_family}" for row in coverage[coverage.status.eq("missing")].itertuples()) or "- None"
    winners = "\n".join(f"- {row.selection_category}: {row.experiment_id}" for row in decision[decision.selection_category.ne("competitive_candidates")].itertuples()) or "- Not available"
    competitive = "\n".join(f"- {row.experiment_id}" for row in eligible[eligible.roc_auc_competitive].itertuples()) or "- None"
    return f"""# Model Selection Meeting Notes

## Available Candidates

{available}

## Missing Candidates

{missing}

## Comparison Protocol

Only recorded training cross-validation results enter this report. ROC-AUC is
the fixed primary metric; AP and threshold metrics preserve relevant trade-offs.
No model is retrained and no arbitrary composite score is calculated.

## Best Candidates by Metric

{winners}

## Competitive Models

{competitive}

The competitive label uses one standard deviation of the best ROC-AUC as a CV
variability heuristic. It is not a formal non-inferiority test.

## Trade-offs

Ranking performance, AP, precision, recall, F1, balanced accuracy,
generalization gap, fit time, simplicity, and candidate purpose must be reviewed
separately. Accuracy alone is insufficient for this imbalanced problem.

## Limitations

Comparability status: `{comparability['comparability_status']}`. Missing protocol
metadata are marked not verifiable. Results from Members 02–04 are currently
absent, so the group comparison is incomplete.

## Questions for the Group

- Which operational error costs and threshold policy should govern the lock?
- Which candidates from Members 02–04 are scientifically retained?
- Are all retained pipelines evaluated with the common CV protocol?

## Decision to be Made

The group must review the completed candidate set and lock exactly one pipeline.
No final model or group decision is recorded by this framework.

## Final Test Rule

The final test set must remain untouched until the group has selected and locked
one final pipeline.
"""


def main() -> None:
    refuse_existing_outputs()
    paths = discover_experiment_summaries()
    registered = [normalize_experiment_summary(load_experiment_summary(path)) for path in paths]
    normalized = registered + selected_search_candidates()
    comparison = build_model_comparison_table(normalized, include_dummy=True)
    eligible, excluded = select_eligible_candidates(comparison)
    eligible = identify_competitive_candidates(add_model_rankings(eligible))
    comparability = validate_candidate_comparability(eligible)
    decision = build_selection_decision_table(eligible)
    coverage = build_coverage_table(comparison, eligible)
    missing = coverage.loc[coverage.status.ne("available"), "expected_family"].tolist()
    selection_status = "waiting_for_additional_models" if missing else "ready_for_group_review"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUTS[0], index=False, encoding="utf-8")
    eligible.to_csv(OUTPUTS[1], index=False, encoding="utf-8")
    excluded.to_csv(OUTPUTS[2], index=False, encoding="utf-8")
    decision.to_csv(OUTPUTS[3], index=False, encoding="utf-8")
    coverage.to_csv(OUTPUTS[4], index=False, encoding="utf-8")
    summary = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "number_of_discovered_experiments": len(paths),
        "number_of_eligible_candidates": len(eligible),
        "number_of_excluded_candidates": len(excluded),
        "members_represented": sorted(eligible.member.dropna().unique().tolist()),
        "model_families_represented": sorted(eligible.model_family.dropna().unique().tolist()),
        "best_by_metric": _best_by_metric(eligible),
        "competitive_candidate_ids": eligible.loc[eligible.roc_auc_competitive, "experiment_id"].tolist(),
        "comparability_status": comparability["comparability_status"],
        "missing_expected_model_families": missing,
        "final_test_used": False, "selection_status": selection_status,
        "selection_method": "Multi-criteria review; no composite score and no final winner.",
        "selected_search_candidate_provenance": "Recorded M01-LR-SEARCH-001 candidate rows; historical files unchanged.",
    }
    OUTPUTS[5].write_text(json.dumps(_strict(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    OUTPUTS[6].write_text(json.dumps(_strict(comparability), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    portfolio = build_portfolio_table(eligible)
    portfolio.to_csv(OUTPUTS[7], index=False, encoding="utf-8")
    OUTPUTS[8].write_text("# Pre-Final Model Comparison Portfolio\n\n" + dataframe_to_markdown(portfolio) + "\n\nCV results only; final-test evaluation has not been performed.\n", encoding="utf-8")
    OUTPUTS[9].write_text(_notes(eligible, coverage, decision, comparability), encoding="utf-8")
    create_selection_figures(eligible, FIGURE_DIR)

    print(f"Discovered experiments: {len(paths)}")
    print(f"Eligible candidates: {len(eligible)}")
    print(f"Excluded candidates: {len(excluded)}")
    print(f"Families present: {', '.join(sorted(eligible.model_family.unique()))}")
    print(f"Families missing: {', '.join(missing)}")
    for metric, experiment_id in summary["best_by_metric"].items(): print(f"Best {metric}: {experiment_id}")
    print("Competitive candidates: " + ", ".join(summary["competitive_candidate_ids"]))
    print(f"Comparability status: {comparability['comparability_status']}")
    print(f"Selection status: {selection_status}")
    print("Final test used: false")


if __name__ == "__main__":
    main()

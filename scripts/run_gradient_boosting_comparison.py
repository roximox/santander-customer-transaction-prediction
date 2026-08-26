"""Consolidate completed Member 4 HGB artifacts without any model training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradient_boosting_comparison import (  # noqa: E402
    build_hist_gradient_boosting_comparison,
    load_hist_gradient_boosting_baseline_metrics,
    load_hist_gradient_boosting_tuned_oof_metrics,
    load_hist_gradient_boosting_tuned_parameters,
    save_hist_gradient_boosting_comparison_figure,
    save_hist_gradient_boosting_comparison_summary,
    save_hist_gradient_boosting_comparison_table,
)


def _print_improvement(comparison_label: str, comparison_value: object) -> None:
    """Print baseline, tuned, and absolute change for one comparison row."""
    print(f"{comparison_label} baseline: {comparison_value.baseline_value:.6f}")
    print(f"{comparison_label} tuned: {comparison_value.tuned_value:.6f}")
    print(
        f"{comparison_label} absolute improvement: "
        f"{comparison_value.absolute_change:.6f}"
    )


def main() -> None:
    """Create comparison artifacts exclusively from completed Member 4 results."""
    baseline_metrics = load_hist_gradient_boosting_baseline_metrics()
    tuned_metrics = load_hist_gradient_boosting_tuned_oof_metrics()
    tuned_parameters = load_hist_gradient_boosting_tuned_parameters()
    comparison = build_hist_gradient_boosting_comparison(baseline_metrics, tuned_metrics)

    table_path = save_hist_gradient_boosting_comparison_table(comparison)
    summary_path = save_hist_gradient_boosting_comparison_summary(
        comparison,
        tuned_parameters,
    )
    figure_path = save_hist_gradient_boosting_comparison_figure(comparison)

    print("Member 4 HistGradientBoosting comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    for metric_label in ("ROC-AUC", "Average Precision", "F1"):
        row = comparison.loc[comparison["metric"] == metric_label].iloc[0]
        print()
        _print_improvement(metric_label, row)
    all_metrics_increased = bool((comparison["absolute_change"] > 0).all())
    print(f"\nAll seven compared metrics increased: {all_metrics_increased}")
    if all_metrics_increased:
        print("Tuned HGB improved over the Member 4 baseline on all seven metrics.")
    else:
        print("Tuned HGB did not improve over the Member 4 baseline on every metric.")
    print(f"\nComparison CSV path: {table_path}")
    print(f"Comparison JSON path: {summary_path}")
    print(f"Comparison PDF path: {figure_path}")
    print("Final test partition was not used for this comparison.")


if __name__ == "__main__":
    main()

# Logbook Entry

## Metadata

- Date: 2026-08-30
- Member: Aya Olali
- Sprint: Portfolio and presentation consolidation
- Ticket ID: ADA-PORT-01
- Branch: develop
- Pull Request: Not applicable — individual portfolio drafting
- Time spent: 12 hours (retrospective estimate)
- Related meeting: [2026-08-28 — Final model comparison, project consolidation and presentation planning](../../../meetings/2026-08-28_final-model-comparison-project-consolidation-and-presentation-planning.md)

## Title

Drafting the Member 02 portfolio narrative and evidence selection

## Objective

Organize the Member 02 contribution into a coherent portfolio covering the
scientific question, methodology, EDA, tree experiments, results, limitations,
collaboration, conclusions, and personal contribution.

## Context

The group had completed model selection and moved to consolidation. Individual
portfolios needed to distinguish personal analysis from shared infrastructure
and collective outcomes.

## Work performed

- Reviewed Member 02 notebooks, experiment artifacts, figures, and Logbooks.
- Structured the portfolio from project context through EDA and tree baselines.
- Selected traceable tables and figures rather than reproducing every output.
- Wrote explanations of imbalance, leakage prevention, shared CV, and metrics.
- Drafted the Decision Tree versus Random Forest result interpretation.
- Documented limitations, collaboration, and the collective final-model decision.
- Revised wording to separate individual contribution from group results.
- Checked numerical claims against stored JSON and CSV evidence.

## Methodology

Portfolio claims were grounded in repository-local evidence. The work was
documentation-only: no model refitting, selection change, or final-test access.

## Results

A portfolio narrative was prepared around a reproducible scientific sequence:
data understanding, leakage-safe EDA, explicit tree baselines, shared
cross-validation, quantitative comparison, limitations, and contribution to
collective selection and communication.

## Interpretation

The portfolio shows that Member 02 contributed both scientific analysis and
engineering alignment. It also preserves the distinction between Random Forest
as the strongest Member 02 baseline and HGB as the collective final model.

## Decision

Use only verifiable metrics and figures, retain limitations beside performance
claims, and reference the full Logbook for chronological evidence.

## Difficulties

Condensing a large technical workflow without losing methodological boundaries
or overstating personal ownership required repeated review and restructuring.

## Adaptations and deviations from the plan

The portfolio was organized around evidence and decisions rather than a simple
chronological copy of notebook cells.

## Rejected approaches

Copying all notebook outputs, claiming shared infrastructure as personal work,
presenting final-test evidence as Member 02 development, and omitting negative
or limiting findings were rejected.

## Files changed

- Portfolio draft content and this retrospective Logbook entry.

## Code references

- `notebooks/02_eda.ipynb`
- `notebooks/04_tree_models.ipynb`
- `scripts/run_tree_models.py`

## Figure and table references

- `reports/figures/tree_model_metrics.pdf`
- `reports/tables/tree_model_comparison.csv`
- registered M02 experiment summaries
- collective model-selection reports

## Reproducibility notes

The portfolio uses stored artifacts only. Modeling remained closed and the
final test was neither rerun nor used to revise claims.

## Sources and tools used

Member 02 code and Logbooks, registered experiment artifacts, meeting records,
collective reports, Markdown editing, and peer feedback.

## Next step

Perform a final Logbook and portfolio consistency audit after the last meeting.

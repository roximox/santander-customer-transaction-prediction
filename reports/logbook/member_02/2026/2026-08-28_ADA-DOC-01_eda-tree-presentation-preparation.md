# Logbook Entry

## Metadata

- Date: 2026-08-28
- Member: Aya Olali
- Sprint: Project consolidation and presentation
- Ticket ID: ADA-DOC-01
- Branch: develop
- Pull Request: Committed directly to develop as documentation
- Time spent: 6 hours (retrospective estimate)
- Related meeting: [2026-08-28 — Final model comparison, project consolidation and presentation planning](../../../meetings/2026-08-28_final-model-comparison-project-consolidation-and-presentation-planning.md)

## Title

Preparation of the EDA and tree-model contribution for final communication

## Objective

Convert the detailed Member 02 analysis into a concise, evidence-based story
for the project presentation without overstating individual results or mixing
development and final-test evidence.

## Context

The experimental workflow and collective model selection were complete. The
remaining Member 02 responsibility was to communicate the dataset, EDA
observations, tree-model protocol, comparative results, and limitations in a
form consistent with the shared project narrative.

## Work performed

- Selected the dataset facts needed to introduce the analysis: 200,000 rows,
  200 anonymous numeric features, and a strongly imbalanced target.
- Structured the EDA narrative around training-only analysis, feature
  distributions, weak pairwise correlations, class-conditional differences,
  and descriptive IQR flags.
- Selected ROC-AUC, Average Precision, recall, and balanced accuracy as the
  central tree-model metrics.
- Prepared a direct Decision Tree versus Random Forest comparison.
- Identified the Random Forest train-validation gap and untuned parameters as
  limitations that must remain visible.
- Separated the Member 02 baseline conclusion from the collective final-model
  decision.

## Methodology

All presentation values were transcribed from registered JSON/CSV artifacts and
the existing comparison table. No notebook was rerun and no new result was
calculated from the reserved final-test data.

## Results

The presentation material communicates that Random Forest improved validation
ROC-AUC from 0.633935 to 0.793736 and Average Precision from 0.163973 to
0.369114 relative to Decision Tree. It also explains why accuracy is
insufficient for the imbalanced target and why the Random Forest remained a
baseline rather than the final selected model.

## Interpretation

The clearest Member 02 scientific message is that ensemble trees captured more
predictive structure than a single shallow tree, but stronger project models
were available. Stable cross-validation results support the comparison, while
the train-validation gap limits claims about generalization.

## Decision

Use a compact sequence for final communication: dataset and leakage boundary,
EDA findings, model definitions, shared evaluation protocol, quantitative
comparison, limitations, and connection to the group decision.

## Difficulties

The detailed experiment artifacts contain many metrics and parameters. The main
challenge was reducing them to a short presentation without removing essential
limitations or confusing individual model results with collective outcomes.

## Adaptations and deviations from the plan

The communication focuses on a few defensible findings instead of reproducing
all notebook tables. Hard-coded claims are checked against stored artifacts.

## Rejected approaches

- Presenting accuracy as the main metric.
- Describing the Random Forest as the final project model.
- Showing final-test results as if they belonged to Member 02 development.
- Omitting the Random Forest generalization gap.
- Filling slides with all 200 feature distributions.

## Files changed

- This Logbook entry; presentation content was prepared from existing project
  artifacts.

## Code references

- `notebooks/02_eda.ipynb`
- `notebooks/04_tree_models.ipynb`
- `scripts/run_tree_models.py`

## Figure and table references

- `reports/figures/tree_model_metrics.pdf`
- `reports/tables/tree_model_comparison.csv`
- `reports/experiments/M02-DT-001_summary.json`
- `reports/experiments/M02-RF-001_summary.json`

## Reproducibility notes

Every quantitative statement is traceable to a stored training-only experiment
artifact. The later one-time final-test evaluation is not presented as Member
02 model-development evidence.

## Sources and tools used

Member 02 notebooks, registered experiment artifacts, Matplotlib figure,
meeting notes, and project model-selection documentation.

## Next step

Perform a final consistency review of Member 02 Logbooks, presentation values,
file references, and terminology before submission.

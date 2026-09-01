# Logbook Entry

## Metadata

- Date: 2026-08-23
- Member: Aya Olali
- Sprint: Model progress and evaluation
- Ticket ID: ADA-ANALYSIS-02
- Branch: develop
- Pull Request: Not applicable — result analysis and group discussion
- Time spent: 8 hours (retrospective estimate)
- Related meeting: [2026-08-23 — Model progress, optimization and evaluation](../../../meetings/2026-08-23_model-progress-optimization-and-evaluation.md)

## Title

Tree-result consolidation, scientific interpretation, and colleague discussion

## Objective

Interpret the registered Decision Tree and Random Forest results, compare their
strengths and limitations, and prepare the Member 02 evidence for group-level comparison.

## Context

The two Member 02 training-only experiments were complete and other members
were also consolidating Logistic Regression, PCA, feature-selection, and HGB results.

## Work performed

- Reviewed all five fold scores for both tree models.
- Compared ROC-AUC, Average Precision, F1, recall, and balanced accuracy.
- Calculated and interpreted the Random Forest improvement over Decision Tree.
- Examined train-validation gaps and computational cost.
- Discussed class weighting and threshold-dependent behaviour with colleagues.
- Compared the Member 02 evidence conceptually with other model tracks.
- Drafted conclusions and limitations for the Logbook and later portfolio.

## Methodology

The analysis used registered fold CSV and summary JSON artifacts only. Models
were compared under identical training data, folds, seed, and scoring functions.

## Results

Random Forest achieved validation ROC-AUC 0.793736 ± 0.002392 versus 0.633935
± 0.003119 for Decision Tree. Average Precision improved from 0.163973 to
0.369114. The Random Forest also showed a larger train-validation gap.

## Interpretation

The ensemble captured substantially more predictive signal and remained stable
across folds, but its gap showed that stronger validation performance did not
remove overfitting concerns.

## Decision

Keep Random Forest as the stronger Member 02 baseline and carry both tree
experiments into the collective model comparison with transparent limitations.

## Difficulties

Metrics reflected different aspects of imbalanced classification, while the
Random Forest required much more computation than the shallow Decision Tree.

## Adaptations and deviations from the plan

Interpretation expanded beyond ROC-AUC to Average Precision, recall, balanced
accuracy, stability, and generalization gaps.

## Rejected approaches

Declaring Random Forest optimal, ignoring train scores, evaluating the final
test, or tuning after seeing results under the same experiment IDs were rejected.

## Files changed

- This retrospective analysis and discussion entry only.

## Code references

- `scripts/run_tree_models.py`
- `src/experiments.py`
- `src/evaluation.py`

## Figure and table references

- `reports/tables/tree_model_comparison.csv`
- `reports/figures/tree_model_metrics.pdf`
- `reports/experiments/M02-DT-001_summary.json`
- `reports/experiments/M02-RF-001_summary.json`

## Reproducibility notes

All numerical claims are traceable to registered shared-CV artifacts. The final
test remained closed during analysis and discussion.

## Sources and tools used

Experiment CSV/JSON artifacts, comparison figure, project evaluation framework,
colleague discussion, and meeting record.

## Next step

Contribute the tree evidence to final model selection and individual portfolio work.

# Model Selection Meeting Notes

## Available Candidates

- M01-ET-001: Extra Trees Baseline (Member 01)
- M01-LR-001: Logistic Regression L2 Baseline (Member 01)
- M01-LR-002: Logistic Regression L2 Balanced (Member 01)
- M02-DT-001: Decision Tree (Member 02)
- M02-RF-001: Random Forest (Member 02)
- M03-FS-001: L1 Feature Selection + Logistic Regression (Member 03)
- M03-PCA-001: PCA + Logistic Regression (Member 03)
- M04-HGB-001: HistGradientBoosting Baseline (Member 04)
- M04-HGB-002: HistGradientBoosting Tuned (Member 04)
- M01-LR-SEARCH-001::candidate_002: Logistic Regression L2 C=0.01 (unweighted) (Member 01)
- M01-LR-SEARCH-001::candidate_004: Logistic Regression L2 C=0.01 (balanced) (Member 01)

## Missing Candidates

- None

## Comparison Protocol

Only recorded training cross-validation results enter this report. ROC-AUC is
the fixed primary metric; AP and threshold metrics preserve relevant trade-offs.
No model is retrained and no arbitrary composite score is calculated.

## Best Candidates by Metric

- best_roc_auc: M04-HGB-002
- best_average_precision: M04-HGB-002
- best_f1: M01-ET-001
- best_recall: M01-LR-SEARCH-001::candidate_004
- best_precision: M04-HGB-002
- best_balanced_accuracy: M01-LR-SEARCH-001::candidate_004
- fastest: M01-LR-001
- lowest_generalization_gap: M01-LR-SEARCH-001::candidate_002

## Competitive Models

- M04-HGB-002

The competitive label uses one standard deviation of the best ROC-AUC as a CV
variability heuristic. It is not a formal non-inferiority test.

## Trade-offs

Ranking performance, AP, precision, recall, F1, balanced accuracy,
generalization gap, fit time, simplicity, and candidate purpose must be reviewed
separately. Accuracy alone is insufficient for this imbalanced problem.

## Limitations

Comparability status: `partially_comparable`. Missing protocol
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

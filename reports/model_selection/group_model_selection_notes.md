# Model Selection Meeting Notes

## Available Candidates

- M01-LR-001: Logistic Regression L2 Baseline (Member 01)
- M01-LR-002: Logistic Regression L2 Balanced (Member 01)
- M01-LR-SEARCH-001::candidate_002: Logistic Regression L2 C=0.01 (unweighted) (Member 01)
- M01-LR-SEARCH-001::candidate_004: Logistic Regression L2 C=0.01 (balanced) (Member 01)

## Missing Candidates

- Member 02: RANDOM_FOREST
- Member 02: EXTRA_TREES
- Member 03: PCA
- Member 03: FEATURE_SELECTION
- Member 04: HIST_GRADIENT_BOOSTING

## Comparison Protocol

Only recorded training cross-validation results enter this report. ROC-AUC is
the fixed primary metric; AP and threshold metrics preserve relevant trade-offs.
No model is retrained and no arbitrary composite score is calculated.

## Best Candidates by Metric

- best_roc_auc: M01-LR-SEARCH-001::candidate_002
- best_average_precision: M01-LR-SEARCH-001::candidate_002
- best_f1: M01-LR-SEARCH-001::candidate_004
- best_recall: M01-LR-SEARCH-001::candidate_004
- best_precision: M01-LR-SEARCH-001::candidate_002
- best_balanced_accuracy: M01-LR-SEARCH-001::candidate_004
- fastest: M01-LR-001
- lowest_generalization_gap: M01-LR-SEARCH-001::candidate_002

## Competitive Models

- M01-LR-001
- M01-LR-002
- M01-LR-SEARCH-001::candidate_002
- M01-LR-SEARCH-001::candidate_004

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

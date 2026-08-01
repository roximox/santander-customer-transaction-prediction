# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-05
- Branch: feature/data_processing
- Pull Request: To be updated after Pull Request creation
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI

## Title

Logistic Regression class-weight comparison

## Scientific question

Does balanced class weighting improve positive-class recall, F1, and balanced
accuracy without excessive losses in precision, ROC-AUC, and Average Precision?

## Controlled change

`M01-LR-002`, named `Logistic Regression L2 Balanced`, changed only
`class_weight` from `None` to `"balanced"` relative to `M01-LR-001`. The
`StandardScaler` → `LogisticRegression` Pipeline, L2 penalty, `C=1.0`, `lbfgs`,
`max_iter=1000`, `random_state=42`, shared five folds, split, and metrics were
unchanged. The official split fingerprints matched, and only training data was
evaluated.

## Results and deltas

| Metric | M01-LR-001 | M01-LR-002 | Balanced − unweighted |
|---|---:|---:|---:|
| ROC-AUC | 0.859188 | 0.859011 | -0.000176 |
| Average Precision | 0.507566 | 0.506430 | -0.001135 |
| Precision | 0.688813 | 0.284560 | -0.404253 |
| Recall | 0.272484 | 0.773541 | +0.501057 |
| F1 | 0.390361 | 0.416059 | +0.025698 |
| Accuracy | 0.914481 | 0.781794 | -0.132687 |
| Balanced accuracy | 0.629343 | 0.778128 | +0.148786 |

The balanced model's train ROC-AUC was 0.861823 and validation ROC-AUC was
0.859011, giving a gap of 0.002812. Validation ROC-AUC standard deviation was
0.003121 and Average Precision standard deviation was 0.008519, indicating
similar fold stability to the unweighted baseline. No `ConvergenceWarning` was
detected.

## Interpretation and decision

Balanced weighting greatly increased recall and balanced accuracy, and produced
a smaller F1 gain. The cost was a large precision decrease and a substantial
accuracy decrease. ROC-AUC and Average Precision were effectively stable but
slightly lower, so class weighting changed the operating trade-off rather than
improving ranking discrimination.

`M01-LR-002` is retained as a recall-oriented alternative, not declared globally
better. With ROC-AUC as the primary metric and no documented relative cost for
false negatives and false positives, `M01-LR-001` remains the neutral baseline.
A selection between them should follow an explicit operational cost objective.

## Limits and next step

This comparison does not optimize a threshold, tune regularization, calibrate
probabilities, or establish deployment costs. A future experiment should first
define the desired precision-recall or cost trade-off under a new ID. The final
test partition remains closed.

## Decision

Retain M01-LR-002 as a recall-oriented alternative while M01-LR-001 remains the
neutral baseline until operational error costs are defined.

## Difficulties

Balanced weighting improves recall but sharply lowers precision, so no single
threshold metric gives a sufficient decision rule.

## Adaptations and deviations from the plan

Only `class_weight` changed, preserving a controlled comparison.

## Rejected approaches

Declaring the balanced model globally superior, tuning the threshold, and using
the final test partition were rejected.

## Files changed

- `src/logistic_class_weight.py`
- `scripts/run_logistic_class_weight_comparison.py`
- `tests/test_logistic_class_weight.py`

## Code references

Controlled comparison and output builders in `src/logistic_class_weight.py`.

## Figure and table references

- `reports/experiments/M01-LR-002_fold_results.csv`
- `reports/experiments/M01-LR-002_summary.json`
- `reports/tables/logistic_class_weight_comparison.csv`
- `reports/figures/logistic_class_weight_metrics.pdf`
- `reports/figures/logistic_class_weight_cv.pdf`

## Reproducibility notes

The shared split, five folds, Pipeline, and seed match M01-LR-001. The final
test set remained closed.

## Sources and tools used

scikit-learn, pandas, Matplotlib, pytest, Python, and the experiment registry.

## Next step

Compare predeclared regularization and class-weight configurations by CV only.

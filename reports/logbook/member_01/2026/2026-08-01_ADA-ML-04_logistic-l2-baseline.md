# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-04
- Branch: feature/data_processing
- Pull Request: To be updated after Pull Request creation
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI

## Title

Logistic Regression L2 baseline

## Scientific question

Does an untuned L2-regularized linear model learn discriminative signal beyond
the registered naive baselines under the shared cross-validation protocol?

## Pipeline and protocol

Experiment `M01-LR-001`, named `Logistic Regression L2 Baseline`, used the
shared factory to create `StandardScaler` followed by `LogisticRegression`.
Scaling is necessary for comparable feature scales during coefficient
optimization and remained inside the Pipeline, so it was learned separately in
each fold. L2 and `C=1.0` define a baseline rather than an optimized setting.

Parameters were `penalty="l2"`, `C=1.0`, `class_weight=None`, `solver="lbfgs"`,
`max_iter=1000`, `random_state=42`, `with_mean=True`, and `with_std=True`. The
official train and reserved-test fingerprints were verified. Only the 160,000
training rows entered five-fold stratified cross-validation; the test partition
was not evaluated.

## Results

| Metric | Train mean | Validation mean | Validation std |
|---|---:|---:|---:|
| ROC-AUC | 0.861525 | 0.859188 | 0.003239 |
| Average Precision | 0.513131 | 0.507566 | 0.008490 |
| F1 | 0.395235 | 0.390361 | 0.002371 |
| Precision | 0.693796 | 0.688813 | 0.018405 |
| Recall | 0.276325 | 0.272484 | 0.004297 |
| Accuracy | 0.915025 | 0.914481 | 0.000830 |
| Balanced accuracy | 0.631351 | 0.629343 | 0.001565 |

Validation ROC-AUC by fold was 0.858534, 0.857496, 0.865475, 0.858185, and
0.856249. Its standard deviation of 0.003239 indicates stable performance over
these folds. The mean train-minus-validation ROC-AUC gap was 0.002337.

## Dummy comparison

Relative to `M01-DUMMY-001`, ROC-AUC increased absolutely by 0.359188, Average
Precision increased absolutely by 0.407078 and relatively by 4.051032
(405.10%), and balanced accuracy increased by 0.129343. Logistic mean fold fit
time was 0.644985 seconds, 3.739 times the recorded majority Dummy fit time.

## Convergence

No `ConvergenceWarning` was detected with `lbfgs` and `max_iter=1000`. The
recorded experiment was not modified after evaluation.

## Interpretation and limitations

The baseline learns substantially more discriminative signal than the Dummy
references, but it is not established as optimal. The default decision
threshold still gives recall 0.272484, and this ticket did not tune thresholds,
regularization, class weights, calibration, or any other hyperparameter. No
causal interpretation is made.

## Outputs and next step

Fold results and the summary are registered under `M01-LR-001`. Comparison
tables are in `reports/tables/`, and the metric and fold figures are in
`reports/figures/`. A later experiment may assess one explicitly motivated
alternative under a new ID; the final test partition remains closed.

## Decision

Keep M01-LR-001 as the neutral learned baseline; it is not an optimized or final
model.

## Difficulties

Class imbalance makes accuracy insufficient and the default threshold produces
limited positive recall despite strong ranking metrics.

## Adaptations and deviations from the plan

Scaling was placed inside the Pipeline so every fold learns its own parameters.

## Rejected approaches

Global scaling, threshold tuning, test-set comparison, and declaring the
baseline optimal were rejected.

## Files changed

- `src/logistic_baseline.py`
- `scripts/run_logistic_baseline.py`
- `tests/test_logistic_baseline.py`

## Code references

Experiment and comparison helpers in `src/logistic_baseline.py`.

## Figure and table references

- `reports/experiments/M01-LR-001_fold_results.csv`
- `reports/experiments/M01-LR-001_summary.json`
- `reports/tables/logistic_baseline_comparison.csv`
- `reports/figures/logistic_vs_dummy_metrics.pdf`
- `reports/figures/logistic_cv_scores.pdf`

## Reproducibility notes

The official training fingerprint, five shared folds, and `random_state=42`
were used. The final test set was fingerprint-verified only and remained closed.

## Sources and tools used

scikit-learn, pandas, Matplotlib, pytest, Python, and the shared experiment API.

## Next step

Compare the same model with balanced class weighting under a new experiment ID.

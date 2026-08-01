# Logbook Entry

## Metadata

- Date: 2026-08-01
- Member: Yassine Elhari
- Sprint: Sprint 1
- Ticket ID: ADA-ML-02
- Branch: feature/data_processing
- Pull Request: To be updated after Pull Request creation
- Time spent: TO BE COMPLETED BY YASSINE ELHARI
- Related meeting: TO BE COMPLETED BY YASSINE ELHARI

## Title

DummyClassifier baselines

## Objective

Establish naive reference scores before fitting a model that learns from the
features. Four `DummyClassifier` strategies were compared: `most_frequent`,
`prior`, `stratified`, and `uniform`, registered respectively as
`M01-DUMMY-001` through `M01-DUMMY-004`.

## Protocol

The OpenML features were explicitly converted to `float32`, then the shared
stratified split was recreated. Its train and reserved-test fingerprints matched
the official values. Only the 160,000-row training partition entered the common
five-fold stratified cross-validation. The reserved 40,000-row test partition
was not evaluated. Random strategies used `random_state=42`, and repeated
technical evaluations reproduced the same scores.

Metrics were ROC-AUC, Average Precision, F1, precision, recall, accuracy, and
balanced accuracy. The training target positive prevalence was 0.1005.

## Results

| Strategy | ROC-AUC | Average Precision | F1 | Precision | Recall | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| most_frequent | 0.5000 | 0.1005 | 0.0000 | 0.0000 | 0.0000 | 0.8995 | 0.5000 |
| prior | 0.5000 | 0.1005 | 0.0000 | 0.0000 | 0.0000 | 0.8995 | 0.5000 |
| stratified | 0.4995 | 0.1004 | 0.0994 | 0.0996 | 0.0992 | 0.8194 | 0.4995 |
| uniform | 0.5000 | 0.1005 | 0.1673 | 0.1005 | 0.5002 | 0.4997 | 0.4999 |

## Interpretation

`most_frequent` and `prior` predict the majority class. Their accuracy of
0.8995 reflects target imbalance rather than positive-class detection: recall
and F1 are zero, while balanced accuracy is 0.5000. Their ROC-AUC of 0.5000
indicates chance discrimination. Average Precision of 0.1005 matches the
positive prevalence and is therefore the naive reference level.

The random strategies produce different threshold-dependent accuracy, recall,
and F1 values, but their ROC-AUC and balanced accuracy remain approximately
0.5. None of the Dummy strategies learns discriminative feature information.

## Outputs

- Fold results and summaries: `reports/experiments/M01-DUMMY-00*_*.csv/json`
- Registry: `reports/experiments/experiment_registry.csv`
- Comparison: `reports/tables/dummy_baseline_comparison.csv` and `.json`
- Figure: `reports/figures/dummy_baseline_metrics.pdf`

## Limitations and next step

Dummy classifiers establish metric floors only and cannot model relationships
between customer features and the target. The next step is a separately
identified Logistic Regression experiment using training-fold preprocessing.
No Logistic Regression or final-test evaluation was performed in this ticket.

## Decision

Retain all four registered strategies as metric floors and use the
majority/prior AP of 0.1005 as the imbalance-aware naive reference.

## Difficulties

High majority-class accuracy can appear strong despite zero positive recall, so
the full common metric set was required.

## Adaptations and deviations from the plan

Random baselines received the shared seed; existing experiment IDs are protected
against reruns.

## Rejected approaches

Accuracy-only reporting and evaluation on the reserved test partition were
rejected.

## Files changed

- `src/dummy_baselines.py`
- `scripts/run_dummy_baselines.py`
- `tests/test_dummy_baselines.py`

## Code references

Baseline definitions and comparison builders in `src/dummy_baselines.py`.

## Figure and table references

- `reports/tables/dummy_baseline_comparison.csv`
- `reports/tables/dummy_baseline_comparison.json`
- `reports/figures/dummy_baseline_metrics.pdf`
- `reports/experiments/M01-DUMMY-001_summary.json` through `M01-DUMMY-004_summary.json`

## Reproducibility notes

All results use the official training partition, five stratified folds, and
`random_state=42`. The final test set remained closed.

## Sources and tools used

scikit-learn, pandas, Matplotlib, pytest, Python, and the shared experiment API.

## Next step

Fit the shared Logistic Regression baseline on training folds only.

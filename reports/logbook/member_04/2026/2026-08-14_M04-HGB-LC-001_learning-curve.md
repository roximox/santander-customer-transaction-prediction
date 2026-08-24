# Logbook Entry

## Metadata

Date: 2026-08-14
Member: Chaymae Akouaouch
Sprint: Not confirmed
Ticket ID: M04-HGB-LC-001
Branch: feature/model-optimization
Pull Request: Not created yet
Time spent: Not recorded yet
Related meeting: No related meeting yet

## Title

HistGradientBoosting learning-curve analysis

## Goal

Analyze how the tuned HistGradientBoosting model behaves as the amount of
training data increases, using only the shared training partition and keeping
the reserved final test set untouched.

## Tuned model

The learning curve uses the best configuration found by
`M04-HGB-SEARCH-001`:

- `learning_rate = 0.05`
- `max_iter = 700`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 100`
- `l2_regularization = 10.0`
- `random_state = 42`

This configuration was frozen from the previous randomized search and was not
tuned using the learning-curve results.

## Method

- Reused the common project dataset loader with `load_dataset(optimize_memory=True)`.
- Recreated the common 80/20 train/test split.
- Verified the shared train and reserved-test fingerprints.
- Used only `X_train` and `y_train` for the learning curve; the reserved final
  test partition was not evaluated.
- Reused the common five-fold stratified cross-validation.
- Used training sizes of 10%, 25%, 50%, 75%, and 100%, corresponding to
  effective training sizes of 12,800, 32,000, 64,000, 96,000, and 128,000.
- Analyzed ROC-AUC and Average Precision with `n_jobs=1`.

## Results

| Train size | Train ROC-AUC | Validation ROC-AUC | ROC-AUC gap | Train AP | Validation AP | AP gap |
|---:|---:|---:|---:|---:|---:|---:|
| 12,800 | 0.989269 | 0.851063 | 0.138206 | 0.968960 | 0.475333 | 0.493628 |
| 32,000 | 0.989880 | 0.877188 | 0.112692 | 0.967501 | 0.550490 | 0.417011 |
| 64,000 | 0.983864 | 0.885981 | 0.097883 | 0.936334 | 0.574884 | 0.361450 |
| 96,000 | 0.980122 | 0.889333 | 0.090789 | 0.917675 | 0.584480 | 0.333195 |
| 128,000 | 0.973342 | 0.891245 | 0.082098 | 0.887635 | 0.589899 | 0.297736 |

## Interpretation

Validation ROC-AUC improves from 0.851063 to 0.891245 as training size
increases. Validation Average Precision also improves, from 0.475333 to
0.589899. The ROC-AUC train-validation gap decreases from 0.138206 to
0.082098, and the Average Precision gap decreases from 0.493628 to 0.297736.
The observed learning-curve pattern indicates that additional training data
improves validation performance and reduces the train-validation gap.

A train-validation gap still remains, especially for Average Precision, so this
does not establish that overfitting has disappeared completely. The learning
curve is diagnostic only and was not used for another round of hyperparameter
tuning.

The maximum-training-size learning-curve results are consistent with
`M04-HGB-SEARCH-001`:

| Metric | Randomized-search best | Learning curve at maximum training size |
|---|---:|---:|
| Validation ROC-AUC | 0.891449 | 0.891245 |
| Average Precision | 0.591089 | 0.589899 |
| ROC-AUC generalization gap | 0.082131 | 0.082098 |

The close values provide a consistency check between the search and
learning-curve analyses.

## Implementation

`src/gradient_boosting_learning_curve.py` contains tuned-estimator creation,
learning-curve computation, CSV persistence, and PDF visualization.

`scripts/run_gradient_boosting_learning_curve.py` handles dataset loading, the
shared split, fingerprint verification, one learning-curve computation, CSV
saving, PDF saving, and terminal reporting.

## Generated artifacts

The following artifacts were generated locally:

- `reports/tables/M04-HGB-learning-curve.csv`
- `reports/figures/M04-HGB-learning-curve.pdf`

## Reproducibility / leakage prevention

- Shared `random_state=42`.
- Common stratified CV.
- Verified shared split.
- Only training data used.
- Final test partition remained reserved.
- No final-test prediction or score was produced.
- The learning curve did not alter the selected HGB hyperparameters.

## Next planned work

Further training-only model evaluation/diagnostics will follow before
group-level model selection and final-test evaluation.

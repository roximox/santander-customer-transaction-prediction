# Logbook Entry
 
## Metadata
 
**Date:** 2026-08-14
**Member:** Chaymae Akouaouch
**Phase:** Model optimization and generalization analysis
**Ticket ID:** M04-HGB-LC-001
**Branch:** feature/model-optimization
**Related meeting:** 2026-08-09 — Data Processing, Validation Strategy and Start of Individual Analysis
 
## Title
 
HistGradientBoosting Learning-Curve Analysis
 
## Context and Goal
 
Following the validation strategy the group agreed on at the meeting on 2026-08-09, I continued the Member 04 model-optimization track after the initial HistGradientBoosting baseline and the randomized hyperparameter search.
 
The goal here was to see how the tuned HistGradientBoosting configuration behaves as the amount of training data increases — specifically, whether validation performance keeps improving with more data and whether the train-validation gap narrows.
 
The learning curve was purely a diagnostic step, not another round of hyperparameter tuning.
 
## Tuned Model
 
The analysis used the best configuration identified earlier by `M04-HGB-SEARCH-001`:
 
- `learning_rate=0.05`
- `max_iter=700`
- `max_leaf_nodes=31`
- `min_samples_leaf=100`
- `l2_regularization=10.0`
- `random_state=42`
I kept this configuration fixed throughout, so the learning-curve results were never used to change the selected hyperparameters.
 
## Method
 
I built the learning-curve workflow on top of the shared project validation infrastructure. The analysis:
 
- recreated the common 80/20 train/test split;
- verified the shared train and reserved-test fingerprints;
- used only the development partition for the learning-curve computation;
- reused the common five-fold stratified cross-validation;
- evaluated ROC-AUC and Average Precision;
- used `n_jobs=1`;
- evaluated training fractions of 10%, 25%, 50%, 75%, and 100%.
Since each cross-validation training fold contains 80% of the 160,000-row development partition, these fractions correspond to effective training sizes of 12,800, 32,000, 64,000, 96,000, and 128,000 observations.
 
## Work Performed
 
1. Reused the tuned HistGradientBoosting configuration identified by `M04-HGB-SEARCH-001`.
2. Implemented the learning-curve computation in `src/gradient_boosting_learning_curve.py`.
3. Created `scripts/run_gradient_boosting_learning_curve.py` for the reproducible analysis workflow.
4. Verified the shared split and dataset fingerprints.
5. Computed five-fold learning curves for ROC-AUC and Average Precision at five different training sizes.
6. Compared training and validation performance as the available training data increased.
7. Examined how the train-validation gaps changed with increasing data.
8. Compared the maximum-training-size results with the previous randomized search as a consistency check.
9. Saved the numerical results and learning-curve visualization.
10. Confirmed that the learning-curve analysis left the selected hyperparameters unchanged and did not use the reserved final-test partition for scoring.
## Results
 
| Train size | Train ROC-AUC | Validation ROC-AUC | ROC-AUC gap | Train AP | Validation AP | AP gap |
|---:|---:|---:|---:|---:|---:|---:|
| 12,800 | 0.989269 | 0.851063 | 0.138206 | 0.968960 | 0.475333 | 0.493628 |
| 32,000 | 0.989880 | 0.877188 | 0.112692 | 0.967501 | 0.550490 | 0.417011 |
| 64,000 | 0.983864 | 0.885981 | 0.097883 | 0.936334 | 0.574884 | 0.361450 |
| 96,000 | 0.980122 | 0.889333 | 0.090789 | 0.917675 | 0.584480 | 0.333195 |
| 128,000 | 0.973342 | 0.891245 | 0.082098 | 0.887635 | 0.589899 | 0.297736 |
 
 
Validation ROC-AUC rose from 0.851063 at the smallest effective training size to 0.891245 at the largest, and validation Average Precision rose from 0.475333 to 0.589899 over the same range.
 
At the same time, the ROC-AUC train-validation gap narrowed from 0.138206 to 0.082098, while the Average Precision gap dropped from 0.493628 to 0.297736.
 
## Interpretation and Decision
 
The learning curve showed that more training data was associated with better validation performance for both ROC-AUC and Average Precision, and the shrinking train-validation gaps pointed to improved generalization as well.
 
That said, a gap remained even at the largest training size, particularly for Average Precision. I do not interpret this as proof that overfitting has disappeared — rather, it's evidence that the tuned model benefits from more data while still needing further diagnostic evaluation.
 
As a consistency check, I compared the maximum-training-size learning-curve results against the earlier randomized-search results:
 
| Metric | Randomized-search best | Learning curve at maximum training size |
|---|---:|---:|
| Validation ROC-AUC | 0.891449 | 0.891245 |
| Average Precision | 0.591089 | 0.589899 |
| ROC-AUC generalization gap | 0.082131 | 0.082098 |
 
 
The two sets of numbers line up closely, which supports the consistency of both analyses.
 
Based on this, I kept the selected HGB configuration unchanged and moved on to additional training-only evaluation and diagnostics instead of starting another tuning round.
 
## Difficulties and Observations
 
Interpreting the learning curve required some care, since improved validation performance and a smaller train-validation gap don't automatically prove that overfitting has been eliminated.
 
Average Precision showed a noticeably larger train-validation gap than ROC-AUC, which reinforced the need to evaluate the model with metrics that reflect the imbalanced target rather than relying on ROC-AUC alone.
 
It was also important to keep this analysis separate from hyperparameter selection. The configuration had already been chosen by `M04-HGB-SEARCH-001`, so the learning-curve results were not used to tune the model again.
 
## Implementation
 
`src/gradient_boosting_learning_curve.py` contains the tuned-estimator creation, learning-curve computation, CSV persistence, and PDF visualization.
 
`scripts/run_gradient_boosting_learning_curve.py` handles dataset loading, shared-split reconstruction, fingerprint verification, learning-curve execution, artifact persistence, and terminal reporting.
 
## Generated Artifacts
 
- `reports/tables/M04-HGB-learning-curve.csv`
- `reports/figures/M04-HGB-learning-curve.pdf`
## Reproducibility
 
The analysis used the shared `random_state=42`, common stratified cross-validation, and the verified development/test split.
 
Only development data entered the learning-curve computation. The reserved final-test partition was not used for model fitting, hyperparameter selection, or scoring.
 
The selected HGB hyperparameters remained unchanged after this diagnostic analysis.
 
## Next Step
 
Perform additional training-only diagnostics using out-of-fold predictions to evaluate the tuned HistGradientBoosting model beyond aggregate cross-validation scores.
 

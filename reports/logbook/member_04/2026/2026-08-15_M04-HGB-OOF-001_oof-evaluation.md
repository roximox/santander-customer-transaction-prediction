# Logbook Entry
 
## Metadata
 
**Date:** 2026-08-15
**Member:** Chaymae Akouaouch
**Phase:** Model evaluation and diagnostic analysis
**Ticket ID:** M04-HGB-OOF-001
**Branch:** feature/model-optimization
**Related meeting:** 2026-08-09 — Data Processing, Validation Strategy and Start of Individual Analysis

## Title
 
Tuned HistGradientBoosting out-of-fold evaluation
 
## Goal
 
Evaluate the tuned HistGradientBoosting model using training-only out-of-fold (OOF) predictions while keeping the reserved final-test partition outside model fitting, tuning, and scoring. OOF evaluation gives every training row a prediction from a model that was not fitted on that row.
 
## Evaluated model
 
The model configuration was frozen from `M04-HGB-SEARCH-001`:
 
- `learning_rate = 0.05`
- `max_iter = 700`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 100`
- `l2_regularization = 10.0`
- `random_state = 42`
No hyperparameter optimization was performed during OOF evaluation. The classification threshold stayed fixed at the standard value of 0.5, and no threshold optimization was performed.
 
## Method
 
- Reused the common project dataset loader.
- Recreated the common 80/20 train/reserved-test split.
- Verified the train fingerprint and reserved-test fingerprint.
- Removed the reserved test objects before evaluation.
- Evaluated only the 160,000-row training partition.
- Reused the common five-fold stratified CV with `n_jobs=1`.
- Generated OOF probabilities with one `cross_val_predict(..., method="predict_proba")` execution.
- Derived predicted classes from OOF positive-class probabilities with the fixed threshold of 0.5.
- Ensured every row was predicted by a model that did not train on that row.
- Reused the same OOF predictions for all metrics, ROC data, PR data, and confusion-matrix values.
## Results
 
| Metric | Value |
|---|---:|
| ROC-AUC | 0.891438 |
| Average Precision | 0.590860 |
| F1 | 0.415242 |
| Precision | 0.795527 |
| Recall | 0.280943 |
| Accuracy | 0.920488 |
| Balanced Accuracy | 0.636438 |
 
OOF prediction rows: 160000
ROC curve points: 19212
Precision-Recall curve points: 160001
 
## Confusion matrix
 
| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 142761 | 1161 |
| Actual Positive | 11561 | 4517 |
 
- True Negatives = 142761
- False Positives = 1161
- False Negatives = 11561
- True Positives = 4517
## Interpretation
 
An ROC-AUC of 0.891438 points to strong ranking performance on the training-only OOF predictions, with Average Precision at 0.590860. Precision is relatively high at 0.795527, while recall is considerably lower at 0.280943. At the unchanged 0.5 threshold, positive predictions are fairly precise, but a large share of actual positive cases go undetected — visible in the 11,561 false negatives against 4,517 true positives.
 
These results do not establish that the model is optimal, nor that 0.5 is the optimal threshold. No threshold tuning was performed in this step.
 
## Consistency with hyperparameter search
 
| Metric | M04-HGB-SEARCH-001 | M04-HGB-OOF-001 | Difference |
|---|---:|---:|---:|
| ROC-AUC | 0.891449 | 0.891438 | approximately 0.000011 |
| Average Precision | 0.591089 | 0.590860 | 0.000229 |
 
The very close values give a useful consistency check between the hyperparameter-search CV results and this dedicated OOF evaluation. That said, the OOF evaluation is not an independent final test, since both analyses draw on the training partition.
 
## Implementation
 
`src/gradient_boosting_evaluation.py` provides:
 
- Training-only OOF prediction generation.
- Aggregate metric computation.
- ROC numerical data.
- Precision-Recall numerical data.
- Confusion-matrix counts.
- Persistence helpers.
- ROC visualization.
- Precision-Recall visualization.
- Confusion-matrix visualization.
`scripts/run_gradient_boosting_evaluation.py` handles:
 
- Dataset loading.
- Shared split recreation.
- Fingerprint verification.
- One OOF prediction computation.
- Derivation of all diagnostics from the same predictions.
- Persistence of numerical artifacts.
- Persistence of figures.
- Terminal reporting.
I reused the common `validate_evaluation_inputs()` from `src/evaluation.py` and the common `save_figure()` from `src/visualization.py` for figure persistence. No Member 1 files were modified.
 
## Generated artifacts
 
The following artifacts were generated locally:
 
- `reports/tables/M04-HGB-OOF-001_predictions.csv`
- `reports/tables/M04-HGB-OOF-001_metrics.json`
- `reports/tables/M04-HGB-OOF-001_roc_curve.csv`
- `reports/tables/M04-HGB-OOF-001_precision_recall_curve.csv`
- `reports/figures/M04-HGB-OOF-001_roc_curve.pdf`
- `reports/figures/M04-HGB-OOF-001_precision_recall_curve.pdf`
- `reports/figures/M04-HGB-OOF-001_confusion_matrix.pdf`
Their Git tracking status was not verified in this entry.
 
## Reproducibility and leakage prevention
 
- Tuned model frozen before this evaluation.
- `random_state=42`.
- Common five-fold stratified CV.
- Verified shared train/test fingerprints.
- Only the training partition used for predictions and metrics.
- Reserved final-test partition was not evaluated.
- No final-test prediction was produced.
- No final-test metric was produced.
- Threshold remained 0.5.
- No threshold tuning.
- No hyperparameter tuning.
- No model selection was performed during this step.
## Next planned work
 
Compare the tuned HistGradientBoosting OOF results with the original M04-HGB-001 baseline to quantify the effect of the optimization across the reported evaluation metrics. This comparison will consolidate the Member 04 results before the tuned HGB candidate is handed over to the group-level model-selection process.
 
# Logbook Entry
 
## Metadata
 
**Date:** 2026-08-15
**Member:** Chaymae Akouaouch
**Phase:** Model evaluation and comparison
**Ticket ID:** M04-HGB-COMP-001
**Branch:** feature/model-optimization
**Related meeting:** 2026-08-09 — Data Processing, Validation Strategy and Start of Individual Analysis
 
## Title
 
HistGradientBoosting Baseline vs. Tuned Model Comparison
 
## Context and Goal
 
With the training-only OOF evaluation from `M04-HGB-OOF-001` already complete, I compared its tuned HistGradientBoosting results with the original `M04-HGB-001` baseline.
 
The goal was to quantify whether the optimization work actually improved the Member 04 HGB model across the common evaluation metrics, before handing the candidate over to the group-level model-selection process.
 
This step only compared results that already existed. It involved no new model training, hyperparameter tuning, or OOF prediction — the comparison script reads the stored baseline and OOF artifacts and computes the differences between them.
 
## Data Sources
 
The comparison reused existing artifacts:
 
- `reports/experiments/M04-HGB-001_summary.json`
- `reports/tables/M04-HGB-OOF-001_metrics.json`
- `reports/searches/M04-HGB-SEARCH-001_summary.json`
The baseline values are five-fold validation means; the tuned values come from metrics computed on the aggregated five-fold OOF predictions generated in `M04-HGB-OOF-001`.
 
Both sets of results are training-only estimates and do not represent final-test performance.
 
## Tuned Configuration
 
The tuned configuration had already been selected by `M04-HGB-SEARCH-001` and frozen for the OOF evaluation:
 
- `learning_rate=0.05`
- `max_iter=700`
- `max_leaf_nodes=31`
- `min_samples_leaf=100`
- `l2_regularization=10.0`
- `random_state=42`
No additional hyperparameter or threshold optimization was performed during this comparison — the configuration was carried over unchanged from the earlier steps.
 
## Work Performed
 
1. Loaded the existing `M04-HGB-001` baseline results.
2. Loaded the existing `M04-HGB-OOF-001` tuned-model metrics.
3. Validated the required source artifacts.
4. Built a common comparison table across seven evaluation metrics.
5. Calculated the absolute change from baseline to tuned HGB for each metric.
6. Checked whether the optimization produced consistent improvements.
7. Generated CSV and JSON comparison artifacts.
8. Created a visualization of the baseline-versus-tuned comparison.
9. Confirmed that this step required no dataset loading or model fitting — it worked entirely from the previously saved results.
10. Kept the reserved final-test partition outside this comparison.
## Results
 
| Metric | Baseline HGB | Tuned HGB OOF | Absolute change |
|---|---:|---:|---:|
| ROC-AUC | 0.884596 | 0.891438 | +0.006842 |
| Average Precision | 0.572879 | 0.590860 | +0.017981 |
| F1 | 0.387255 | 0.415242 | +0.027987 |
| Precision | 0.782671 | 0.795527 | +0.012856 |
| Recall | 0.257307 | 0.280943 | +0.023636 |
| Accuracy | 0.918181 | 0.920488 | +0.002307 |
| Balanced Accuracy | 0.624658 | 0.636438 | +0.011780 |
 
All seven reported metrics increased from the original baseline to the tuned HGB evaluation.
 
## Interpretation and Decision
 
The tuned HGB results showed improvements across all seven reported metrics.
 
ROC-AUC rose from 0.884596 to 0.891438, Average Precision from 0.572879 to 0.590860, and F1 from 0.387255 to 0.415242, with both precision and recall improving as well.
 
The gains in Average Precision, F1, recall, and Balanced Accuracy matter in particular because the project dataset has an imbalanced target, where accuracy alone would not provide a sufficient picture of model quality.
 
Still, this comparison needs to be read carefully. The baseline metrics are averages across five validation folds, while the tuned metrics come from aggregated OOF predictions computed in the earlier `M04-HGB-OOF-001` step. The two procedures are closely related but not identical, so I treated this as evidence that tuning improved the Member 04 HGB candidate — not as an independent final-test comparison.
 
Based on these results, I considered the tuned HGB configuration ready to hand over as the Member 04 candidate for the later group-level model comparison. No group-level final-model decision was made at this stage.
 
## Difficulties and Observations
 
The main methodological point was making sure the baseline and tuned results were interpreted appropriately despite their slightly different aggregation procedures.
 
Since the baseline uses mean fold-level metrics while the tuned evaluation uses metrics from aggregated OOF predictions, I avoided reading small numerical differences as an exact like-for-like benchmark.
 
It was also worth keeping two things separate: improving my own HGB baseline versus selecting the best model for the whole project. This comparison supports the first, but the group-level model selection still needs to weigh the other project candidates.
 
## Implementation
 
`src/gradient_boosting_comparison.py` contains the reusable logic for loading and validating the existing Member 04 results, building the comparison table, computing absolute metric changes, saving the numerical outputs, and creating the comparison figure.
 
`scripts/run_gradient_boosting_comparison.py` is the lightweight runner that reads the existing result artifacts, builds the comparison, persists the outputs, and reports the metric changes.
 
No Santander dataset loading, model fitting, cross-validation, OOF prediction, hyperparameter search, or threshold optimization happened during this comparison — everything was read from the artifacts produced in the earlier steps.
 
I reused the common `save_figure()` helper from `src/visualization.py` without modifying the shared file.
 
## Generated Artifacts
 
- `reports/tables/M04-HGB-model-comparison.csv`
- `reports/tables/M04-HGB-model-comparison.json`
- `reports/figures/M04-HGB-model-comparison.pdf`
## Reproducibility and Leakage Prevention
 
The comparison was based entirely on previously generated training-only results.
 
No new dataset loading, model training, cross-validation, OOF prediction, hyperparameter tuning, threshold tuning, or model selection was performed in this step.
 
The reserved final-test partition was not used for this comparison or for selecting the Member 04 candidate.
 
## Next Step
 
Verify and consolidate the Member 04 implementation, generated artifacts, tests, and documentation before handing the tuned HistGradientBoosting candidate over to the group-level model-selection process.
 

# Member 04 Logbook

## Introduction

This logbook documents the work completed by Member 04 as part of the
Santander Customer Transaction Prediction project.

Member 04 is responsible for the Gradient Boosting model-development track,
with a focus on building, tuning, evaluating, and documenting a
HistGradientBoosting classification model.

The work follows the common project methodology and uses only the training
partition for model development and model selection. The reserved final test
partition remains untouched until the group-level final model selection is
completed.

## Member information

- **Member:** Chaymae Akouaouch
- **Member ID:** Member 04
- **Branch:** `feature/model-optimization`
- **Main responsibility:** Gradient Boosting model development and optimization

## Main responsibilities

Member 04's main responsibilities are:

- Implement a reproducible HistGradientBoosting baseline.
- Evaluate the baseline using the common project cross-validation framework.
- Perform controlled hyperparameter tuning on the training data only.
- Select and register the tuned HistGradientBoosting configuration.
- Analyze model learning behavior using learning curves.
- Perform training-only out-of-fold (OOF) evaluation.
- Generate ROC, Precision-Recall, and confusion-matrix diagnostics.
- Compare the baseline and tuned HistGradientBoosting models.
- Document experiments, results, limitations, and reproducibility decisions.
- Prepare the tuned HistGradientBoosting model as Member 04's candidate for
  later group-level model selection.
- Keep the reserved final test partition untouched during Member 04 model
  development.

## Machine Learning responsibility

The primary Machine Learning responsibility of Member 04 is
**HistGradientBoosting classification and model optimization**.

The workflow consists of:

`Baseline → Hyperparameter Search → Tuned Experiment → Learning Curve → OOF Evaluation → Model Comparison`

The main evaluation metric is ROC-AUC, supported by Average Precision, F1,
Precision, Recall, Accuracy, and Balanced Accuracy.

Member 04 does not independently select the final project model. The tuned
HistGradientBoosting model is handed over as a candidate for the common
group-level model-selection stage.

## Entry naming convention

Logbook entries follow this naming convention:

`YYYY-MM-DD_<TICKET-ID>_<short-description>.md`

For example:

`2026-08-10_M04-HGB-001_hist-gradient-boosting-baseline.md`

Member 04 ticket identifiers use the following structure:

`M04-HGB-<STAGE>-<NUMBER>`

Examples used in this work include:

- `M04-HGB-001` — HistGradientBoosting baseline
- `M04-HGB-SEARCH-001` — Hyperparameter search
- `M04-HGB-LC-001` — Learning-curve analysis
- `M04-HGB-OOF-001` — Out-of-fold evaluation
- `M04-HGB-COMP-001` — Baseline vs tuned comparison
- `M04-HGB-002` — Registered tuned HistGradientBoosting experiment

## Chronological index

| Date | Ticket ID | Title | Logbook file | Status |
|---|---|---|---|---|
| 2026-08-10 | M04-HGB-001 | HistGradientBoosting baseline | [2026-08-10_M04-HGB-001_hist-gradient-boosting-baseline.md](2026/2026-08-10_M04-HGB-001_hist-gradient-boosting-baseline.md) | Completed |
| 2026-08-14 | M04-HGB-SEARCH-001 | Hyperparameter tuning | [2026-08-14_M04-HGB-SEARCH-001_hyperparameter-tuning.md](2026/2026-08-14_M04-HGB-SEARCH-001_hyperparameter-tuning.md) | Completed |
| 2026-08-14 | M04-HGB-LC-001 | Learning-curve analysis | [2026-08-14_M04-HGB-LC-001_learning-curve.md](2026/2026-08-14_M04-HGB-LC-001_learning-curve.md) | Completed |
| 2026-08-15 | M04-HGB-OOF-001 | Training-only OOF evaluation | [2026-08-15_M04-HGB-OOF-001_oof-evaluation.md](2026/2026-08-15_M04-HGB-OOF-001_oof-evaluation.md) | Completed |
| 2026-08-15 | M04-HGB-COMP-001 | Baseline vs tuned model comparison | [2026-08-15_M04-HGB-COMP-001_model-comparison.md](2026/2026-08-15_M04-HGB-COMP-001_model-comparison.md) | Completed |
| 2026-08-22 | M04-HGB-002 | Registered tuned HistGradientBoosting experiment | [2026-08-22_M04-HGB-002_tuned-hist-gradient-boosting.md](2026/2026-08-22_M04-HGB-002_tuned-hist-gradient-boosting.md) | Completed |

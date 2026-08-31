# Member 04 Logbook
 
## Introduction
 
This logbook documents Chaymae Akouaouch's work as Member 04 on the Santander Customer Transaction Prediction project.
 
Member 04's main responsibility was the HistGradientBoosting modeling and optimization track. Throughout this work and the candidate selection that followed, the reserved final-test partition stayed untouched. It was opened only after the group collectively froze `M04-HGB-002` on 2026-08-26; the final-evaluation workflow itself was implemented and recorded separately by Member 01.
 
## Member Information
 
- **Member:** Chaymae Akouaouch
- **Member ID:** Member 04
- **Branch:** `feature/model-optimization`
- **Main responsibility:** HistGradientBoosting model development and optimization
## Main Responsibilities
 
- Build and evaluate a reproducible HistGradientBoosting baseline using the shared cross-validation framework.
- Run a controlled, training-only hyperparameter search and preserve the selected configuration.
- Produce learning-curve, training-only OOF, ROC, Precision-Recall, and confusion-matrix diagnostics.
- Compare the baseline and tuned HGB evidence, formally register the tuned experiment, and consolidate the reporting notebook.
- Document limitations, reproducibility controls, and the relationship between the search-selected configuration and its later formal registration.
- Take part in the collective final-model review and confirmation, supported by the completed Member 04 HGB evidence.
## Machine Learning Responsibility
 
Member 04's core Machine Learning responsibility was **HistGradientBoosting classification and model optimization**. The main evaluation metric is ROC-AUC, supported by Average Precision, F1, Precision, Recall, Accuracy, and Balanced Accuracy.
 
### Experimental Chronology
 
`M04-HGB-001 → M04-HGB-SEARCH-001 → M04-HGB-LC-001 → M04-HGB-OOF-001 → M04-HGB-COMP-001 → M04-HGB-002 → M04-NB-001 → M04-SEL-001`
 
The hyperparameter search selected and froze the configuration before the learning-curve, OOF, and comparison analyses ran. `M04-HGB-002` later formally registered that same, unchanged configuration through the common experiment framework — it was not another tuning stage.
 
Member 04 took part in the collective selection of `M04-HGB-002`, but did not individually implement the group model-selection framework, the final model lock, dashboard integration, or the final-test workflow.
 
## Entry Naming Convention
 
Logbook files use:
 
`YYYY-MM-DD_<TICKET-ID>_<short-description>.md`
 
Member 04 ticket IDs use the `M04-...` prefix and a task-family identifier where applicable. The current families are:
 
- `HGB` — HistGradientBoosting experiments and diagnostics
- `NB` — notebook consolidation
- `SEL` — collective model-selection participation
## Chronological Index
 
| Date | Ticket ID | Title | Logbook file | Status |
|---|---|---|---|---|
| 2026-08-10 | M04-HGB-001 | HistGradientBoosting Baseline | [2026-08-10_M04-HGB-001_hist-gradient-boosting-baseline.md](2026/2026-08-10_M04-HGB-001_hist-gradient-boosting-baseline.md) | Completed |
| 2026-08-14 | M04-HGB-SEARCH-001 | HistGradientBoosting Hyperparameter Optimization | [2026-08-14_M04-HGB-SEARCH-001_hyperparameter-tuning.md](2026/2026-08-14_M04-HGB-SEARCH-001_hyperparameter-tuning.md) | Completed |
| 2026-08-14 | M04-HGB-LC-001 | HistGradientBoosting Learning-Curve Analysis | [2026-08-14_M04-HGB-LC-001_learning-curve.md](2026/2026-08-14_M04-HGB-LC-001_learning-curve.md) | Completed |
| 2026-08-15 | M04-HGB-OOF-001 | Tuned HistGradientBoosting Out-of-Fold Evaluation | [2026-08-15_M04-HGB-OOF-001_oof-evaluation.md](2026/2026-08-15_M04-HGB-OOF-001_oof-evaluation.md) | Completed |
| 2026-08-15 | M04-HGB-COMP-001 | HistGradientBoosting Baseline vs. Tuned Model Comparison | [2026-08-15_M04-HGB-COMP-001_model-comparison.md](2026/2026-08-15_M04-HGB-COMP-001_model-comparison.md) | Completed |
| 2026-08-22 | M04-HGB-002 | Registered Tuned HistGradientBoosting Experiment | [2026-08-22_M04-HGB-002_tuned-hist-gradient-boosting.md](2026/2026-08-22_M04-HGB-002_tuned-hist-gradient-boosting.md) | Completed |
| 2026-08-24 | M04-NB-001 | Gradient Boosting Analysis Notebook Consolidation | [2026-08-24_M04-NB-001_gradient-boosting-analysis-notebook-consolidation.md](2026/2026-08-24_M04-NB-001_gradient-boosting-analysis-notebook-consolidation.md) | Completed |
| 2026-08-26 | M04-SEL-001 | Collective Final-Model Selection Participation and Confirmation | [2026-08-26_M04-SEL-001_collective-final-model-selection.md](2026/2026-08-26_M04-SEL-001_collective-final-model-selection.md) | Completed |
 